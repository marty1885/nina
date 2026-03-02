use crate::channel::CallContext;
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, ToolCallInfo};
use crate::memory::MemoryStore;
use crate::session::SessionManager;
use crate::tools::ToolRegistry;
use anyhow::Result;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info};

const MAX_ITERATIONS: usize = 15;
const MAX_MESSAGES: usize = 80;
const ERROR_BURST_THRESHOLD: usize = 5;

pub struct Agent {
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    sessions: Arc<SessionManager>,
    memory: Arc<MemoryStore>,
    debug: bool,
}

impl Agent {
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        tools: Arc<ToolRegistry>,
        sessions: Arc<SessionManager>,
        memory: Arc<MemoryStore>,
        debug: bool,
    ) -> Self {
        Self {
            llm,
            tools,
            sessions,
            memory,
            debug,
        }
    }

    /// Process a user message and return the assistant's final text response.
    ///
    /// RAG context is prepended to the user message in-memory only; the original
    /// text is what gets persisted to the DB so that recall is rebuilt fresh each turn.
    ///
    /// `on_interim_text` is called (and awaited) with any text the model emits
    /// alongside tool calls, so it can be sent to the user *before* the tools
    /// (e.g. TTS audio) execute, preserving the correct display order.
    ///
    /// `absorb_rx` receives messages that were classified as related to this
    /// branch's processing. When a message is absorbed, the current text response
    /// is suppressed and the loop continues so the agent can incorporate it.
    ///
    /// Returns `None` if all responses were absorbed/suppressed, `Some(text)` otherwise.
    pub async fn process_message<F, Fut>(
        &self,
        session_key: &str,
        user_text: &str,
        display_text: &str,
        on_interim_text: F,
        absorb_rx: &mut mpsc::Receiver<String>,
        ctx: &CallContext,
    ) -> Result<Option<String>>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: Future<Output = ()> + Send,
    {
        // --- RAG: recall relevant memories and prepend to in-session user message ---
        let memories = self.memory.recall(user_text).await.unwrap_or_else(|_| String::new());

        let no_memories = memories.is_empty() || memories == "No memories found.";

        if self.debug && !no_memories {
            info!("[debug] RAG injected memories for: {}", user_text);
        }

        let user_content = if no_memories {
            display_text.to_string()
        } else {
            format!("[Relevant context from memory:\n{}]\n\n{}", memories, display_text)
        };

        // Persist original text; push RAG-augmented content to the in-memory session.
        self.sessions
            .push_user_message(session_key, user_text, &user_content)
            .await;

        let tool_defs = self.tools.definitions().await;
        let mut consecutive_error_iterations: usize = 0;

        for iteration in 0..MAX_ITERATIONS {
            debug!(iteration, session_key, "Agent iteration");

            // Check for absorbed messages between iterations (non-blocking).
            // We inject them here — at a safe boundary after all tool results
            // have been pushed — so we never break message ordering.
            if iteration > 0 {
                let mut absorbed_any = false;
                while let Ok(absorbed) = absorb_rx.try_recv() {
                    absorbed_any = true;
                    info!(
                        session_key,
                        "Absorbed related message into processing loop: {}",
                        truncate_debug(&absorbed, 100)
                    );
                    // Persist and push to session at this safe boundary.
                    self.sessions
                        .push_user_message(
                            session_key,
                            &absorbed,
                            &format!(
                                "[IMPORTANT — The user just sent a follow-up correction/addition while you were still working. \
                                 You MUST address this new message. If it contradicts your previous direction, STOP and change course.]\n\n{}",
                                absorbed
                            ),
                        )
                        .await;
                }
                if absorbed_any {
                    // Force a fresh LLM call that will see the correction
                    // at the end of the message history, in the right place.
                }
            }

            let messages = self.sessions.get_messages(session_key).await;
            let response = self.llm.complete(&messages, &tool_defs).await?;

            match response {
                LlmResponse::Text(text) => {
                    if self.debug {
                        info!("[debug] Assistant response:\n{text}");
                    }

                    // Before returning, check if a message was absorbed while we were
                    // waiting for the LLM. If so, suppress this response and continue
                    // so the agent can address the correction.
                    if let Ok(absorbed) = absorb_rx.try_recv() {
                        info!(
                            session_key,
                            "Message absorbed after LLM response, suppressing and continuing: {}",
                            truncate_debug(&absorbed, 100)
                        );
                        // Push the assistant text to session (useful context),
                        // then inject the correction at this safe boundary.
                        self.sessions
                            .push_message(
                                session_key,
                                ChatMessage {
                                    role: "assistant".into(),
                                    content: Some(text),
                                    tool_calls: None,
                                    tool_call_id: None,
                                },
                            )
                            .await;

                        // Inject the absorbed correction (and any others that queued up)
                        self.sessions
                            .push_user_message(
                                session_key,
                                &absorbed,
                                &format!(
                                    "[IMPORTANT — The user just sent a follow-up correction/addition while you were still working. \
                                     You MUST address this new message. If it contradicts your previous direction, STOP and change course.]\n\n{}",
                                    absorbed
                                ),
                            )
                            .await;

                        while let Ok(extra) = absorb_rx.try_recv() {
                            self.sessions
                                .push_user_message(
                                    session_key,
                                    &extra,
                                    &format!(
                                        "[IMPORTANT — Additional follow-up from the user.]\n\n{}",
                                        extra
                                    ),
                                )
                                .await;
                        }

                        continue;
                    }

                    self.sessions
                        .push_message(
                            session_key,
                            ChatMessage {
                                role: "assistant".into(),
                                content: Some(text.clone()),
                                tool_calls: None,
                                tool_call_id: None,
                            },
                        )
                        .await;

                    // Trim after getting a text response
                    self.sessions.trim(session_key, MAX_MESSAGES).await;

                    return Ok(Some(text));
                }
                LlmResponse::ToolCalls(content, tool_calls) => {
                    info!(count = tool_calls.len(), "Executing tool calls in parallel");

                    if self.debug {
                        for tc in &tool_calls {
                            info!(
                                "[debug] Tool call: {}({})",
                                tc.function.name, tc.function.arguments
                            );
                        }
                    }

                    // Send any text the model wrote before the tool calls so it
                    // arrives before tool side-effects (e.g. TTS audio).
                    if let Some(ref text) = content {
                        if self.debug {
                            info!("[debug] Interim text before tool calls:\n{text}");
                        }
                        on_interim_text(text.clone()).await;
                    }

                    // Push the assistant message with tool calls (+ any content)
                    self.sessions
                        .push_message(
                            session_key,
                            ChatMessage {
                                role: "assistant".into(),
                                content: content,
                                tool_calls: Some(tool_calls.clone()),
                                tool_call_id: None,
                            },
                        )
                        .await;

                    // Execute all tool calls concurrently
                    let results = execute_tools_parallel(&self.tools, &tool_calls, ctx).await;

                    // Push all tool results
                    let tool_messages: Vec<ChatMessage> = tool_calls
                        .iter()
                        .zip(results.iter())
                        .map(|(tc, result)| {
                            if self.debug {
                                info!(
                                    "[debug] Tool result [{}]: {}",
                                    tc.function.name,
                                    truncate_debug(&result, 500)
                                );
                            }
                            debug!(
                                tool = %tc.function.name,
                                result_len = result.len(),
                                "Tool result"
                            );
                            ChatMessage::tool_result(&tc.id, result.clone())
                        })
                        .collect();

                    self.sessions
                        .push_messages(session_key, tool_messages)
                        .await;

                    // Burst guard: track iterations where every tool call failed.
                    // A mix of successes and failures means progress is being made.
                    let all_failed = results.iter().all(|r| is_tool_error(r));
                    if all_failed {
                        consecutive_error_iterations += 1;
                        if consecutive_error_iterations >= ERROR_BURST_THRESHOLD {
                            consecutive_error_iterations = 0;
                            self.sessions
                                .push_user_message(
                                    session_key,
                                    "[burst-guard]",
                                    "[AUTOMATED SYSTEM MESSAGE — NOT FROM THE USER] \
                                     You have failed on every tool call for the past 5 iterations. \
                                     Stop and reconsider your approach from scratch. \
                                     Think about why each attempt failed, whether you have the right \
                                     information to proceed, and whether you should ask the user for \
                                     clarification or additional input before continuing.",
                                )
                                .await;
                        }
                    } else {
                        consecutive_error_iterations = 0;
                    }
                }
            }
        }

        anyhow::bail!("Max tool iterations ({MAX_ITERATIONS}) reached")
    }

    /// Clear session history for the given key (e.g. after an isolated reminder completes).
    pub async fn reset_session(&self, key: &str) {
        self.sessions.reset(key).await;
    }
}

/// Returns true if a tool result string indicates a failure.
/// Matches the error prefixes produced by `tools::dispatch` and `ToolRegistry::call`.
fn is_tool_error(result: &str) -> bool {
    result.starts_with("Tool error: ")
        || result.starts_with("Invalid arguments: ")
        || result.starts_with("Unknown tool: ")
}

fn truncate_debug(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

async fn execute_tools_parallel(tools: &ToolRegistry, calls: &[ToolCallInfo], ctx: &CallContext) -> Vec<String> {
    let futures: Vec<_> = calls
        .iter()
        .map(|tc| tools.call(&tc.function.name, &tc.function.arguments, ctx))
        .collect();
    futures::future::join_all(futures).await
}
