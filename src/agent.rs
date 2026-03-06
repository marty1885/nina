use crate::channel::CallContext;
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, ToolCallInfo};
use crate::memory::MemoryStore;
use crate::session::SessionManager;
use crate::tools::ToolRegistry;
use anyhow::Result;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

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
        let memories = self.memory.recall(user_text, ctx.identity_id).await.unwrap_or_else(|_| String::new());

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

            let (messages, repaired) = sanitize_messages(self.sessions.get_messages(session_key).await);
            if repaired {
                self.sessions.repair_messages(session_key, messages.clone()).await;
            }
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

                    // Execute tools BEFORE writing to the session so we can push
                    // assistant + results in one atomic operation.  This prevents
                    // a concurrent branch from interleaving a user message between
                    // the assistant(tool_calls) and its results, which would produce
                    // "tool message has no corresponding toolcall" API errors.
                    let results = execute_tools_parallel(&self.tools, &tool_calls, ctx).await;

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

                    // Atomic: assistant(tool_calls) + all results land together.
                    self.sessions
                        .push_tool_turn(
                            session_key,
                            ChatMessage {
                                role: "assistant".into(),
                                content,
                                tool_calls: Some(tool_calls.clone()),
                                tool_call_id: None,
                            },
                            tool_messages,
                        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_debug_short_string() {
        assert_eq!(truncate_debug("hello", 10), "hello");
    }

    #[test]
    fn truncate_debug_exact_length() {
        assert_eq!(truncate_debug("hello", 5), "hello");
    }

    #[test]
    fn truncate_debug_long_string() {
        assert_eq!(truncate_debug("hello world", 5), "hello");
    }

    #[test]
    fn truncate_debug_empty() {
        assert_eq!(truncate_debug("", 10), "");
    }

    #[test]
    fn truncate_debug_unicode_boundary() {
        // "日本語" — 3 chars, each 3 bytes; truncate at char 2 should not panic
        let s = "日本語test";
        let result = truncate_debug(s, 2);
        assert_eq!(result, "日本");
    }

    #[test]
    fn is_tool_error_tool_error_prefix() {
        assert!(is_tool_error("Tool error: something failed"));
    }

    #[test]
    fn is_tool_error_invalid_arguments() {
        assert!(is_tool_error("Invalid arguments: bad json"));
    }

    #[test]
    fn is_tool_error_unknown_tool() {
        assert!(is_tool_error("Unknown tool: foo"));
    }

    #[test]
    fn is_tool_error_false_for_success() {
        assert!(!is_tool_error("OK"));
        assert!(!is_tool_error("some result"));
        assert!(!is_tool_error(""));
    }

    #[test]
    fn is_tool_error_false_for_partial_prefix() {
        assert!(!is_tool_error("tool error: lowercase prefix"));
    }
}

/// Repair a message list that may have gotten out of sync (crash or concurrent-branch race).
///
/// Invariant enforced: every `tool` result must have its owning `assistant(tool_calls)`
/// as its nearest non-tool predecessor, and the assistant must declare that tool_call_id.
/// Any block that violates this (wrong ordering OR missing IDs) is dropped entirely.
/// Returns the sanitized message list and a bool indicating whether anything was dropped.
fn sanitize_messages(msgs: Vec<ChatMessage>) -> (Vec<ChatMessage>, bool) {
    let n = msgs.len();
    let mut keep = vec![true; n];

    let mut i = 0;
    while i < n {
        if msgs[i].role != "assistant" {
            // Bare tool message with no preceding assistant — orphaned.
            if msgs[i].role == "tool" && keep[i] {
                warn!(position = i, "Dropping orphaned tool message (no preceding assistant)");
                keep[i] = false;
            }
            i += 1;
            continue;
        }

        // Only care about assistant messages that carry tool calls.
        let tcs = match msgs[i].tool_calls.as_ref().filter(|tc| !tc.is_empty()) {
            Some(tc) => tc,
            None => { i += 1; continue; }
        };

        let expected: std::collections::HashSet<&str> =
            tcs.iter().map(|tc| tc.id.as_str()).collect();

        // Collect the run of tool messages immediately following this assistant.
        let block_start = i + 1;
        let mut j = block_start;
        while j < n && msgs[j].role == "tool" {
            j += 1;
        }
        let block_end = j; // exclusive

        // Check: every consecutive tool result must match an expected ID, and
        // every expected ID must appear exactly once in the block.
        let consecutive_ids: std::collections::HashSet<&str> = msgs[block_start..block_end]
            .iter()
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect();

        if consecutive_ids != expected {
            // Block is incomplete or has unexpected IDs → drop it.
            // This covers: crash (missing results) and race (interleaved user message
            // pushed the results out of the consecutive window).
            warn!(
                position = i,
                expected = expected.len(),
                found = consecutive_ids.len(),
                "Dropping invalid tool block (crash or concurrent-branch race recovery)"
            );
            keep[i] = false;
            for k in block_start..block_end {
                keep[k] = false;
            }
            // Also remove any stray results for these IDs that appear later in history.
            for k in block_end..n {
                if msgs[k].role == "tool" {
                    if msgs[k].tool_call_id.as_deref().is_some_and(|id| expected.contains(id)) {
                        keep[k] = false;
                    }
                }
            }
        }

        i = block_end; // jump past the block
    }

    let dropped = keep.iter().filter(|&&k| !k).count();
    if dropped > 0 {
        warn!(dropped, "Sanitizer removed invalid messages before LLM call");
    }

    let cleaned = msgs.into_iter()
        .zip(keep)
        .filter_map(|(m, k)| if k { Some(m) } else { None })
        .collect();
    (cleaned, dropped > 0)
}

async fn execute_tools_parallel(tools: &ToolRegistry, calls: &[ToolCallInfo], ctx: &CallContext) -> Vec<String> {
    let futures: Vec<_> = calls
        .iter()
        .map(|tc| tools.call(&tc.function.name, &tc.function.arguments, ctx))
        .collect();
    futures::future::join_all(futures).await
}
