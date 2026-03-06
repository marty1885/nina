use crate::llm::{ChatMessage, LlmProvider, LlmResponse};
use crate::session::SessionManager;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

/// Entry in the sent-message log.
struct SentEntry {
    counter: u64,
    branch_id: u64,
    text: String,
}

/// Per-session coordinator that manages concurrent processing branches
/// and merges outgoing messages for coherence.
pub struct SessionOrchestrator {
    /// Monotonic counter for sent messages
    sent_counter: AtomicU64,
    /// Log of messages actually sent to user
    sent_log: Mutex<Vec<SentEntry>>,
    /// Active branch states: branch_id → last_merged_counter
    branches: Mutex<HashMap<u64, u64>>,
    /// Next branch ID
    next_branch: AtomicU64,
    /// Channel for absorbing related messages into active branch
    absorb_channels: Mutex<HashMap<u64, mpsc::Sender<String>>>,
    /// The LLM provider used for classification and merging (cheap/fast model)
    gateway_llm: Arc<dyn LlmProvider>,
}

impl SessionOrchestrator {
    pub fn new(gateway_llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            sent_counter: AtomicU64::new(0),
            sent_log: Mutex::new(Vec::new()),
            branches: Mutex::new(HashMap::new()),
            next_branch: AtomicU64::new(0),
            absorb_channels: Mutex::new(HashMap::new()),
            gateway_llm,
        }
    }

    /// Start a new processing branch. Returns the branch ID and a receiver
    /// for absorbed (related) messages.
    pub async fn start_branch(&self) -> (u64, mpsc::Receiver<String>) {
        let branch_id = self.next_branch.fetch_add(1, Ordering::SeqCst);
        let current_counter = self.sent_counter.load(Ordering::SeqCst);

        self.branches
            .lock()
            .await
            .insert(branch_id, current_counter);

        let (tx, rx) = mpsc::channel(16);
        self.absorb_channels.lock().await.insert(branch_id, tx);

        debug!(branch_id, "Started new branch");
        (branch_id, rx)
    }

    /// Clean up a branch after processing completes.
    pub async fn end_branch(&self, branch_id: u64) {
        self.branches.lock().await.remove(&branch_id);
        self.absorb_channels.lock().await.remove(&branch_id);
        debug!(branch_id, "Ended branch");
    }

    /// Absorb a related message into an active branch.
    /// Sends the message through the absorb channel so the agent can inject it
    /// at a safe boundary in the conversation (not mid-tool-call).
    /// Returns true if absorption succeeded.
    pub async fn absorb_message(
        &self,
        branch_id: u64,
        message: String,
        _sessions: &SessionManager,
        _session_key: &str,
    ) -> bool {
        // NOTE: We intentionally do NOT push to the session here.
        // The agent loop is responsible for injecting absorbed messages at safe
        // boundaries (after tool results, before the next LLM call) so we don't
        // break message ordering (e.g. user message between tool_calls and tool results).

        let channels = self.absorb_channels.lock().await;
        if let Some(tx) = channels.get(&branch_id) {
            match tx.send(message).await {
                Ok(()) => {
                    info!(branch_id, "Absorbed related message into active branch");
                    true
                }
                Err(_) => {
                    warn!(branch_id, "Absorb channel closed");
                    false
                }
            }
        } else {
            warn!(branch_id, "No absorb channel found for branch");
            false
        }
    }

    /// Send a message or merge it with context from other branches.
    /// Returns the text that was actually sent (after merge), or None if
    /// the merge agent decided to suppress the message.
    pub async fn send_or_merge(
        &self,
        branch_id: u64,
        message: &str,
        original_user_message: &str,
    ) -> Option<String> {
        // Collect other-branch messages as owned strings so we can drop the lock.
        let (other_texts, other_count) = {
            let sent_log = self.sent_log.lock().await;
            let branches = self.branches.lock().await;
            let last_merged = branches.get(&branch_id).copied().unwrap_or(0);

            let texts: Vec<String> = sent_log
                .iter()
                .filter(|e| e.counter > last_merged && e.branch_id != branch_id)
                .map(|e| e.text.clone())
                .collect();
            let count = texts.len();
            (texts, count)
        };

        if other_count == 0 {
            // No conflicts — send directly
            let counter = self.sent_counter.fetch_add(1, Ordering::SeqCst) + 1;
            self.sent_log.lock().await.push(SentEntry {
                counter,
                branch_id,
                text: message.to_string(),
            });
            self.branches
                .lock()
                .await
                .entry(branch_id)
                .and_modify(|c| *c = counter);

            debug!(branch_id, counter, "Sent message directly (no merge needed)");
            return Some(message.to_string());
        }

        // Build context of what was already sent
        let recent_sent: String = other_texts
            .iter()
            .map(|t| format!("Assistant: {t}"))
            .collect::<Vec<_>>()
            .join("\n\n");

        // Run the merge agent
        info!(
            branch_id,
            other_count,
            "Running merge agent"
        );

        let merged = self
            .run_merge_agent(&recent_sent, original_user_message, message)
            .await;

        match merged {
            Ok(Some(merged_text)) if !merged_text.trim().is_empty() => {
                let counter = self.sent_counter.fetch_add(1, Ordering::SeqCst) + 1;
                self.sent_log.lock().await.push(SentEntry {
                    counter,
                    branch_id,
                    text: merged_text.clone(),
                });
                self.branches
                    .lock()
                    .await
                    .entry(branch_id)
                    .and_modify(|c| *c = counter);

                info!(branch_id, counter, "Sent merged message");
                Some(merged_text)
            }
            Ok(_) => {
                info!(branch_id, "Merge agent suppressed message");
                // Update the counter even though we didn't send
                let current = self.sent_counter.load(Ordering::SeqCst);
                self.branches
                    .lock()
                    .await
                    .entry(branch_id)
                    .and_modify(|c| *c = current);
                None
            }
            Err(e) => {
                warn!(branch_id, "Merge agent failed: {e}, sending original");
                // Fallback: send the original message
                let counter = self.sent_counter.fetch_add(1, Ordering::SeqCst) + 1;
                self.sent_log.lock().await.push(SentEntry {
                    counter,
                    branch_id,
                    text: message.to_string(),
                });
                self.branches
                    .lock()
                    .await
                    .entry(branch_id)
                    .and_modify(|c| *c = counter);
                Some(message.to_string())
            }
        }
    }

    /// Classify whether an incoming message is related to the one currently
    /// being processed, or is about a different topic.
    /// Returns true if RELATED, false if UNRELATED.
    pub async fn classify(
        &self,
        recent_context: &[ChatMessage],
        message_being_processed: &str,
        new_message: &str,
    ) -> bool {
        let context_str = recent_context
            .iter()
            .filter(|m| m.role != "system")
            .filter_map(|m| {
                m.content
                    .as_ref()
                    .map(|c| format!("{}: {}", m.role, c))
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"You are classifying whether a new user message is related to an ongoing conversation or a separate topic.

Recent conversation context:
{context_str}

Currently being processed by the agent: "{message_being_processed}"

New incoming message: "{new_message}"

Is the new message a correction, follow-up, or continuation of the message being processed? Or is it about a different/unrelated topic?

Reply with exactly one word: RELATED or UNRELATED"#
        );

        let messages = vec![ChatMessage::user(prompt)];
        let llm_future = self.gateway_llm.complete(&messages, &[]);
        match tokio::time::timeout(Duration::from_secs(5), llm_future).await {
            Ok(Ok(LlmResponse::Text(text))) => {
                let word = text.trim().to_uppercase();
                let related = word.contains("RELATED") && !word.contains("UNRELATED");
                info!(
                    classification = if related { "RELATED" } else { "UNRELATED" },
                    raw = %text.trim(),
                    "Classified incoming message"
                );
                related
            }
            Ok(Ok(_)) => {
                warn!("Classifier returned tool calls instead of text, defaulting to UNRELATED");
                false
            }
            Ok(Err(e)) => {
                warn!("Classification failed: {e}, defaulting to UNRELATED");
                false
            }
            Err(_) => {
                warn!("Classification timed out after 5s, defaulting to UNRELATED");
                false
            }
        }
    }

    /// Run a merge agent to contextualize a message given what was already sent.
    /// This is a pure text operation — no tool access.
    /// Returns None to suppress the message, or Some(text) for the merged output.
    async fn run_merge_agent(
        &self,
        recent_sent_messages: &str,
        original_user_message: &str,
        current_message: &str,
    ) -> Result<Option<String>> {
        let system_prompt = format!(
            r#"You are a conversation flow manager. The user is chatting with an AI assistant that processes multiple requests concurrently. Another processing thread has already sent messages to the user.

Your job: decide what to do with a new outgoing message from a different thread.

Rules:
- If redundant, irrelevant, or would break flow → output EXACTLY the string: SUPPRESS
- If it adds value but needs context → output ONLY the rewritten message text
- If it's fine as-is → output the message unchanged

CRITICAL: Output ONLY the final message text or the word SUPPRESS. No reasoning, no explanation, no thinking, no preamble. Your entire response will be sent directly to the user (or used as a suppress signal).

Recent conversation with the user (what they see):
{recent_sent_messages}

The user's original request that triggered this message:
{original_user_message}

New message to evaluate:
{current_message}"#
        );

        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(
                "Output the rewritten message or SUPPRESS. Nothing else.".to_string(),
            ),
        ];

        let response = self.gateway_llm.complete(&messages, &[]).await?;

        match response {
            LlmResponse::Text(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() || trimmed == "SUPPRESS" {
                    Ok(None)
                } else {
                    Ok(Some(trimmed.to_string()))
                }
            }
            LlmResponse::ToolCalls(..) => {
                warn!("Merge agent returned tool calls despite empty tool list, using original message as fallback");
                Ok(Some(current_message.to_string()))
            }
        }
    }
}

/// Global map of session orchestrators, keyed by session_key.
pub struct GatewayMap {
    orchestrators: Mutex<HashMap<String, Arc<SessionOrchestrator>>>,
    gateway_llm: Arc<dyn LlmProvider>,
}

impl GatewayMap {
    pub fn new(gateway_llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            orchestrators: Mutex::new(HashMap::new()),
            gateway_llm,
        }
    }

    /// Get or create the orchestrator for a session.
    pub async fn get(&self, session_key: &str) -> Arc<SessionOrchestrator> {
        let mut map = self.orchestrators.lock().await;
        map.entry(session_key.to_string())
            .or_insert_with(|| Arc::new(SessionOrchestrator::new(self.gateway_llm.clone())))
            .clone()
    }
}
