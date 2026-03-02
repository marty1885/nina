use crate::conversation::ConversationStore;
use crate::llm::ChatMessage;
use chrono::Local;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

pub struct Session {
    pub messages: Vec<ChatMessage>,
    pub last_active: chrono::DateTime<chrono::Utc>,
    /// The local date for which the system prompt was built.
    pub prompt_date: chrono::NaiveDate,
}

impl Session {
    pub fn new(system_prompt: &str, prompt_date: chrono::NaiveDate) -> Self {
        Self {
            messages: vec![ChatMessage::system(system_prompt)],
            last_active: chrono::Utc::now(),
            prompt_date,
        }
    }

    /// Replace the system prompt (messages[0]) and update `prompt_date`.
    pub fn refresh_system_prompt(&mut self, new_prompt: &str, new_date: chrono::NaiveDate) {
        if !self.messages.is_empty() {
            self.messages[0] = ChatMessage::system(new_prompt);
        }
        self.prompt_date = new_date;
    }

    pub fn touch(&mut self) {
        self.last_active = chrono::Utc::now();
    }

    /// Trim history at ~80% of max_messages, keeping system prompt + recent turns.
    /// Never trims mid-tool-call-sequence.
    pub fn trim_if_needed(&mut self, max_messages: usize) {
        let threshold = max_messages * 80 / 100;
        if self.messages.len() <= threshold {
            return;
        }

        let system = self.messages[0].clone();
        let keep_count = max_messages / 2;
        let mut cut_at = self.messages.len().saturating_sub(keep_count);

        // Don't cut in the middle of a tool call sequence.
        while cut_at < self.messages.len() {
            let role = self.messages[cut_at].role.as_str();
            if role == "user" {
                break;
            }
            cut_at += 1;
        }

        if cut_at >= self.messages.len() {
            return;
        }

        let mut trimmed = vec![system];
        trimmed.extend_from_slice(&self.messages[cut_at..]);
        self.messages = trimmed;
    }
}

pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    system_prompt: String,
    conv_store: Arc<ConversationStore>,
    /// Path to soul file, for rebuilding system prompt on new sessions.
    soul_path: String,
    /// Context directory, for rebuilding system prompt on new sessions.
    context_dir: String,
    /// Skill prompt additions, for rebuilding system prompt.
    skill_additions: Vec<String>,
    /// Agent working directory.
    workspace_dir: String,
    /// LLM model name for runtime info.
    model_name: String,
    /// Tool (name, description) pairs for the Tooling section.
    tool_summaries: Vec<(String, String)>,
}

impl SessionManager {
    pub fn new(
        system_prompt: String,
        conv_store: Arc<ConversationStore>,
        soul_path: String,
        context_dir: String,
        skill_additions: Vec<String>,
        workspace_dir: String,
        model_name: String,
        tool_summaries: Vec<(String, String)>,
    ) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            system_prompt,
            conv_store,
            soul_path,
            context_dir,
            skill_additions,
            workspace_dir,
            model_name,
            tool_summaries,
        }
    }

    /// Build a fresh system prompt with today's daily notes.
    fn fresh_system_prompt(&self) -> String {
        match crate::identity::build_system_prompt(
            &self.soul_path,
            &self.context_dir,
            &self.skill_additions,
            &self.workspace_dir,
            &self.model_name,
            None,
            &self.tool_summaries,
        ) {
            Ok(prompt) => prompt,
            Err(e) => {
                warn!("Failed to rebuild system prompt: {e}, using cached");
                self.system_prompt.clone()
            }
        }
    }

    /// Ensure the session for `key` is loaded from DB into the in-memory map.
    /// If already present, this is a fast no-op (read-lock only).
    async fn ensure_loaded(&self, key: &str) {
        // Fast path: already in memory.
        {
            let sessions = self.sessions.read().await;
            if sessions.contains_key(key) {
                return;
            }
        }

        // Load history from DB (blocking sqlite read; acceptable for startup / cold sessions).
        let history = match self.conv_store.load_history(key, 100) {
            Ok(h) => h,
            Err(e) => {
                warn!(session_key = %key, "Failed to load conversation history: {e}");
                vec![]
            }
        };

        let mut sessions = self.sessions.write().await;
        // Double-check: another task may have inserted it while we were loading.
        if sessions.contains_key(key) {
            return;
        }

        // New session: rebuild system prompt fresh so daily notes are current.
        let prompt = self.fresh_system_prompt();
        let mut session = Session::new(&prompt, Local::now().date_naive());
        session.messages.extend(history);
        sessions.insert(key.to_string(), session);
    }

    pub async fn get_messages(&self, key: &str) -> Vec<ChatMessage> {
        let sessions = self.sessions.read().await;
        sessions
            .get(key)
            .map(|s| s.messages.clone())
            .unwrap_or_else(|| vec![ChatMessage::system(&self.system_prompt)])
    }

    /// Push a user message: persists `persist_text` (the original, without RAG prefix)
    /// to the DB, but adds a message with `session_text` (possibly RAG-augmented) to
    /// the in-memory session.  Call this instead of `push_message` for user turns.
    pub async fn push_user_message(
        &self,
        key: &str,
        persist_text: &str,
        session_text: &str,
    ) {
        // Load history first so that the in-memory session reflects all previous turns
        // before we append the new user message.
        self.ensure_loaded(key).await;

        // If the day has changed since the system prompt was built, refresh it
        // so daily notes stay current across midnight.
        {
            let today = Local::now().date_naive();
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(key) {
                if session.prompt_date != today {
                    info!(session_key = %key, old = %session.prompt_date, new = %today, "Day changed, refreshing system prompt");
                    let prompt = self.fresh_system_prompt();
                    session.refresh_system_prompt(&prompt, today);
                }
            }
        }

        // Persist the ORIGINAL text (RAG is rebuilt fresh each turn).
        if let Err(e) = self
            .conv_store
            .save_message(key, &ChatMessage::user(persist_text))
        {
            warn!(session_key = %key, "Failed to persist user message: {e}");
        }

        let mut sessions = self.sessions.write().await;
        let prompt = self.system_prompt.clone();
        let session = sessions
            .entry(key.to_string())
            .or_insert_with(|| Session::new(&prompt, Local::now().date_naive()));
        session.messages.push(ChatMessage::user(session_text));
        session.touch();
    }

    /// Push a non-user message (assistant text, tool calls, tool results).
    /// Persists to DB and adds to the in-memory session.
    pub async fn push_message(&self, key: &str, msg: ChatMessage) {
        // Persist before acquiring the session lock.
        if msg.role != "system" {
            if let Err(e) = self.conv_store.save_message(key, &msg) {
                warn!(session_key = %key, "Failed to persist message (role={}): {e}", msg.role);
            }
        }

        let mut sessions = self.sessions.write().await;
        let prompt = self.system_prompt.clone();
        let session = sessions
            .entry(key.to_string())
            .or_insert_with(|| Session::new(&prompt, Local::now().date_naive()));
        session.messages.push(msg);
        session.touch();
    }

    /// Push multiple messages atomically (used for tool results).
    pub async fn push_messages(&self, key: &str, msgs: Vec<ChatMessage>) {
        // Persist all before acquiring the session lock.
        for msg in &msgs {
            if msg.role != "system" {
                if let Err(e) = self.conv_store.save_message(key, msg) {
                    warn!(session_key = %key, "Failed to persist message (role={}): {e}", msg.role);
                }
            }
        }

        let mut sessions = self.sessions.write().await;
        let prompt = self.system_prompt.clone();
        let session = sessions
            .entry(key.to_string())
            .or_insert_with(|| Session::new(&prompt, Local::now().date_naive()));
        session.messages.extend(msgs);
        session.touch();
    }

    /// Clear the in-memory session and wipe DB history (called on /reset).
    pub async fn reset(&self, key: &str) {
        if let Err(e) = self.conv_store.clear(key) {
            warn!(session_key = %key, "Failed to clear conversation history: {e}");
        }
        // Build fresh prompt on reset too
        let prompt = self.fresh_system_prompt();
        let mut sessions = self.sessions.write().await;
        sessions.insert(key.to_string(), Session::new(&prompt, Local::now().date_naive()));
    }

    pub async fn trim(&self, key: &str, max_messages: usize) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(key) {
            session.trim_if_needed(max_messages);
        }
    }

    /// Rebuild and apply a fresh system prompt to all active in-memory sessions.
    /// Called after context files (MEMORY.md, USER.md, etc.) are modified so the
    /// agent sees the updated content immediately without waiting for a new session.
    pub async fn refresh_all_prompts(&self) {
        let prompt = self.fresh_system_prompt();
        let today = Local::now().date_naive();
        let mut sessions = self.sessions.write().await;
        for session in sessions.values_mut() {
            session.refresh_system_prompt(&prompt, today);
        }
        info!(count = sessions.len(), "Refreshed system prompt in all active sessions");
    }
}
