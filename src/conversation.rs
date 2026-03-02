use crate::llm::ChatMessage;
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use tracing::warn;

pub struct ConversationStore {
    conn: Arc<Mutex<Connection>>,
}

impl ConversationStore {
    pub fn new(data_dir: &str) -> Result<Arc<Self>> {
        let db_path = format!("{data_dir}/nina.db");
        let conn = Connection::open(&db_path).context("Failed to open conversation DB")?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS conversations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session_key TEXT NOT NULL,
                 role TEXT NOT NULL,
                 content TEXT,
                 tool_call_id TEXT,
                 tool_calls TEXT,
                 created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_conversations_session
                 ON conversations(session_key, created_at);",
        )
        .context("Failed to initialize conversations table")?;

        Ok(Arc::new(Self {
            conn: Arc::new(Mutex::new(conn)),
        }))
    }

    /// Persist a message.  System messages and RAG-injected user messages should
    /// NOT be passed here — callers are responsible for that distinction.
    pub fn save_message(&self, session_key: &str, msg: &ChatMessage) -> Result<()> {
        let tool_calls_json = msg
            .tool_calls
            .as_ref()
            .map(|tc| serde_json::to_string(tc))
            .transpose()
            .context("Failed to serialize tool_calls")?;

        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO conversations(session_key, role, content, tool_call_id, tool_calls, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_key,
                msg.role,
                msg.content,
                msg.tool_call_id,
                tool_calls_json,
                now,
            ],
        )
        .context("Failed to insert conversation message")?;
        Ok(())
    }

    /// Load the last `limit` messages for a session, oldest first.
    pub fn load_history(&self, session_key: &str, limit: usize) -> Result<Vec<ChatMessage>> {
        let conn = self.conn.lock().unwrap();
        // Subquery to get the most recent N rows, then return them oldest-first.
        let mut stmt = conn.prepare(
            "SELECT role, content, tool_call_id, tool_calls
             FROM (
                 SELECT id, role, content, tool_call_id, tool_calls, created_at
                 FROM conversations
                 WHERE session_key = ?1
                 ORDER BY created_at DESC, id DESC
                 LIMIT ?2
             )
             ORDER BY created_at ASC, id ASC",
        )?;

        let mut msgs: Vec<ChatMessage> = stmt
            .query_map(params![session_key, limit as i64], |row| {
                let role: String = row.get(0)?;
                let content: Option<String> = row.get(1)?;
                let tool_call_id: Option<String> = row.get(2)?;
                let tool_calls_json: Option<String> = row.get(3)?;
                Ok((role, content, tool_call_id, tool_calls_json))
            })?
            .filter_map(|r| match r {
                Ok(row) => Some(row),
                Err(e) => {
                    warn!("Skipping malformed conversation row: {e}");
                    None
                }
            })
            .map(|(role, content, tool_call_id, tool_calls_json)| {
                let tool_calls = tool_calls_json
                    .and_then(|j| serde_json::from_str(&j).ok());
                ChatMessage {
                    role,
                    content,
                    tool_call_id,
                    tool_calls,
                }
            })
            .collect();

        // Drop orphaned leading messages: the LIMIT window may start mid-tool-call
        // sequence, leaving tool-result messages without their preceding assistant
        // message. Strip until we reach a user or assistant (non-tool-result) message.
        while msgs.first().is_some_and(|m| m.role == "tool") {
            warn!("Dropping orphaned tool message from history window");
            msgs.remove(0);
        }

        Ok(msgs)
    }

    /// Delete all messages for a session (called on /reset).
    pub fn clear(&self, session_key: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM conversations WHERE session_key = ?1",
            params![session_key],
        )?;
        Ok(())
    }
}
