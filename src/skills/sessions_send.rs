use async_trait::async_trait;
use serde::Deserialize;
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;
use tracing::info;

use crate::channel::{CallContext, ChannelRouter, ChannelTarget};
use crate::llm::ChatMessage;
use crate::session::SessionManager;

pub struct SendToSessionSkill {
    router: Arc<RwLock<ChannelRouter>>,
    sessions: Arc<OnceLock<Arc<SessionManager>>>,
}

#[derive(Deserialize)]
struct SendArgs {
    session_key: String,
    text: String,
    reason: String,
}

impl SendToSessionSkill {
    pub fn new(router: Arc<RwLock<ChannelRouter>>, sessions: Arc<OnceLock<Arc<SessionManager>>>) -> Self {
        Self { router, sessions }
    }
}

#[async_trait]
impl super::Skill for SendToSessionSkill {
    fn name(&self) -> &str {
        "Send to session"
    }

    fn description(&self) -> Option<&str> {
        Some("Route a message directly to another chat session")
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "send_to_session",
                "description": "Send a text message directly to a specific chat session. The sent text is recorded in that session's history so the next conversation turn has full context. The session_key format is 'channel_id:chat_id' (e.g. 'telegram:123456789').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_key": {
                            "type": "string",
                            "description": "Target session in 'channel_id:chat_id' format"
                        },
                        "text": {
                            "type": "string",
                            "description": "The message text to send to the user"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why you are sending to this specific session and what prompted the outreach. Stored as internal context in the target session so the next conversation turn understands the background."
                        }
                    },
                    "required": ["session_key", "text", "reason"]
                }
            }
        })]
    }

    async fn call(&self, tool_name: &str, args: &str, ctx: &CallContext) -> Option<String> {
        if tool_name != "send_to_session" {
            return None;
        }

        let args: SendArgs = match serde_json::from_str(args) {
            Ok(a) => a,
            Err(e) => return Some(format!("Invalid arguments: {e}")),
        };

        // Parse "channel_id:chat_id" — split on first ':'
        let Some(colon) = args.session_key.find(':') else {
            return Some(format!(
                "Invalid session_key '{}': expected 'channel_id:chat_id'",
                args.session_key
            ));
        };
        let channel_id = &args.session_key[..colon];
        let chat_id_str = &args.session_key[colon + 1..];
        let chat_id: i64 = match chat_id_str.parse() {
            Ok(id) => id,
            Err(_) => {
                return Some(format!(
                    "Invalid chat_id in session_key '{}': not a number",
                    args.session_key
                ))
            }
        };

        let target = ChannelTarget {
            channel_id: channel_id.to_string(),
            chat_id,
        };

        info!(session_key = %args.session_key, "send_to_session tool");
        self.router.read().await.send_text(&target, &args.text).await;

        // Record the sent text in the target session's history so the next
        // conversation turn has full context. The internal annotation is stored
        // only — the user received the clean text above.
        if let Some(sessions) = self.sessions.get() {
            let annotated = format!(
                "{}\n\n[Proactive outreach — user received this text. Sent from: {}. Reason: {}]",
                args.text,
                ctx.target.session_key(),
                args.reason,
            );
            let msg = ChatMessage {
                role: "assistant".into(),
                content: Some(annotated),
                tool_calls: None,
                tool_call_id: None,
            };
            sessions.push_message(&args.session_key, msg).await;
        }

        Some(format!("Message sent to {}.", args.session_key))
    }
}
