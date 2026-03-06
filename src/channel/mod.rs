pub mod telegram;

use async_trait::async_trait;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChannelTarget {
    pub channel_id: String,
    pub chat_id: i64,
}

impl ChannelTarget {
    /// Stable string key for session maps.
    pub fn session_key(&self) -> String {
        format!("{}:{}", self.channel_id, self.chat_id)
    }
}

/// Per-invocation context passed to tool calls so skills know where to route responses.
#[derive(Clone, Debug)]
pub struct CallContext {
    pub target: ChannelTarget,
    /// Identity of the user this call is on behalf of. None for system-initiated calls.
    pub identity_id: Option<i64>,
}

#[derive(Debug)]
pub struct ReplyContext {
    pub sender_name: String,
    pub sender_id: String,
    pub text: String,
}

#[derive(Debug)]
pub struct IncomingMessage {
    pub source: ChannelTarget,
    pub text: String,
    pub user_name: String,
    /// Stable per-platform user ID (Telegram user.id, Matrix @user:server, etc.)
    pub sender_id: String,
    pub reply_to: Option<ReplyContext>,
}

#[async_trait]
pub trait Channel: Send + Sync {
    fn id(&self) -> &str;
    async fn poll_updates(&self) -> Vec<IncomingMessage>;
    async fn send_text(&self, target: &ChannelTarget, text: &str);
    async fn send_voice(&self, target: &ChannelTarget, audio: Vec<u8>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_key_format() {
        let target = ChannelTarget {
            channel_id: "telegram".into(),
            chat_id: 12345,
        };
        assert_eq!(target.session_key(), "telegram:12345");
    }

    #[test]
    fn session_key_negative_chat_id() {
        // Group chats have negative IDs in Telegram
        let target = ChannelTarget {
            channel_id: "telegram".into(),
            chat_id: -100123456,
        };
        assert_eq!(target.session_key(), "telegram:-100123456");
    }

    #[test]
    fn session_key_zero_chat_id() {
        let target = ChannelTarget {
            channel_id: "matrix".into(),
            chat_id: 0,
        };
        assert_eq!(target.session_key(), "matrix:0");
    }

    #[test]
    fn channel_target_equality() {
        let a = ChannelTarget { channel_id: "telegram".into(), chat_id: 1 };
        let b = ChannelTarget { channel_id: "telegram".into(), chat_id: 1 };
        let c = ChannelTarget { channel_id: "telegram".into(), chat_id: 2 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

pub struct ChannelRouter {
    channels: HashMap<String, Box<dyn Channel>>,
}

impl ChannelRouter {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    pub fn register(&mut self, channel: Box<dyn Channel>) {
        let id = channel.id().to_string();
        tracing::info!(channel = %id, "Registered channel");
        self.channels.insert(id, channel);
    }

    pub async fn send_text(&self, target: &ChannelTarget, text: &str) {
        if let Some(ch) = self.channels.get(&target.channel_id) {
            ch.send_text(target, text).await;
        } else {
            tracing::warn!(channel = %target.channel_id, "No channel registered for target");
        }
    }

    pub async fn send_voice(&self, target: &ChannelTarget, audio: Vec<u8>) {
        if let Some(ch) = self.channels.get(&target.channel_id) {
            ch.send_voice(target, audio).await;
        } else {
            tracing::warn!(channel = %target.channel_id, "No channel registered for target");
        }
    }

    pub async fn poll_all(&self) -> Vec<IncomingMessage> {
        let mut all = Vec::new();
        for ch in self.channels.values() {
            all.extend(ch.poll_updates().await);
        }
        all
    }
}
