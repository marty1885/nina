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
