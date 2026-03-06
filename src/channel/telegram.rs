use super::{Channel, ChannelTarget, IncomingMessage, ReplyContext};
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::atomic::{AtomicI64, Ordering};
use tracing::{error, warn};

pub struct TelegramBot {
    client: reqwest::Client,
    token: String,
    offset: AtomicI64,
}

#[derive(Debug, Deserialize)]
struct TgResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Update {
    update_id: i64,
    message: Option<Message>,
}

#[derive(Debug, Deserialize)]
struct Message {
    #[allow(dead_code)]
    message_id: i64,
    chat: Chat,
    from: Option<User>,
    text: Option<String>,
    reply_to_message: Option<Box<Message>>,
}

#[derive(Debug, Deserialize)]
struct Chat {
    id: i64,
}

#[derive(Debug, Deserialize)]
struct User {
    id: i64,
    first_name: String,
}

impl TelegramBot {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            token: token.into(),
            offset: AtomicI64::new(0),
        }
    }

    fn api_url(&self, method: &str) -> String {
        format!("https://api.telegram.org/bot{}/{}", self.token, method)
    }

    /// Send "typing" indicator.
    pub async fn send_typing(&self, chat_id: i64) {
        let _ = self
            .client
            .post(self.api_url("sendChatAction"))
            .json(&serde_json::json!({
                "chat_id": chat_id,
                "action": "typing",
            }))
            .send()
            .await;
    }

    /// Send a text message, splitting into chunks if needed.
    async fn send_message_raw(&self, chat_id: i64, text: &str) {
        let chunks = chunk_text(text, 4000);

        for chunk in chunks {
            let result = self
                .client
                .post(self.api_url("sendMessage"))
                .json(&serde_json::json!({
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown",
                }))
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let body = match resp.text().await {
                        Ok(b) => b,
                        Err(e) => {
                            error!("Failed to read sendMessage response: {e}");
                            continue;
                        }
                    };
                    let is_ok = serde_json::from_str::<serde_json::Value>(&body)
                        .ok()
                        .and_then(|v| v.get("ok")?.as_bool())
                        .unwrap_or(false);

                    if !is_ok {
                        // Retry without Markdown
                        let _ = self
                            .client
                            .post(self.api_url("sendMessage"))
                            .json(&serde_json::json!({
                                "chat_id": chat_id,
                                "text": chunk,
                            }))
                            .send()
                            .await;
                    }
                }
                Err(e) => error!("Failed to send message: {e}"),
            }
        }
    }

    /// Send an audio voice note.
    async fn send_voice_raw(&self, chat_id: i64, audio_bytes: Vec<u8>) -> Result<()> {
        let part = reqwest::multipart::Part::bytes(audio_bytes)
            .file_name("voice.ogg")
            .mime_str("audio/ogg")?;

        let form = reqwest::multipart::Form::new()
            .text("chat_id", chat_id.to_string())
            .part("voice", part);

        let resp = self
            .client
            .post(self.api_url("sendVoice"))
            .multipart(form)
            .send()
            .await?;

        let body: serde_json::Value = resp.json().await?;
        if !body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            warn!("sendVoice failed: {}", body);
        }

        Ok(())
    }
}

#[async_trait]
impl Channel for TelegramBot {
    fn id(&self) -> &str {
        "telegram"
    }

    async fn poll_updates(&self) -> Vec<IncomingMessage> {
        let resp = self
            .client
            .get(self.api_url("getUpdates"))
            .query(&[
                ("offset", self.offset.load(Ordering::Relaxed).to_string()),
                ("timeout", "30".into()),
            ])
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                // Redact the URL (which contains the bot token) from the error string.
                let msg = e.to_string();
                let safe = if let Some(idx) = msg.find("/bot") {
                    let after = &msg[idx + 4..]; // skip "/bot"
                    let end = after.find('/').map(|i| idx + 4 + i).unwrap_or(msg.len());
                    format!("{}[token]{}", &msg[..idx + 4], &msg[end..])
                } else {
                    msg
                };
                error!("Telegram poll error: {safe}");
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                return vec![];
            }
        };

        let parsed: TgResponse<Vec<Update>> = match resp.json().await {
            Ok(p) => p,
            Err(e) => {
                error!("Telegram JSON parse error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                return vec![];
            }
        };

        if !parsed.ok {
            warn!("Telegram API error: {:?}", parsed.description);
            return vec![];
        }

        let updates = parsed.result.unwrap_or_default();
        if let Some(last) = updates.last() {
            self.offset.store(last.update_id + 1, Ordering::Relaxed);
        }

        updates
            .into_iter()
            .filter_map(|u| {
                let msg = u.message?;
                let text = msg.text?;
                let user_name = msg
                    .from
                    .as_ref()
                    .map(|u| u.first_name.clone())
                    .unwrap_or_else(|| "User".into());
                let sender_id = msg
                    .from
                    .as_ref()
                    .map(|u| u.id.to_string())
                    .unwrap_or_else(|| msg.chat.id.to_string());
                let reply_to = msg.reply_to_message.as_deref().and_then(|r| {
                    let reply_text = r.text.as_deref()?.to_string();
                    let reply_sender_name = r
                        .from
                        .as_ref()
                        .map(|u| u.first_name.clone())
                        .unwrap_or_else(|| "Unknown".into());
                    let reply_sender_id = r
                        .from
                        .as_ref()
                        .map(|u| u.id.to_string())
                        .unwrap_or_default();
                    Some(ReplyContext {
                        sender_name: reply_sender_name,
                        sender_id: reply_sender_id,
                        text: reply_text,
                    })
                });
                Some(IncomingMessage {
                    source: ChannelTarget {
                        channel_id: "telegram".into(),
                        chat_id: msg.chat.id,
                    },
                    text,
                    user_name,
                    sender_id,
                    reply_to,
                })
            })
            .collect()
    }

    async fn send_text(&self, target: &ChannelTarget, text: &str) {
        self.send_message_raw(target.chat_id, text).await;
    }

    async fn send_voice(&self, target: &ChannelTarget, audio: Vec<u8>) {
        if let Err(e) = self.send_voice_raw(target.chat_id, audio).await {
            error!("Failed to send voice: {e}");
        }
    }
}

/// Split text into chunks, respecting UTF-8 boundaries.
fn chunk_text(text: &str, max_len: usize) -> Vec<&str> {
    if text.len() <= max_len {
        return vec![text];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + max_len).min(text.len());
        end = text.floor_char_boundary(end);

        if end < text.len() {
            if let Some(nl) = text[start..end].rfind('\n') {
                end = start + nl + 1;
            }
        }

        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}
