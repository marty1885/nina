use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
use tracing::info;

use crate::channel::{CallContext, ChannelRouter};
use crate::tts::TtsClient;
use tokio::sync::RwLock;

pub struct TtsSkill {
    tts: Arc<TtsClient>,
    router: Arc<RwLock<ChannelRouter>>,
}

#[derive(Deserialize)]
struct SpeakArgs {
    text: String,
}

impl TtsSkill {
    pub fn new(tts: Arc<TtsClient>, router: Arc<RwLock<ChannelRouter>>) -> Self {
        Self { tts, router }
    }
}

#[async_trait]
impl super::Skill for TtsSkill {
    fn name(&self) -> &str {
        "Text to speech"
    }

    fn description(&self) -> Option<&str> {
        Some("Synthesize speech and send as voice note")
    }

    fn system_prompt_addition(&self) -> Option<String> {
        Some(
            "You can speak responses aloud using speak(). Use it when the user asks for audio \
             or when a voice response would be more natural."
                .into(),
        )
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "speak",
                "description": "Synthesize text to speech and send as a voice note to the current chat.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to speak"
                        }
                    },
                    "required": ["text"]
                }
            }
        })]
    }

    async fn call(&self, tool_name: &str, args: &str, ctx: &CallContext) -> Option<String> {
        if tool_name != "speak" {
            return None;
        }

        let args: SpeakArgs = match serde_json::from_str(args) {
            Ok(a) => a,
            Err(e) => return Some(format!("Invalid arguments: {e}")),
        };

        let target = &ctx.target;

        info!(text_len = args.text.len(), target = ?target, "speak tool");

        let audio = match self.tts.synthesize(&args.text).await {
            Ok(bytes) => bytes,
            Err(e) => return Some(format!("TTS synthesis failed: {e}")),
        };

        self.router.read().await.send_voice(target, audio).await;
        Some("Voice note sent.".into())
    }
}
