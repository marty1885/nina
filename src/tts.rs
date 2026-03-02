use anyhow::{bail, Result};
use tracing::debug;

pub struct TtsClient {
    client: reqwest::Client,
    base_url: String,
    voice: String,
    model: String,
}

impl TtsClient {
    pub fn new(base_url: &str, voice: &str, model: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to build TTS HTTP client");

        Self {
            client,
            base_url: base_url.into(),
            voice: voice.into(),
            model: model.into(),
        }
    }

    /// Synthesize text to audio. Returns opus bytes.
    pub async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        debug!(text_len = text.len(), "TTS synthesis request");

        let resp = self
            .client
            .post(format!("{}/audio/speech", self.base_url))
            .json(&serde_json::json!({
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": "opus",
            }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("TTS error {status}: {body}");
        }

        Ok(resp.bytes().await?.to_vec())
    }
}
