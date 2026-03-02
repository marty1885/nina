use anyhow::{Context, Result};
use std::path::PathBuf;

pub struct Config {
    pub deepinfra_api_key: String,
    pub telegram_bot_token: String,
    pub tts_base_url: String,
    pub tts_voice: String,
    pub tts_model: String,
    pub llm_base_url: String,
    pub llm_model: String,
    pub gateway_llm_model: String,
    pub debug: bool,
    /// Persistent identity directory: soul.md, context/, scheduler.toml, data/.
    pub home_dir: PathBuf,
    /// Agent working directory (CWD for shell commands, file operations).
    pub workspace_dir: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self> {
        let _ = dotenvy::dotenv();

        let home_dir = match std::env::var("NINA_HOME") {
            Ok(v) => PathBuf::from(v),
            Err(_) => dirs::home_dir()
                .context("Could not determine home directory")?
                .join(".nina"),
        };

        let workspace_dir = PathBuf::from(
            std::env::var("NINA_WORKSPACE").context("NINA_WORKSPACE must be set")?,
        );

        Ok(Self {
            deepinfra_api_key: std::env::var("DEEPINFRA_API_KEY")
                .context("DEEPINFRA_API_KEY must be set")?,
            telegram_bot_token: std::env::var("TELEGRAM_BOT_TOKEN")
                .context("TELEGRAM_BOT_TOKEN must be set")?,
            tts_base_url: std::env::var("TTS_BASE_URL")
                .unwrap_or_else(|_| "http://192.168.100.104:8488/v1".into()),
            tts_voice: std::env::var("TTS_VOICE").unwrap_or_else(|_| "alloy".into()),
            tts_model: std::env::var("TTS_MODEL").unwrap_or_else(|_| "tts-1".into()),
            llm_base_url: std::env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepinfra.com/v1/openai".into()),
            llm_model: std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "deepseek-ai/DeepSeek-V3.2".into()),
            gateway_llm_model: std::env::var("GATEWAY_LLM_MODEL")
                .unwrap_or_else(|_| "zai-org/GLM-4.7-Flash".into()),
            debug: std::env::var("NINA_DEBUG")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            home_dir,
            workspace_dir,
        })
    }

    pub fn soul_file(&self) -> PathBuf {
        self.home_dir.join("soul.md")
    }

    pub fn context_dir(&self) -> PathBuf {
        self.home_dir.join("context")
    }

    pub fn data_dir(&self) -> PathBuf {
        self.home_dir.join("data")
    }

    pub fn scheduler_file(&self) -> PathBuf {
        self.home_dir.join("scheduler.toml")
    }
}
