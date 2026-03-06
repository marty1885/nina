use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

// --- TOML config types (deserialization only) ---

#[derive(Deserialize)]
struct TomlConfig {
    #[serde(default)]
    env_file: Option<String>,
    agent: AgentToml,
    gateway: GatewayToml,
    #[serde(default)]
    tts: Option<TtsToml>,
    #[serde(rename = "providers", default)]
    providers: Vec<ProviderToml>,
}

#[derive(Deserialize)]
struct AgentToml {
    providers: Vec<WeightEntryToml>,
    #[serde(default)]
    fallback: Vec<WeightEntryToml>,
}

#[derive(Deserialize)]
struct GatewayToml {
    providers: Vec<WeightEntryToml>,
    #[serde(default)]
    fallback: Vec<WeightEntryToml>,
}

#[derive(Deserialize)]
struct WeightEntryToml {
    id: String,
    weight: u32,
}

#[derive(Deserialize)]
struct TtsToml {
    base_url: String,
    voice: String,
    model: String,
}

#[derive(Deserialize, Clone)]
struct ProviderToml {
    id: String,
    base_url: String,
    model: String,
    api_key_env: String,
}

// --- Public resolved provider ---

/// A fully-resolved provider definition with the API key value (not the env var name).
#[derive(Clone, Debug)]
pub struct ResolvedProvider {
    pub id: String,
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

// --- Config ---

pub struct Config {
    pub telegram_bot_token: String,
    /// TTS settings — None if not configured (skill won't be registered).
    pub tts: Option<(String, String, String)>,
    pub debug: bool,
    /// Persistent identity directory: soul.md, context/, scheduler.toml, data/.
    pub home_dir: PathBuf,
    /// Agent working directory (CWD for shell commands, file operations).
    pub workspace_dir: PathBuf,
    /// Base interval in hours between self-awareness passes (default 5).
    pub self_awareness_interval_hours: u64,
    /// Weighted pool of resolved LLM providers for the main agent (primary tier).
    /// On each request one provider is sampled by weight; on infra error the next is tried.
    pub agent_providers: Vec<(ResolvedProvider, u32)>,
    /// Optional fallback tier for the agent — tried only when all primary providers are exhausted.
    pub agent_fallback: Vec<(ResolvedProvider, u32)>,
    /// Weighted pool of resolved LLM providers for the gateway (primary tier).
    pub gateway_providers: Vec<(ResolvedProvider, u32)>,
    /// Optional fallback tier for the gateway.
    pub gateway_fallback: Vec<(ResolvedProvider, u32)>,
}

impl Config {
    pub fn load() -> Result<Self> {
        // Determine home dir first (needed to find nina.toml).
        let home_dir = match std::env::var("NINA_HOME") {
            Ok(v) => PathBuf::from(v),
            Err(_) => dirs::home_dir()
                .context("Could not determine home directory")?
                .join(".nina"),
        };

        let toml_path = home_dir.join("nina.toml");
        if !toml_path.exists() {
            eprintln!(
                "Error: nina.toml not found at {}\n\
                 Copy nina.toml.example from the project root to that path and fill in your providers.",
                toml_path.display()
            );
            std::process::exit(1);
        }

        let toml_str = std::fs::read_to_string(&toml_path)
            .with_context(|| format!("Failed to read {}", toml_path.display()))?;
        let toml: TomlConfig = toml::from_str(&toml_str)
            .with_context(|| format!("Failed to parse {}", toml_path.display()))?;

        // Load env: use env_file from toml if specified, else implicit .env.
        if let Some(ref env_file) = toml.env_file {
            let _ = dotenvy::from_path(env_file);
        } else {
            let _ = dotenvy::dotenv();
        }

        let workspace_dir = PathBuf::from(
            std::env::var("NINA_WORKSPACE").context("NINA_WORKSPACE must be set")?,
        );

        // Build a lookup map from provider id → definition.
        let provider_map: HashMap<String, ProviderToml> = toml
            .providers
            .into_iter()
            .map(|p| (p.id.clone(), p))
            .collect();

        // Resolve a provider id to a ResolvedProvider (looks up API key from env).
        let resolve = |id: &str| -> Result<ResolvedProvider> {
            let p = provider_map.get(id).with_context(|| {
                format!(
                    "Provider '{}' referenced in [agent] but not defined in [[providers]]",
                    id
                )
            })?;
            let api_key = std::env::var(&p.api_key_env).with_context(|| {
                format!(
                    "Provider '{}' requires env var '{}' which is not set",
                    id, p.api_key_env
                )
            })?;
            if api_key.is_empty() {
                anyhow::bail!(
                    "Provider '{}' requires env var '{}' which is set but empty",
                    id,
                    p.api_key_env
                );
            }
            Ok(ResolvedProvider {
                id: p.id.clone(),
                base_url: p.base_url.clone(),
                model: p.model.clone(),
                api_key,
            })
        };

        // Resolve agent provider pool.
        let agent_providers: Vec<(ResolvedProvider, u32)> = toml
            .agent
            .providers
            .iter()
            .map(|w| Ok((resolve(&w.id)?, w.weight)))
            .collect::<Result<Vec<_>>>()?;

        if agent_providers.is_empty() {
            anyhow::bail!("[agent].providers must not be empty");
        }

        let agent_fallback: Vec<(ResolvedProvider, u32)> = toml
            .agent
            .fallback
            .iter()
            .map(|w| Ok((resolve(&w.id)?, w.weight)))
            .collect::<Result<Vec<_>>>()?;

        // Resolve gateway provider pool.
        let gateway_providers: Vec<(ResolvedProvider, u32)> = toml
            .gateway
            .providers
            .iter()
            .map(|w| Ok((resolve(&w.id)?, w.weight)))
            .collect::<Result<Vec<_>>>()?;

        if gateway_providers.is_empty() {
            anyhow::bail!("[gateway].providers must not be empty");
        }

        let gateway_fallback: Vec<(ResolvedProvider, u32)> = toml
            .gateway
            .fallback
            .iter()
            .map(|w| Ok((resolve(&w.id)?, w.weight)))
            .collect::<Result<Vec<_>>>()?;

        // TTS — from [tts] section, or fall back to env vars. None if unconfigured.
        let tts = match toml.tts {
            Some(t) => Some((t.base_url, t.voice, t.model)),
            None => std::env::var("TTS_BASE_URL").ok().map(|url| {
                let voice = std::env::var("TTS_VOICE").unwrap_or_else(|_| "alloy".into());
                let model = std::env::var("TTS_MODEL").unwrap_or_else(|_| "tts-1".into());
                (url, voice, model)
            }),
        };

        Ok(Self {
            telegram_bot_token: std::env::var("TELEGRAM_BOT_TOKEN")
                .context("TELEGRAM_BOT_TOKEN must be set")?,
            tts,
            debug: std::env::var("NINA_DEBUG")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            home_dir,
            workspace_dir,
            self_awareness_interval_hours: std::env::var("NINA_SELF_AWARENESS_HOURS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            agent_providers,
            agent_fallback,
            gateway_providers,
            gateway_fallback,
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

    pub fn identity_file(&self) -> PathBuf {
        self.home_dir.join("identity.toml")
    }

    /// Read the agent name from identity.toml, or extract from soul.md, or fall back to "Assistant".
    pub fn agent_name(&self) -> String {
        // Try identity.toml first
        if let Ok(content) = std::fs::read_to_string(self.identity_file()) {
            if let Ok(table) = content.parse::<toml::Table>() {
                if let Some(toml::Value::String(name)) = table.get("name") {
                    if !name.is_empty() {
                        return name.clone();
                    }
                }
            }
        }

        // Try extracting from soul.md: "You are **{Name}**," or "You are {Name},"
        if let Ok(soul) = std::fs::read_to_string(self.soul_file()) {
            // Match "You are **Name**," or "You are Name,"
            if let Some(name) = extract_name_from_soul(&soul) {
                return name;
            }
        }

        "Assistant".to_string()
    }
}

/// Extract agent name from soul text patterns like "You are **Name**," or "You are Name,".
fn extract_name_from_soul(soul: &str) -> Option<String> {
    // Try bold pattern first: "You are **Name**"
    if let Some(rest) = soul.strip_prefix("You are **") {
        if let Some(end) = rest.find("**") {
            let name = rest[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    // Try plain pattern: "You are Name,"
    if let Some(rest) = soul.strip_prefix("You are ") {
        if let Some(end) = rest.find(',') {
            let name = rest[..end].trim();
            if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_uppercase()) {
                return Some(name.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(home: &str) -> Config {
        let dummy = ResolvedProvider {
            id: "test".into(),
            base_url: "http://localhost".into(),
            model: "test-model".into(),
            api_key: "test-key".into(),
        };
        Config {
            telegram_bot_token: "token".into(),
            tts: None,
            debug: false,
            home_dir: PathBuf::from(home),
            workspace_dir: PathBuf::from("/workspace"),
            self_awareness_interval_hours: 5,
            agent_providers: vec![(dummy.clone(), 100)],
            agent_fallback: vec![],
            gateway_providers: vec![(dummy, 100)],
            gateway_fallback: vec![],
        }
    }

    #[test]
    fn soul_file_path() {
        let c = make_config("/home/user/.nina");
        assert_eq!(c.soul_file(), PathBuf::from("/home/user/.nina/soul.md"));
    }

    #[test]
    fn context_dir_path() {
        let c = make_config("/home/user/.nina");
        assert_eq!(c.context_dir(), PathBuf::from("/home/user/.nina/context"));
    }

    #[test]
    fn data_dir_path() {
        let c = make_config("/home/user/.nina");
        assert_eq!(c.data_dir(), PathBuf::from("/home/user/.nina/data"));
    }

    #[test]
    fn scheduler_file_path() {
        let c = make_config("/home/user/.nina");
        assert_eq!(c.scheduler_file(), PathBuf::from("/home/user/.nina/scheduler.toml"));
    }

    #[test]
    fn debug_false_by_default() {
        let c = make_config("/tmp");
        assert!(!c.debug);
    }

    #[test]
    fn identity_file_path() {
        let c = make_config("/home/user/.nina");
        assert_eq!(c.identity_file(), PathBuf::from("/home/user/.nina/identity.toml"));
    }

    #[test]
    fn extract_name_bold_pattern() {
        assert_eq!(
            extract_name_from_soul("You are **Nina**, a helpful assistant."),
            Some("Nina".to_string()),
        );
    }

    #[test]
    fn extract_name_plain_pattern() {
        assert_eq!(
            extract_name_from_soul("You are Nina, a helpful assistant."),
            Some("Nina".to_string()),
        );
    }

    #[test]
    fn extract_name_no_match() {
        assert_eq!(extract_name_from_soul("Some other prompt text"), None);
    }

    #[test]
    fn agent_name_fallback() {
        let c = make_config("/nonexistent/path/.nina");
        assert_eq!(c.agent_name(), "Assistant");
    }
}
