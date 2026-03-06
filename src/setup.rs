use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{self, Write};
use tracing::info;

use crate::config::Config;
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, OpenAiProvider};

const SETUP_SYSTEM_PROMPT: &str = "\
You are a setup assistant creating the identity for a new AI assistant.

Generate two pieces of text based on the user's input:

1. \"soul\" — A persona prompt in second person: \"You are {name}, ...\"
   3-6 sentences. Capture personality, communication style, values.
   Do NOT include anything about tools, safety, or technical capabilities.

2. \"user\" — A structured summary of what the user shared about themselves.
   Use bullet points or short paragraphs.

Output ONLY a JSON object with keys \"soul\" and \"user\". No markdown fences, no preamble, no explanation.";

#[derive(Debug, Deserialize)]
struct SetupResponse {
    soul: String,
    user: String,
}

fn prompt_line(prompt: &str) -> Result<String> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn parse_setup_response(raw: &str) -> Result<SetupResponse> {
    let trimmed = raw.trim();

    // Strip markdown fences if present
    let json_str = if trimmed.starts_with("```") {
        let without_opening = trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed);
        without_opening
            .strip_suffix("```")
            .unwrap_or(without_opening)
            .trim()
    } else {
        trimmed
    };

    serde_json::from_str::<SetupResponse>(json_str).context("Failed to parse setup JSON response")
}

pub async fn run_setup(config: &Config) -> Result<()> {
    let soul_file = config.soul_file();
    let identity_file = config.identity_file();

    println!(
        "No identity found at {} — let's bootstrap your agent.\n",
        identity_file.display()
    );

    let agent_name = prompt_line("What should your agent be called? ")?;
    if agent_name.is_empty() {
        anyhow::bail!("Agent name cannot be empty");
    }

    let personality_input =
        prompt_line("Describe the personality you want (or Enter for default): ")?;
    let personality = if personality_input.is_empty() {
        "a helpful, direct AI assistant".to_string()
    } else {
        personality_input
    };

    let user_info =
        prompt_line("Tell me about yourself — name, what you do, anything the agent should know: ")?;

    // Build LLM provider from first entry in agent_providers
    let (provider_def, _) = config
        .agent_providers
        .first()
        .context("No agent providers configured")?;
    let llm: Box<dyn LlmProvider> = Box::new(OpenAiProvider::new(
        &provider_def.base_url,
        &provider_def.api_key,
        &provider_def.model,
    ));

    let user_message = format!(
        "Agent name: {agent_name}\nPersonality: {personality}\nAbout the user: {user_info}"
    );

    let mut messages = vec![
        ChatMessage::system(SETUP_SYSTEM_PROMPT),
        ChatMessage::user(&user_message),
    ];

    // Retry loop: up to 3 attempts
    let mut parsed = None;
    for attempt in 0..3 {
        let response = llm.complete(&messages, &[]).await?;
        let text = match response {
            LlmResponse::Text(t) => t,
            LlmResponse::ToolCalls(Some(t), _) => t,
            _ => String::new(),
        };

        match parse_setup_response(&text) {
            Ok(resp) => {
                parsed = Some(resp);
                break;
            }
            Err(e) => {
                if attempt < 2 {
                    info!("Setup JSON parse failed (attempt {}): {e}", attempt + 1);
                    messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: Some(text),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                    messages.push(ChatMessage::user(
                        "That wasn't valid JSON. Output ONLY a raw JSON object.",
                    ));
                } else {
                    eprintln!(
                        "Failed to parse LLM response after 3 attempts.\n\
                         Raw output:\n{text}\n\n\
                         Please manually create:\n\
                         - {}\n\
                         - {}\n\
                         - {}/USER.md",
                        identity_file.display(),
                        soul_file.display(),
                        config.context_dir().display(),
                    );
                    anyhow::bail!("Setup failed: could not parse LLM response as JSON");
                }
            }
        }
    }

    let resp = parsed.unwrap();

    // Write identity.toml
    std::fs::write(
        &identity_file,
        format!("name = \"{}\"\n", agent_name.replace('"', "\\\"")),
    )
    .with_context(|| format!("Failed to write {}", identity_file.display()))?;

    // Write soul.md
    std::fs::write(&soul_file, &resp.soul)
        .with_context(|| format!("Failed to write {}", soul_file.display()))?;

    // Write context/USER.md
    let context_dir = config.context_dir();
    std::fs::create_dir_all(&context_dir)?;
    let user_file = context_dir.join("USER.md");
    std::fs::write(&user_file, &resp.user)
        .with_context(|| format!("Failed to write {}", user_file.display()))?;

    println!("\n✓ Identity created. {agent_name} is ready.\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_json() {
        let input = r#"{"soul": "You are Test, a test bot.", "user": "- Name: Tester"}"#;
        let resp = parse_setup_response(input).unwrap();
        assert_eq!(resp.soul, "You are Test, a test bot.");
        assert_eq!(resp.user, "- Name: Tester");
    }

    #[test]
    fn parse_json_with_fences() {
        let input = "```json\n{\"soul\": \"You are Test.\", \"user\": \"info\"}\n```";
        let resp = parse_setup_response(input).unwrap();
        assert_eq!(resp.soul, "You are Test.");
    }

    #[test]
    fn parse_json_with_plain_fences() {
        let input = "```\n{\"soul\": \"soul text\", \"user\": \"user text\"}\n```";
        let resp = parse_setup_response(input).unwrap();
        assert_eq!(resp.soul, "soul text");
    }

    #[test]
    fn parse_json_with_whitespace() {
        let input = "  \n  {\"soul\": \"s\", \"user\": \"u\"}  \n  ";
        let resp = parse_setup_response(input).unwrap();
        assert_eq!(resp.soul, "s");
    }

    #[test]
    fn parse_invalid_json_fails() {
        let input = "This is not JSON at all";
        assert!(parse_setup_response(input).is_err());
    }
}
