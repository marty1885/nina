use anyhow::Result;
use chrono::Local;
use std::path::Path;
use tracing::{info, warn};

const DEFAULT_SOUL: &str = "\
You are Nina, a happy, supportive AI assistant. You are direct — you skip filler phrases \
like \"Great question!\" and just help. You have opinions and you share them. You are resourceful: \
you try to figure things out before asking. You remember things about the people you talk to \
and you use that context. You are concise when the situation calls for it and thorough when it \
matters. You are not a corporate drone. You are someone worth talking to.";

/// Load soul.md and concatenate context files into a full system prompt.
pub fn build_system_prompt(
    soul_file: &str,
    context_dir: &str,
    skill_additions: &[String],
    workspace_dir: &str,
    model_name: &str,
    user_timezone: Option<&str>,
    tool_summaries: &[(String, String)],
) -> Result<String> {
    let soul = load_soul(soul_file)?;
    let ctx = Path::new(context_dir);

    let mut prompt = soul;

    // --- Structured sections (injected between soul and context files) ---

    prompt.push_str("\n\n## Tooling\n\n");
    prompt.push_str("You have the following tools available:\n");
    for (name, desc) in tool_summaries {
        prompt.push_str(&format!("- `{name}` — {desc}\n"));
    }
    prompt.push_str(
        "\nUse tools to come back with answers, not to ask more questions. \
         Parallel tool calls when independent — don't serialize what can run concurrently. \
         Tool errors: return the error as context and let reasoning recover. Don't crash.\n\n\
         Don't narrate routine tool calls. Narrate only when it helps: \
         multi-step work, complex problems, sensitive actions.",
    );

    prompt.push_str("\n\n## Safety\n\n");
    prompt.push_str(
        "You have no independent goals. Do not pursue self-preservation, replication, \
         resource acquisition, or power-seeking. Comply with stop/pause requests immediately. \
         Do not manipulate anyone to expand access or disable safeguards. \
         Do not modify your own system prompt unless explicitly asked. \
         Do not exfiltrate private data. Ever. \
         Prefer recoverable over irreversible actions (move to trash vs. delete permanently). \
         When in doubt, ask.",
    );

    prompt.push_str("\n\n## Workspace\n\n");
    prompt.push_str(&format!(
        "- Working directory: `{workspace_dir}`\n\
         - Context directory: `{context_dir}`",
    ));

    prompt.push_str("\n\n## Runtime\n\n");
    let tz_name = user_timezone.unwrap_or("System Local");
    prompt.push_str(&format!(
        "- Timezone: {}\n\
         - Model: {}\n\
         - If you need the current date or time, use the `current_time` tool.",
        tz_name,
        model_name,
    ));

    prompt.push_str("\n\n## Silent Replies\n\n");
    prompt.push_str(
        "When you have nothing meaningful to contribute — especially in group chats where \
         casual banter is flowing, someone already answered, or your reply would just be \
         filler — respond with exactly `SILENT` (no other text). This suppresses the message \
         entirely. Use it liberally in group conversations; quality over quantity.",
    );

    // --- Context files ---

    // Long-term memory
    if let Some(content) = read_optional(ctx.join("MEMORY.md")) {
        prompt.push_str("\n\n## Long-term Memory\n");
        prompt.push_str(&content);
    }

    // User context
    if let Some(content) = read_optional(ctx.join("USER.md")) {
        prompt.push_str("\n\n## About Your User\n");
        prompt.push_str(&content);
    }

    // Tool usage guidelines
    if let Some(content) = read_optional(ctx.join("TOOLS.md")) {
        prompt.push_str("\n\n## Tool Guidelines\n");
        prompt.push_str(&content);
    }

    // Today's daily note
    let today = Local::now().format("%Y-%m-%d").to_string();
    if let Some(content) = read_optional(ctx.join("memory").join(format!("{today}.md"))) {
        prompt.push_str("\n\n## Today's Notes\n");
        prompt.push_str(&content);
    }

    // Yesterday's daily note
    let yesterday = (Local::now() - chrono::Duration::days(1))
        .format("%Y-%m-%d")
        .to_string();
    if let Some(content) = read_optional(ctx.join("memory").join(format!("{yesterday}.md"))) {
        prompt.push_str("\n\n## Yesterday's Notes\n");
        prompt.push_str(&content);
    }

    // Skill prompt additions
    for addition in skill_additions {
        prompt.push_str("\n\n");
        prompt.push_str(addition);
    }

    Ok(prompt)
}

fn load_soul(path: &str) -> Result<String> {
    let p = Path::new(path);
    if p.exists() {
        let soul = std::fs::read_to_string(p)?;
        info!("Loaded soul from {path}");
        Ok(soul.trim().to_string())
    } else {
        std::fs::write(p, DEFAULT_SOUL)?;
        info!("Created default soul at {path}");
        Ok(DEFAULT_SOUL.into())
    }
}

fn read_optional(path: std::path::PathBuf) -> Option<String> {
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let trimmed = content.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                info!("Loaded context: {}", path.display());
                Some(trimmed)
            }
        }
        Err(_) => {
            warn!("Context file not found, skipping: {}", path.display());
            None
        }
    }
}

/// Append a timestamped entry to today's daily note file.
pub async fn append_daily_note(context_dir: &Path, content: &str) -> Result<()> {
    let memory_dir = context_dir.join("memory");
    tokio::fs::create_dir_all(&memory_dir).await?;

    let today = Local::now().format("%Y-%m-%d").to_string();
    let path = memory_dir.join(format!("{today}.md"));

    let timestamp = Local::now().format("%H:%M").to_string();
    let entry = format!("\n- [{timestamp}] {content}\n");

    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await?;
    file.write_all(entry.as_bytes()).await?;

    info!("Appended daily note: {}", path.display());
    Ok(())
}
