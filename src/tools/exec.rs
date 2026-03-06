use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use tracing::info;

pub struct ExecTool;

#[derive(Deserialize)]
pub struct ExecArgs {
    pub command: String,
    /// Set to true when a non-zero exit code is expected (e.g. grep with no
    /// matches, `test -f`, probing commands). The output is still returned but
    /// won't count as a tool error toward the burst guard.
    pub allow_error: Option<bool>,
    /// Optional timeout in seconds. Defaults to 30s. Clamped to a maximum of
    /// 1800s (30 minutes).
    pub timeout_secs: Option<u64>,
}

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_TIMEOUT_SECS: u64 = 1800;

#[derive(Debug)]
pub struct ExecError(pub String);

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ExecError {}

impl Tool for ExecTool {
    const NAME: &'static str = "exec";
    type Error = ExecError;
    type Args = ExecArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "exec".into(),
            description: format!("Execute a shell command. Returns stdout followed by stderr (prefixed with \"[stderr]\"). Output is truncated (200 char/line, 80 lines, 4000 bytes max). Default timeout is {DEFAULT_TIMEOUT_SECS}s; automatically clamped to {MAX_TIMEOUT_SECS}s max."),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "allow_error": {
                        "type": "boolean",
                        "description": "Set to true when a non-zero exit code is expected (e.g. grep with no matches, test -f, probing commands). Output is still returned but the failure won't count against the error burst guard."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": format!("Timeout in seconds for the command. Defaults to {DEFAULT_TIMEOUT_SECS}. Automatically clamped to {MAX_TIMEOUT_SECS} max. Use longer timeouts for builds, package installs, or other slow operations.")
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<String, Self::Error> {
        info!(cmd = %args.command, "exec tool");

        let timeout = args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS).min(MAX_TIMEOUT_SECS);
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout),
            tokio::process::Command::new("/bin/bash")
                .arg("-c")
                .arg(&args.command)
                .output(),
        )
        .await
        .map_err(|_| ExecError(format!("Command timed out after {timeout}s")))?
        .map_err(|e| ExecError(format!("Failed to execute: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr] ");
            result.push_str(&stderr);
        }
        if result.is_empty() {
            result.push_str("(no output)");
        }

        result = truncate_output(&result);

        if !output.status.success() && !args.allow_error.unwrap_or(false) {
            let code = output.status.code().map(|c| c.to_string()).unwrap_or_else(|| "?".into());
            return Err(ExecError(format!("exit code {code}\n{result}")));
        }

        Ok(result)
    }
}

const MAX_LINE_CHARS: usize = 200;
const MAX_LINES: usize = 80;
const MAX_BYTES: usize = 4000;

/// Truncate command output to keep context windows manageable.
///
/// Applies three levels of truncation:
/// 1. Per-line: lines longer than MAX_LINE_CHARS get head/tail with "..." in the middle
/// 2. Line count: keeps first and last lines, drops middle if > MAX_LINES
/// 3. Byte cap: final safety net at MAX_BYTES
fn truncate_output(s: &str) -> String {
    let mut lines: Vec<String> = s
        .lines()
        .map(|line| {
            if line.chars().count() <= MAX_LINE_CHARS {
                line.to_string()
            } else {
                // Show first 100 and last 80 chars
                let chars: Vec<char> = line.chars().collect();
                let head: String = chars[..100].iter().collect();
                let tail: String = chars[chars.len() - 80..].iter().collect();
                format!("{head} ... ({} chars omitted) ... {tail}", chars.len() - 180)
            }
        })
        .collect();

    // Truncate line count: keep first 60 + last 20
    if lines.len() > MAX_LINES {
        let total = lines.len();
        let head: Vec<String> = lines[..60].to_vec();
        let tail: Vec<String> = lines[total - 20..].to_vec();
        lines = head;
        lines.push(format!("\n... ({} lines omitted) ...\n", total - 80));
        lines.extend(tail);
    }

    let mut result = lines.join("\n");

    // Final byte cap as safety net
    if result.len() > MAX_BYTES {
        let end = result.floor_char_boundary(MAX_BYTES);
        result.truncate(end);
        result.push_str("\n... (truncated)");
    }

    result
}
