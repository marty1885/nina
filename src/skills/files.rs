use crate::session::SessionManager;
use async_trait::async_trait;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tracing::info;

pub struct FilesSkill {
    allowed_dirs: Vec<PathBuf>,
    context_dir: PathBuf,
    sessions: Arc<OnceLock<Arc<SessionManager>>>,
}

impl FilesSkill {
    pub fn new(
        allowed_dirs: Vec<PathBuf>,
        context_dir: PathBuf,
        sessions: Arc<OnceLock<Arc<SessionManager>>>,
    ) -> Self {
        Self {
            allowed_dirs: allowed_dirs
                .into_iter()
                .map(|d| d.canonicalize().unwrap_or(d))
                .collect(),
            context_dir: context_dir.canonicalize().unwrap_or(context_dir),
            sessions,
        }
    }

    fn validate_path(&self, path: &str) -> Result<PathBuf, String> {
        let p = PathBuf::from(path);
        // Resolve to absolute
        let resolved = if p.is_absolute() {
            p
        } else {
            std::env::current_dir()
                .map_err(|e| format!("Cannot resolve path: {e}"))?
                .join(p)
        };

        // Canonicalize parent to resolve symlinks (file may not exist yet for write)
        let canonical = if resolved.exists() {
            resolved
                .canonicalize()
                .map_err(|e| format!("Cannot resolve path: {e}"))?
        } else {
            let parent = resolved
                .parent()
                .ok_or_else(|| "Invalid path".to_string())?;
            let parent_canonical = parent
                .canonicalize()
                .map_err(|e| format!("Parent directory does not exist: {e}"))?;
            parent_canonical.join(resolved.file_name().unwrap_or_default())
        };

        for allowed in &self.allowed_dirs {
            if canonical.starts_with(allowed) {
                return Ok(canonical);
            }
        }

        Err(format!(
            "Path '{}' is outside allowed directories",
            path
        ))
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct EditFileArgs {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl super::Skill for FilesSkill {
    fn name(&self) -> &str {
        "File operations"
    }

    fn description(&self) -> Option<&str> {
        Some("Read and write files within allowed directories")
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file and return its contents (truncated to 8000 chars). Paths restricted to allowed directories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing an exact string with a new string. Fails if old_string is not found. Fails if old_string matches multiple locations and replace_all is false. To append, set old_string to the last line(s) of the file and include them in new_string. To create a new file, use write_file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "old_string": {
                                "type": "string",
                                "description": "The exact string to find and replace"
                            },
                            "new_string": {
                                "type": "string",
                                "description": "The string to replace it with"
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "Replace all occurrences. Defaults to false — errors if multiple matches found."
                            }
                        },
                        "required": ["path", "old_string", "new_string"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file. Paths restricted to allowed directories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write"
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            }),
        ]
    }

    async fn call(&self, tool_name: &str, args: &str, _ctx: &crate::channel::CallContext) -> Option<String> {
        match tool_name {
            "read_file" => Some(self.read_file(args).await),
            "edit_file" => Some(self.edit_file(args).await),
            "write_file" => Some(self.write_file(args).await),
            _ => None,
        }
    }
}

impl FilesSkill {
    async fn read_file(&self, args_json: &str) -> String {
        let args: ReadFileArgs = match serde_json::from_str(args_json) {
            Ok(a) => a,
            Err(e) => return format!("Invalid arguments: {e}"),
        };

        let path = match self.validate_path(&args.path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        info!(path = %path.display(), "read_file tool");

        match tokio::fs::read_to_string(&path).await {
            Ok(mut content) => {
                if content.len() > 8000 {
                    let end = content.floor_char_boundary(8000);
                    content.truncate(end);
                    content.push_str("\n... (truncated)");
                }
                content
            }
            Err(e) => format!("Failed to read file: {e}"),
        }
    }

    async fn write_file(&self, args_json: &str) -> String {
        let args: WriteFileArgs = match serde_json::from_str(args_json) {
            Ok(a) => a,
            Err(e) => return format!("Invalid arguments: {e}"),
        };

        let path = match self.validate_path(&args.path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        info!(path = %path.display(), "write_file tool");

        // Create parent dirs if needed
        if let Some(parent) = path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return format!("Failed to create directories: {e}");
            }
        }

        let len = args.content.len();
        match tokio::fs::write(&path, &args.content).await {
            Ok(()) => {
                self.maybe_refresh_prompts(&path).await;
                format!("Written {len} bytes to {}", path.display())
            }
            Err(e) => format!("Failed to write file: {e}"),
        }
    }

    /// If the written path is inside context_dir, refresh all session prompts so the
    /// agent sees the change immediately in the current conversation.
    async fn maybe_refresh_prompts(&self, path: &PathBuf) {
        if path.starts_with(&self.context_dir) {
            if let Some(sessions) = self.sessions.get() {
                sessions.refresh_all_prompts().await;
            }
        }
    }

    async fn edit_file(&self, args_json: &str) -> String {
        let args: EditFileArgs = match serde_json::from_str(args_json) {
            Ok(a) => a,
            Err(e) => return format!("Invalid arguments: {e}"),
        };

        let path = match self.validate_path(&args.path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return format!("Failed to read file: {e}"),
        };

        let count = content.matches(args.old_string.as_str()).count();
        if count == 0 {
            return "old_string not found in file".to_string();
        }
        if count > 1 && !args.replace_all {
            return format!("old_string matches {count} locations — set replace_all to true or provide more context to make it unique");
        }

        let updated = if args.replace_all {
            content.replace(args.old_string.as_str(), args.new_string.as_str())
        } else {
            content.replacen(args.old_string.as_str(), args.new_string.as_str(), 1)
        };

        info!(path = %path.display(), replace_all = args.replace_all, "edit_file tool");

        match tokio::fs::write(&path, updated).await {
            Ok(()) => {
                self.maybe_refresh_prompts(&path).await;
                "Edited.".to_string()
            }
            Err(e) => format!("Failed to write file: {e}"),
        }
    }
}
