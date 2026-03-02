use std::path::{Path, PathBuf};
use tracing::{info, warn};

use async_trait::async_trait;

use super::Skill;
use crate::channel::CallContext;

const MAX_SKILL_SIZE: u64 = 64 * 1024; // 64KB per file
const MAX_TOTAL_CHARS: usize = 30_000; // total budget for all markdown skills

/// A markdown-based skill: pure knowledge injected into the system prompt.
/// Implements the Skill trait — no tools, just a system_prompt_addition.
pub struct MarkdownSkill {
    name: String,
    description: String,
    body: String,
}

#[async_trait]
impl Skill for MarkdownSkill {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some(&self.description)
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![]
    }

    async fn call(&self, _tool_name: &str, _args: &str, _ctx: &CallContext) -> Option<String> {
        None
    }

    fn system_prompt_addition(&self) -> Option<String> {
        Some(format!(
            "### Skill: {} — {}\n\n{}",
            self.name, self.description, self.body
        ))
    }
}

/// YAML-like frontmatter requirements.
#[derive(Debug, Default)]
struct Requirements {
    bins: Vec<String>,
    env: Vec<String>,
}

/// Parse simple YAML frontmatter between `---` fences.
/// Returns (name, description, requirements, body) or None if invalid.
fn parse_frontmatter(content: &str) -> Option<(String, String, Requirements, String)> {
    let content = content.trim_start_matches('\u{feff}'); // strip BOM
    if !content.starts_with("---") {
        warn!("Skill file missing frontmatter fence");
        return None;
    }

    let after_first = &content[3..];
    let end_pos = after_first.find("\n---")?;
    let frontmatter = &after_first[..end_pos];
    let body = after_first[end_pos + 4..].trim().to_string();

    let mut name = None;
    let mut description = None;
    let mut reqs = Requirements::default();
    let mut in_bins = false;
    let mut in_env = false;

    for line in frontmatter.lines() {
        let trimmed = line.trim();

        // Detect list items under requires.bins / requires.env
        if in_bins || in_env {
            if trimmed.starts_with("- ") {
                let val = trimmed[2..].trim().trim_matches('"').to_string();
                if in_bins {
                    reqs.bins.push(val);
                } else {
                    reqs.env.push(val);
                }
                continue;
            } else {
                in_bins = false;
                in_env = false;
            }
        }

        if let Some(val) = strip_yaml_key(trimmed, "name") {
            name = Some(val);
        } else if let Some(val) = strip_yaml_key(trimmed, "description") {
            description = Some(val);
        } else if trimmed == "bins:" {
            in_bins = true;
        } else if trimmed == "env:" {
            in_env = true;
        } else if let Some(val) = strip_yaml_key(trimmed, "bins") {
            // Inline array: bins: ["tmux", "foo"]
            reqs.bins = parse_inline_array(&val);
        } else if let Some(val) = strip_yaml_key(trimmed, "env") {
            reqs.env = parse_inline_array(&val);
        }
    }

    let name = name?;
    let description = description?;
    if body.is_empty() {
        warn!(name, "Skill has empty body, skipping");
        return None;
    }

    Some((name, description, reqs, body))
}

fn strip_yaml_key(line: &str, key: &str) -> Option<String> {
    let line = line.trim();
    if line.starts_with(key) {
        let rest = &line[key.len()..];
        if rest.starts_with(':') {
            let val = rest[1..].trim().trim_matches('"').to_string();
            if val.is_empty() {
                return None;
            }
            return Some(val);
        }
    }
    None
}

fn parse_inline_array(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.starts_with('[') && s.ends_with(']') {
        s[1..s.len() - 1]
            .split(',')
            .map(|v| v.trim().trim_matches('"').trim_matches('\'').to_string())
            .filter(|v| !v.is_empty())
            .collect()
    } else {
        vec![s.to_string()]
    }
}

/// Check that all required binaries and env vars are available.
fn check_requirements(reqs: &Requirements) -> bool {
    for bin in &reqs.bins {
        if which::which(bin).is_err() {
            warn!(binary = %bin, "Required binary not found, skipping skill");
            return false;
        }
    }
    for var in &reqs.env {
        if std::env::var(var).is_err() {
            warn!(env_var = %var, "Required env var not set, skipping skill");
            return false;
        }
    }
    true
}

/// Scan a single skills directory for SKILL.md files.
fn scan_skills_dir(dir: &Path) -> Vec<(String, String, String, PathBuf)> {
    let mut results = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return results,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let skill_file = path.join("SKILL.md");
        if !skill_file.exists() {
            continue;
        }

        // Size check
        if let Ok(meta) = std::fs::metadata(&skill_file) {
            if meta.len() > MAX_SKILL_SIZE {
                warn!(
                    path = %skill_file.display(),
                    size = meta.len(),
                    "Skill file exceeds 64KB limit, skipping"
                );
                continue;
            }
        }

        let content = match std::fs::read_to_string(&skill_file) {
            Ok(c) => c,
            Err(e) => {
                warn!(path = %skill_file.display(), "Failed to read skill: {e}");
                continue;
            }
        };

        let (name, description, reqs, body) = match parse_frontmatter(&content) {
            Some(parsed) => parsed,
            None => {
                warn!(path = %skill_file.display(), "Failed to parse skill frontmatter");
                continue;
            }
        };

        if !check_requirements(&reqs) {
            info!(name = %name, "Skill requirements not met, skipping");
            continue;
        }

        results.push((name, description, body, skill_file));
    }

    results
}

/// Load all markdown skills from the given directories, ordered lowest-to-highest precedence.
/// Later directories override earlier ones by skill name.
/// Returns boxed Skills ready for registration into SkillRegistry.
pub fn load_markdown_skills(skill_dirs: &[PathBuf]) -> Vec<Box<dyn Skill>> {
    let mut skill_map: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();

    for dir in skill_dirs {
        let skills_dir = dir.join("skills");
        for (name, description, body, path) in scan_skills_dir(&skills_dir) {
            if skill_map.contains_key(&name) {
                info!(name = %name, path = %path.display(), "Overriding skill from higher-precedence dir");
            } else {
                info!(name = %name, path = %path.display(), "Loaded markdown skill");
            }
            skill_map.insert(name, (description, body));
        }
    }

    let mut entries: Vec<(String, String, String)> = skill_map
        .into_iter()
        .map(|(name, (desc, body))| (name, desc, body))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Apply total character budget with binary-search truncation
    let total_chars: usize = entries.iter().map(|(_, _, body)| body.len()).sum();
    if total_chars > MAX_TOTAL_CHARS {
        warn!(
            total_chars,
            budget = MAX_TOTAL_CHARS,
            "Markdown skills exceed total char budget, truncating"
        );
        truncate_to_budget(&mut entries);
    }

    info!(count = entries.len(), "Markdown skills loaded");

    entries
        .into_iter()
        .map(|(name, description, body)| -> Box<dyn Skill> {
            Box::new(MarkdownSkill {
                name,
                description,
                body,
            })
        })
        .collect()
}

/// Truncate skill bodies to fit within the total character budget.
/// Uses binary search on max-per-skill to find the sweet spot.
fn truncate_to_budget(entries: &mut [(String, String, String)]) {
    if entries.is_empty() {
        return;
    }

    let max_body = entries.iter().map(|(_, _, b)| b.len()).max().unwrap_or(0);
    let mut lo: usize = 0;
    let mut hi: usize = max_body;

    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        let total: usize = entries.iter().map(|(_, _, b)| b.len().min(mid)).sum();
        if total <= MAX_TOTAL_CHARS {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    for (name, _, body) in entries.iter_mut() {
        if body.len() > lo {
            let truncated = &body[..lo];
            let cut = truncated.rfind('\n').unwrap_or(lo);
            *body = format!("{}\n\n[... truncated ...]", &body[..cut]);
            warn!(name = %name, "Skill body truncated to fit budget");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontmatter_basic() {
        let content = r#"---
name: tmux
description: "Remote-control tmux sessions"
requires:
  bins:
    - "tmux"
---

# tmux Session Control

Use tmux to manage sessions.
"#;
        let (name, desc, reqs, body) = parse_frontmatter(content).unwrap();
        assert_eq!(name, "tmux");
        assert_eq!(desc, "Remote-control tmux sessions");
        assert_eq!(reqs.bins, vec!["tmux"]);
        assert!(reqs.env.is_empty());
        assert!(body.contains("tmux Session Control"));
    }

    #[test]
    fn test_parse_frontmatter_inline_array() {
        let content = r#"---
name: test
description: "Test skill"
requires:
  bins: ["foo", "bar"]
  env: ["MY_VAR"]
---

Body text.
"#;
        let (_, _, reqs, _) = parse_frontmatter(content).unwrap();
        assert_eq!(reqs.bins, vec!["foo", "bar"]);
        assert_eq!(reqs.env, vec!["MY_VAR"]);
    }

    #[test]
    fn test_parse_frontmatter_no_requires() {
        let content = r#"---
name: simple
description: "No requirements"
---

Just a body.
"#;
        let (name, _, reqs, _) = parse_frontmatter(content).unwrap();
        assert_eq!(name, "simple");
        assert!(reqs.bins.is_empty());
        assert!(reqs.env.is_empty());
    }

    #[test]
    fn test_parse_frontmatter_missing_name() {
        let content = r#"---
description: "No name"
---

Body.
"#;
        assert!(parse_frontmatter(content).is_none());
    }
}
