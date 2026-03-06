use async_trait::async_trait;
use chrono::Utc;
use chrono_tz::Tz;
use serde::Deserialize;
use std::sync::Arc;
use tracing::warn;

use crate::channel::CallContext;
use crate::pairing::PairingStore;

pub struct CurrentTimeSkill {
    pairing: Arc<PairingStore>,
}

impl CurrentTimeSkill {
    pub fn new(pairing: Arc<PairingStore>) -> Self {
        Self { pairing }
    }

    fn resolve_tz(&self, ctx: &CallContext) -> Tz {
        let tz_str = ctx.identity_id
            .and_then(|id| self.pairing.get_timezone(id).ok().flatten());

        match tz_str {
            Some(s) => s.parse::<Tz>().unwrap_or_else(|_| {
                warn!(tz = %s, "Unrecognised timezone, falling back to UTC");
                Tz::UTC
            }),
            None => Tz::UTC,
        }
    }
}

#[derive(Deserialize)]
struct SetTimezoneArgs {
    timezone: String,
}

#[async_trait]
impl super::Skill for CurrentTimeSkill {
    fn name(&self) -> &str {
        "Current time"
    }

    fn description(&self) -> Option<&str> {
        Some("Returns the current date/time in the user's timezone, and lets the agent update that timezone")
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "current_time",
                    "description": "Returns the current date, time, and day of week in the user's local timezone.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "set_timezone",
                    "description": "Update the user's timezone. Use an IANA timezone name (e.g. 'Asia/Shanghai', 'Europe/London', 'America/New_York'). Call this when the user mentions they are in a different timezone or have travelled.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "IANA timezone name, e.g. 'Asia/Shanghai'"
                            }
                        },
                        "required": ["timezone"]
                    }
                }
            }),
        ]
    }

    async fn call(&self, tool_name: &str, args: &str, ctx: &CallContext) -> Option<String> {
        match tool_name {
            "current_time" => {
                let tz = self.resolve_tz(ctx);
                let now = Utc::now().with_timezone(&tz);
                Some(now.format("%A, %B %-d, %Y — %H:%M %Z").to_string())
            }
            "set_timezone" => {
                let identity_id = ctx.identity_id?;
                let parsed: SetTimezoneArgs = match serde_json::from_str(args) {
                    Ok(a) => a,
                    Err(e) => return Some(format!("Invalid arguments: {e}")),
                };
                // Validate before saving.
                if parsed.timezone.parse::<Tz>().is_err() {
                    return Some(format!(
                        "'{}' is not a recognised IANA timezone. Use a name like 'Asia/Shanghai' or 'Europe/London'.",
                        parsed.timezone
                    ));
                }
                match self.pairing.set_timezone(identity_id, &parsed.timezone) {
                    Ok(()) => Some(format!("Timezone updated to {}.", parsed.timezone)),
                    Err(e) => Some(format!("Failed to save timezone: {e}")),
                }
            }
            _ => None,
        }
    }
}
