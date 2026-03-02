use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use cron::Schedule;
use serde::Deserialize;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;
use tracing::info;

use crate::agent::Agent;
use crate::channel::{CallContext, ChannelRouter};
use crate::reminders::{PendingReminder, ReminderStore, spawn_reminder_task};

pub struct ReminderSkill {
    store: Arc<ReminderStore>,
    /// Set after agent construction to break the circular dependency.
    agent: Arc<OnceLock<Arc<Agent>>>,
    router: Arc<RwLock<ChannelRouter>>,
}

impl ReminderSkill {
    pub fn new(
        store: Arc<ReminderStore>,
        agent: Arc<OnceLock<Arc<Agent>>>,
        router: Arc<RwLock<ChannelRouter>>,
    ) -> Self {
        Self {
            store,
            agent,
            router,
        }
    }
}

#[derive(Deserialize)]
struct RemindMeArgs {
    message: String,
    delay_seconds: Option<u64>,
    at_time: Option<String>,
    cron: Option<String>,
    isolated: Option<bool>,
}

#[async_trait]
impl super::Skill for ReminderSkill {
    fn name(&self) -> &str {
        "Reminders"
    }

    fn description(&self) -> Option<&str> {
        Some("Set timed reminders that fire a message after a delay or at a specific time")
    }

    fn system_prompt_addition(&self) -> Option<String> {
        Some(
            "You can set reminders with remind_me(). Provide delay_seconds (seconds from now), \
             at_time (ISO8601 datetime), or cron (6-field cron expression: sec min hour dom month dow), \
             plus a message to deliver when the reminder fires. \
             Cron reminders repeat on schedule until explicitly cancelled. \
             Example cron: \"0 0 9 * * Mon-Fri\" fires every weekday at 09:00. \
             Set isolated=true to run the reminder in an ephemeral autonomous session: \
             the response is not sent anywhere automatically — use send_to_session() inside the reminder to deliver output. \
             Use list_reminders() to see scheduled reminders (filterable by state and message). \
             Use cancel_reminder(token) to cancel a pending reminder by its token (get tokens from list_reminders)."
                .into(),
        )
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "remind_me",
                    "description": "Set a reminder. When it fires, you will receive the message back as a '[Scheduled reminder from yourself — visible to you only, the user did not send this]' prompt and should act on it. The user never sees the trigger — only your response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The reminder message to deliver when the reminder fires"
                            },
                            "delay_seconds": {
                                "type": "integer",
                                "description": "Seconds from now before the reminder fires (one-shot)"
                            },
                            "at_time": {
                                "type": "string",
                                "description": "ISO8601 datetime when the reminder should fire (one-shot)"
                            },
                            "cron": {
                                "type": "string",
                                "description": "6-field cron expression (sec min hour dom month dow) for recurring reminders. Example: \"0 0 9 * * Mon-Fri\" for weekdays at 09:00."
                            },
                            "isolated": {
                                "type": "boolean",
                                "description": "Run in an ephemeral isolated session with no chat history. Silent by default — use send_to_session to deliver output if needed. Good for autonomous background work."
                            }
                        },
                        "required": ["message"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "list_reminders",
                    "description": "List reminders for this chat. Use filters to narrow results — always prefer filtering over retrieving everything.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {
                                "type": "string",
                                "enum": ["pending", "fired", "all"],
                                "description": "Filter by state. Defaults to 'pending'."
                            },
                            "search": {
                                "type": "string",
                                "description": "Optional case-insensitive substring match on the reminder message."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Optional max results to return. Defaults to 20."
                            }
                        },
                        "required": []
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "cancel_reminder",
                    "description": "Cancel a pending reminder by its token. Get tokens from list_reminders. Cannot cancel reminders that have already fired.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token": {
                                "type": "string",
                                "description": "The reminder token (from list_reminders)"
                            }
                        },
                        "required": ["token"]
                    }
                }
            }),
        ]
    }

    async fn call(&self, tool_name: &str, args: &str, ctx: &CallContext) -> Option<String> {
        match tool_name {
            "remind_me" => self.handle_remind_me(args, ctx).await,
            "list_reminders" => self.handle_list_reminders(args, ctx).await,
            "cancel_reminder" => self.handle_cancel_reminder(args, ctx).await,
            _ => None,
        }
    }
}

// ── per-tool handlers ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ListRemindersArgs {
    state: Option<String>,
    search: Option<String>,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct CancelReminderArgs {
    token: String,
}

impl ReminderSkill {
    async fn handle_remind_me(&self, args: &str, ctx: &CallContext) -> Option<String> {
        let args: RemindMeArgs = match serde_json::from_str(args) {
            Ok(a) => a,
            Err(e) => return Some(format!("Invalid arguments: {e}")),
        };

        const MIN_CRON_INTERVAL_SECS: i64 = 60;

        let (fire_at, cron_expr): (DateTime<Utc>, Option<String>) =
            if let Some(ref expr) = args.cron {
                match Schedule::from_str(expr) {
                    Ok(schedule) => {
                        let mut upcoming = schedule.upcoming(Utc);
                        let first = match upcoming.next() {
                            Some(t) => t,
                            None => return Some(format!("Cron expression '{expr}' produces no future occurrences")),
                        };
                        if let Some(second) = upcoming.next() {
                            let interval = (second - first).num_seconds();
                            if interval < MIN_CRON_INTERVAL_SECS {
                                return Some(format!(
                                    "Cron expression '{expr}' fires too frequently \
                                     (every {interval}s). Minimum interval is {MIN_CRON_INTERVAL_SECS}s."
                                ));
                            }
                        }
                        (first, Some(expr.clone()))
                    }
                    Err(e) => return Some(format!("Invalid cron expression '{expr}': {e}")),
                }
            } else if let Some(delay) = args.delay_seconds {
                (Utc::now() + Duration::seconds(delay as i64), None)
            } else if let Some(ref ts) = args.at_time {
                match chrono::DateTime::parse_from_rfc3339(ts) {
                    Ok(dt) => (dt.with_timezone(&Utc), None),
                    Err(e) => return Some(format!("Invalid at_time '{ts}': {e}")),
                }
            } else {
                return Some("One of delay_seconds, at_time, or cron must be provided.".into());
            };

        let target = ctx.target.clone();
        let isolated = args.isolated.unwrap_or(false);

        let (id, token) = match self.store.add(&args.message, &target, fire_at, cron_expr.as_deref(), isolated) {
            Ok(r) => r,
            Err(e) => return Some(format!("Failed to store reminder: {e}")),
        };

        let agent = match self.agent.get() {
            Some(a) => a.clone(),
            None => return Some("Agent not yet initialized; reminder stored but not scheduled.".into()),
        };

        info!(id, ?fire_at, isolated, "Spawning reminder task");

        spawn_reminder_task(
            self.store.clone(),
            agent,
            self.router.clone(),
            PendingReminder {
                id,
                message: args.message,
                target,
                fire_at,
                cron: cron_expr,
                isolated,
            },
        );

        let human_time = fire_at.format("%Y-%m-%d %H:%M:%S UTC").to_string();
        Some(format!("Reminder set for {human_time}. Token: {token}"))
    }

    async fn handle_list_reminders(&self, args: &str, ctx: &CallContext) -> Option<String> {
        let args: ListRemindersArgs = match serde_json::from_str(args) {
            Ok(a) => a,
            Err(e) => return Some(format!("Invalid arguments: {e}")),
        };

        let state = args.state.as_deref().unwrap_or("pending");
        let limit = args.limit.unwrap_or(20);

        let reminders = match self.store.list_for_target(&ctx.target, state, args.search.as_deref(), limit) {
            Ok(r) => r,
            Err(e) => return Some(format!("Failed to list reminders: {e}")),
        };

        if reminders.is_empty() {
            return Some("No reminders found.".into());
        }

        let lines: Vec<String> = reminders
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let msg = if r.message.len() > 80 {
                    format!("{}…", &r.message[..80])
                } else {
                    r.message.clone()
                };
                let tags = {
                    let mut t = Vec::new();
                    if r.cron.is_some() { t.push("cron"); } else { t.push("one-shot"); }
                    if r.isolated { t.push("isolated"); }
                    t.join(", ")
                };
                let human_time = r.fire_at.format("%Y-%m-%d %H:%M:%S UTC");
                format!(
                    "{}. [{}] {} | {} | {} | {}",
                    i + 1,
                    r.state,
                    r.token,
                    human_time,
                    tags,
                    msg
                )
            })
            .collect();

        Some(lines.join("\n"))
    }

    async fn handle_cancel_reminder(&self, args: &str, ctx: &CallContext) -> Option<String> {
        let args: CancelReminderArgs = match serde_json::from_str(args) {
            Ok(a) => a,
            Err(e) => return Some(format!("Invalid arguments: {e}")),
        };

        match self.store.cancel(&args.token, &ctx.target) {
            Ok(true) => Some("Reminder cancelled.".into()),
            Ok(false) => Some("Reminder not found or already fired.".into()),
            Err(e) => Some(format!("Failed to cancel reminder: {e}")),
        }
    }
}
