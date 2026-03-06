use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use cron::Schedule;
use rusqlite::{params, Connection};
use std::str::FromStr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{error, info, warn};

use crate::channel::{CallContext, ChannelTarget};

pub struct ListedReminder {
    pub token: String,
    pub message: String,
    pub state: String,
    pub fire_at: DateTime<Utc>,
    pub cron: Option<String>,
    pub isolated: bool,
}

pub struct PendingReminder {
    pub id: i64,
    pub message: String,
    pub target: ChannelTarget,
    pub fire_at: DateTime<Utc>,
    pub cron: Option<String>,
    pub isolated: bool,
}

pub struct ReminderStore {
    db: Arc<Mutex<Connection>>,
}

impl ReminderStore {
    pub fn new(db_path: &Path) -> Result<Arc<Self>> {
        let conn = Connection::open(db_path).context("Failed to open reminders DB")?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS reminders (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 message TEXT NOT NULL,
                 channel_id TEXT NOT NULL,
                 chat_id INTEGER NOT NULL,
                 fire_at INTEGER NOT NULL,
                 state TEXT NOT NULL DEFAULT 'pending',
                 created_at INTEGER NOT NULL
             );",
        )
        .context("Failed to initialize reminders table")?;
        // Migrations: add columns if they don't exist yet.
        let _ = conn.execute_batch("ALTER TABLE reminders ADD COLUMN cron TEXT;");
        let _ = conn.execute_batch("ALTER TABLE reminders ADD COLUMN isolated INTEGER NOT NULL DEFAULT 0;");
        let _ = conn.execute_batch("ALTER TABLE reminders ADD COLUMN token TEXT;");
        let _ = conn.execute_batch(
            "UPDATE reminders SET token = printf('%016x', abs(random())) WHERE token IS NULL AND state = 'pending';"
        );
        Ok(Arc::new(Self {
            db: Arc::new(Mutex::new(conn)),
        }))
    }

    pub fn add(
        &self,
        message: &str,
        target: &ChannelTarget,
        fire_at: DateTime<Utc>,
        cron: Option<&str>,
        isolated: bool,
    ) -> Result<(i64, String)> {
        let token = format!("{:016x}", rand::random::<u64>());
        let conn = self.db.lock().unwrap();
        conn.execute(
            "INSERT INTO reminders (message, channel_id, chat_id, fire_at, cron, isolated, state, created_at, token)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7, ?8)",
            params![
                message,
                target.channel_id,
                target.chat_id,
                fire_at.timestamp(),
                cron,
                isolated as i32,
                Utc::now().timestamp(),
                token
            ],
        )?;
        Ok((conn.last_insert_rowid(), token))
    }

    /// Update fire_at and reset state to pending (for rescheduling cron reminders).
    pub fn reschedule(&self, id: i64, next_fire_at: DateTime<Utc>) -> Result<()> {
        let conn = self.db.lock().unwrap();
        conn.execute(
            "UPDATE reminders SET fire_at = ?1, state = 'pending' WHERE id = ?2",
            params![next_fire_at.timestamp(), id],
        )?;
        Ok(())
    }

    /// Atomically transition pending -> firing. Returns true if this caller claimed it.
    pub fn claim(&self, id: i64) -> Result<bool> {
        let conn = self.db.lock().unwrap();
        let rows = conn.execute(
            "UPDATE reminders SET state = 'firing' WHERE id = ?1 AND state = 'pending'",
            params![id],
        )?;
        Ok(rows == 1)
    }

    /// Transition firing -> fired after successful delivery.
    pub fn complete(&self, id: i64) -> Result<()> {
        let conn = self.db.lock().unwrap();
        conn.execute(
            "UPDATE reminders SET state = 'fired' WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Load all pending reminders with fire_at in the future.
    pub fn load_pending(&self) -> Result<Vec<PendingReminder>> {
        let conn = self.db.lock().unwrap();
        let now = Utc::now().timestamp();
        let mut stmt = conn.prepare(
            "SELECT id, message, channel_id, chat_id, fire_at, cron, isolated
             FROM reminders
             WHERE state = 'pending' AND fire_at > ?1",
        )?;
        let rows = stmt.query_map(params![now], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, i32>(6)?,
            ))
        })?;
        let mut pending = Vec::new();
        for row in rows {
            let (id, message, channel_id, chat_id, fire_at_ts, cron, isolated) = row?;
            let fire_at = Utc
                .timestamp_opt(fire_at_ts, 0)
                .single()
                .unwrap_or_else(Utc::now);
            pending.push(PendingReminder {
                id,
                message,
                target: ChannelTarget {
                    channel_id,
                    chat_id,
                },
                fire_at,
                cron,
                isolated: isolated != 0,
            });
        }
        Ok(pending)
    }

    /// List reminders for a given target, with optional state and message filters.
    pub fn list_for_target(
        &self,
        target: &ChannelTarget,
        state_filter: &str,
        search: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ListedReminder>> {
        let conn = self.db.lock().unwrap();
        let mut sql = String::from(
            "SELECT COALESCE(token, ''), message, state, fire_at, cron, isolated \
             FROM reminders WHERE channel_id = ? AND chat_id = ?"
        );

        let channel_id = target.channel_id.clone();
        let chat_id = target.chat_id;
        let state_str: Option<String> = if state_filter != "all" { Some(state_filter.to_string()) } else { None };
        let search_pat: Option<String> = search.map(|s| format!("%{}%", s));
        let limit_i64 = limit as i64;

        if state_str.is_some() {
            sql.push_str(" AND state = ?");
        }
        if search_pat.is_some() {
            sql.push_str(" AND message LIKE ?");
        }
        sql.push_str(" ORDER BY fire_at ASC LIMIT ?");

        let mut param_refs: Vec<&dyn rusqlite::ToSql> = vec![&channel_id, &chat_id];
        if let Some(ref s) = state_str {
            param_refs.push(s);
        }
        if let Some(ref p) = search_pat {
            param_refs.push(p);
        }
        param_refs.push(&limit_i64);

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, i32>(5)?,
            ))
        })?;

        let mut result = Vec::new();
        for row in rows {
            let (token, message, state, fire_at_ts, cron, isolated) = row?;
            let fire_at = Utc
                .timestamp_opt(fire_at_ts, 0)
                .single()
                .unwrap_or_else(Utc::now);
            result.push(ListedReminder { token, message, state, fire_at, cron, isolated: isolated != 0 });
        }
        Ok(result)
    }

    /// Hard-delete a pending reminder by token, scoped to the given target.
    /// Returns true if a row was deleted, false if not found or already fired.
    pub fn cancel(&self, token: &str, target: &ChannelTarget) -> Result<bool> {
        let conn = self.db.lock().unwrap();
        let rows = conn.execute(
            "DELETE FROM reminders WHERE token = ?1 AND channel_id = ?2 AND chat_id = ?3 AND state = 'pending'",
            params![token, target.channel_id, target.chat_id],
        )?;
        Ok(rows == 1)
    }

    /// Reset stale "firing" reminders back to "pending" (crash recovery).
    /// Resets any reminder stuck in "firing" with fire_at < now - 60 seconds.
    pub fn recover_stale(&self) -> Result<usize> {
        let conn = self.db.lock().unwrap();
        let cutoff = Utc::now().timestamp() - 60;
        let rows = conn.execute(
            "UPDATE reminders SET state = 'pending' WHERE state = 'firing' AND fire_at < ?1",
            params![cutoff],
        )?;
        if rows > 0 {
            warn!("Recovered {rows} stale firing reminders back to pending");
        }
        Ok(rows)
    }
}

/// Spawn a background task that sleeps until fire_at, then claims, fires, and completes the reminder.
pub fn spawn_reminder_task(
    store: Arc<ReminderStore>,
    agent: Arc<crate::agent::Agent>,
    router: Arc<tokio::sync::RwLock<crate::channel::ChannelRouter>>,
    reminder: PendingReminder,
) {
    tokio::spawn(async move {
        let now = Utc::now();
        if reminder.fire_at > now {
            let delay = (reminder.fire_at - now)
                .to_std()
                .unwrap_or_default();
            tokio::time::sleep(delay).await;
        }

        // Atomic claim: only one instance will proceed.
        match store.claim(reminder.id) {
            Ok(true) => {}
            Ok(false) => {
                info!(id = reminder.id, "Reminder already claimed by another task, skipping");
                return;
            }
            Err(e) => {
                error!(id = reminder.id, "Failed to claim reminder: {e}");
                return;
            }
        }

        info!(id = reminder.id, msg = %reminder.message, isolated = reminder.isolated, "Firing reminder");

        let call_ctx = CallContext { target: reminder.target.clone(), identity_id: None };
        // Reminders don't participate in branch/merge — use a dummy absorb channel.
        let (_absorb_tx, mut absorb_rx) = tokio::sync::mpsc::channel::<String>(1);

        if reminder.isolated {
            let isolated_key = format!("reminder:{}", reminder.id);
            let originating = format!("{}:{}", reminder.target.channel_id, reminder.target.chat_id);
            let framed = format!(
                "[Scheduled reminder — isolated autonomous session]\n\
                 This session is ephemeral and will be cleared after you finish.\n\
                 Your response will NOT be sent anywhere automatically. Use send_to_session(session_key, text) to deliver output.\n\
                 This reminder was created from session: {originating} — you may send there, or elsewhere, or nowhere.\n\
                 To pass context to your future self: use remember() for durable facts, remember_daily() for today's notes.\n\n\
                 Reminder: {}", reminder.message
            );
            match agent
                .process_message(&isolated_key, &framed, &framed, |_| async {}, &mut absorb_rx, &call_ctx)
                .await
            {
                Ok(_) => {}
                Err(e) => {
                    error!(id = reminder.id, "Isolated reminder agent error: {e}");
                }
            }
            agent.reset_session(&isolated_key).await;
        } else {
            let session_key = reminder.target.session_key();
            let framed = format!("[Scheduled reminder from yourself — visible to you only, the user did not send this] {}", reminder.message);
            match agent
                .process_message(&session_key, &framed, &framed, |_| async {}, &mut absorb_rx, &call_ctx)
                .await
            {
                Ok(Some(response)) => {
                    if !response.is_empty() {
                        router.read().await.send_text(&reminder.target, &response).await;
                    }
                }
                Ok(None) => {
                    // Response was suppressed (shouldn't happen for reminders)
                }
                Err(e) => {
                    error!(id = reminder.id, "Reminder agent error: {e}");
                }
            }
        }

        // For cron reminders, reschedule the next occurrence instead of completing.
        if let Some(ref expr) = reminder.cron {
            match Schedule::from_str(expr) {
                Ok(schedule) => {
                    if let Some(next) = schedule.upcoming(Utc).next() {
                        info!(id = reminder.id, ?next, "Rescheduling cron reminder");
                        if let Err(e) = store.reschedule(reminder.id, next) {
                            error!(id = reminder.id, "Failed to reschedule cron reminder: {e}");
                            return;
                        }
                        spawn_reminder_task(
                            store,
                            agent,
                            router,
                            PendingReminder {
                                id: reminder.id,
                                message: reminder.message,
                                target: reminder.target,
                                fire_at: next,
                                cron: reminder.cron,
                                isolated: reminder.isolated,
                            },
                        );
                        return;
                    } else {
                        warn!(id = reminder.id, "Cron schedule produced no future occurrences");
                    }
                }
                Err(e) => {
                    error!(id = reminder.id, "Failed to parse cron expression for rescheduling: {e}");
                }
            }
        }

        if let Err(e) = store.complete(reminder.id) {
            error!(id = reminder.id, "Failed to mark reminder as fired: {e}");
        }
    });
}
