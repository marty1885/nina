use anyhow::{Context, Result};
use rand::Rng;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// Persistent timer that survives process restarts and self-heals from mid-run crashes.
///
/// State machine per timer name:
///   'scheduled' + future deadline  → sleep until deadline, then fire
///   'scheduled' + past deadline    → overdue (was offline), fire immediately
///   'running'                      → crashed mid-run, fire immediately
///   no row                         → first boot, pick random deadline and sleep
///
/// Write order guarantees recovery:
///   1. Set state='running' before firing the callback
///   2. Set state='scheduled' + new next_run_at only after the callback returns
///
/// This means a crash during the callback is always detected on next startup.
pub struct TimerStore {
    db: Arc<Mutex<Connection>>,
}

impl TimerStore {
    pub fn new(db_path: &Path) -> Result<Arc<Self>> {
        let conn = Connection::open(db_path).context("Failed to open timer DB")?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS system_timers (
                 name        TEXT PRIMARY KEY,
                 next_run_at INTEGER NOT NULL,
                 state       TEXT NOT NULL DEFAULT 'scheduled'
             );",
        )
        .context("Failed to initialize system_timers table")?;
        Ok(Arc::new(Self {
            db: Arc::new(Mutex::new(conn)),
        }))
    }

    fn now_secs() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    fn fmt_ts(unix_secs: i64) -> String {
        use chrono::{TimeZone, Utc};
        Utc.timestamp_opt(unix_secs, 0)
            .single()
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| unix_secs.to_string())
    }

    /// Returns the initial wait in seconds before the first fire, and whether this is a recovery.
    /// Also transitions the row into the correct pre-sleep state.
    pub fn startup(&self, name: &str, min_secs: u64, max_secs: u64) -> Result<u64> {
        let conn = self.db.lock().unwrap();
        let now = Self::now_secs();

        let row: Option<(i64, String)> = conn
            .query_row(
                "SELECT next_run_at, state FROM system_timers WHERE name = ?1",
                params![name],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .ok();

        match row {
            None => {
                // First boot — schedule a fresh random deadline.
                let wait = rand::thread_rng().gen_range(min_secs..=max_secs);
                let next_run_at = now + wait as i64;
                conn.execute(
                    "INSERT INTO system_timers (name, next_run_at, state) VALUES (?1, ?2, 'scheduled')",
                    params![name, next_run_at],
                )?;
                info!(
                    timer = name,
                    in_hours = format!("{:.1}h", wait as f64 / 3600.0),
                    next_run_at = %Self::fmt_ts(next_run_at),
                    "Timer: first boot, scheduled"
                );
                Ok(wait)
            }
            Some((_, ref state)) if state == "running" => {
                // Crashed mid-run — fire immediately.
                warn!(timer = name, "Timer: crash-recovery detected (state=running), firing immediately");
                Ok(0)
            }
            Some((next_run_at, _)) if next_run_at <= now => {
                // Deadline already passed (was offline) — fire immediately.
                info!(timer = name, "Timer: deadline passed while offline, firing immediately");
                Ok(0)
            }
            Some((next_run_at, _)) => {
                // Deadline is still in the future — resume the countdown.
                let remaining = (next_run_at - now) as u64;
                info!(
                    timer = name,
                    next_run_at = %Self::fmt_ts(next_run_at),
                    "Timer: resuming persisted deadline"
                );
                Ok(remaining)
            }
        }
    }

    /// Mark the timer as running (call before invoking the callback).
    pub fn mark_running(&self, name: &str) -> Result<()> {
        let conn = self.db.lock().unwrap();
        conn.execute(
            "INSERT INTO system_timers (name, next_run_at, state) VALUES (?1, 0, 'running')
             ON CONFLICT(name) DO UPDATE SET state = 'running'",
            params![name],
        )?;
        Ok(())
    }

    /// Mark the timer as scheduled with a new random deadline (call after the callback returns).
    /// Returns (wait_secs, next_run_at_unix).
    pub fn mark_scheduled(&self, name: &str, min_secs: u64, max_secs: u64) -> Result<(u64, i64)> {
        let wait = rand::thread_rng().gen_range(min_secs..=max_secs);
        let next_run_at = Self::now_secs() + wait as i64;
        let conn = self.db.lock().unwrap();
        conn.execute(
            "INSERT INTO system_timers (name, next_run_at, state) VALUES (?1, ?2, 'scheduled')
             ON CONFLICT(name) DO UPDATE SET next_run_at = ?2, state = 'scheduled'",
            params![name, next_run_at],
        )?;
        Ok((wait, next_run_at))
    }
}

/// Spawn a persistent background timer that calls `callback` on a random interval
/// between `min_secs` and `max_secs`, surviving restarts and self-healing from crashes.
pub fn spawn<F, Fut>(
    name: impl Into<String>,
    store: Arc<TimerStore>,
    min_secs: u64,
    max_secs: u64,
    callback: F,
) where
    F: Fn() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let name = name.into();
    tokio::spawn(async move {
        let initial_wait = match store.startup(&name, min_secs, max_secs) {
            Ok(w) => w,
            Err(e) => {
                warn!(timer = %name, "Timer: startup failed: {e}, using fresh random interval");
                rand::thread_rng().gen_range(min_secs..=max_secs)
            }
        };

        tokio::time::sleep(std::time::Duration::from_secs(initial_wait)).await;

        loop {
            if let Err(e) = store.mark_running(&name) {
                warn!(timer = %name, "Timer: failed to mark running: {e}");
            }

            callback().await;

            match store.mark_scheduled(&name, min_secs, max_secs) {
                Ok((wait, next_run_at)) => {
                    info!(
                        timer = %name,
                        in_hours = format!("{:.1}h", wait as f64 / 3600.0),
                        next_run_at = %TimerStore::fmt_ts(next_run_at),
                        "Timer: next run scheduled (persisted)"
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                }
                Err(e) => {
                    warn!(timer = %name, "Timer: failed to persist next deadline: {e}, using in-memory interval");
                    let wait = rand::thread_rng().gen_range(min_secs..=max_secs);
                    let next_run_at = TimerStore::now_secs() + wait as i64;
                    info!(timer = %name, in_hours = format!("{:.1}h", wait as f64 / 3600.0), next_run_at = %TimerStore::fmt_ts(next_run_at), "Timer: next run scheduled (unpersisted)");
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                }
            }
        }
    });
}

/// Convenience: construct a `TimerStore` backed by the same DB path used elsewhere.
pub fn open(db_path: &PathBuf) -> Result<Arc<TimerStore>> {
    TimerStore::new(db_path)
}
