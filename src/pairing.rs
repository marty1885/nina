use anyhow::{Context, Result, bail};
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::channel::ChannelTarget;

pub struct PairingRequest {
    pub id: i64,
    pub code: String,
    pub channel_id: String,
    pub sender_id: String,
    pub chat_id: i64,
    pub display_name: String,
    pub created_at: i64,
}

pub struct ApprovedInfo {
    pub identity_id: i64,
    pub display_name: String,
    pub channel_id: String,
    pub chat_id: i64,
}

pub struct PairingStore {
    db: Arc<Mutex<Connection>>,
}

impl PairingStore {
    pub fn new(db_path: &Path) -> Result<Arc<Self>> {
        let conn = Connection::open(db_path).context("Failed to open pairing DB")?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;

             CREATE TABLE IF NOT EXISTS identities (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 display_name TEXT NOT NULL,
                 created_at INTEGER NOT NULL
             );

             CREATE TABLE IF NOT EXISTS channel_identities (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 identity_id INTEGER NOT NULL REFERENCES identities(id),
                 channel_id TEXT NOT NULL,
                 sender_id TEXT NOT NULL,
                 display_name TEXT,
                 UNIQUE(channel_id, sender_id)
             );

             CREATE TABLE IF NOT EXISTS pairing_requests (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 code TEXT NOT NULL UNIQUE,
                 channel_id TEXT NOT NULL,
                 sender_id TEXT NOT NULL,
                 chat_id INTEGER NOT NULL,
                 display_name TEXT NOT NULL,
                 state TEXT NOT NULL DEFAULT 'pending',
                 notified INTEGER NOT NULL DEFAULT 0,
                 created_at INTEGER NOT NULL,
                 decided_at INTEGER
             );

",
        )
        .context("Failed to initialize pairing tables")?;

        // Additive migrations — ignore "duplicate column" errors on re-runs.
        let _ = conn.execute("ALTER TABLE identities ADD COLUMN access_level TEXT", []);
        let _ = conn.execute("ALTER TABLE identities ADD COLUMN relation TEXT", []);
        Ok(Arc::new(Self {
            db: Arc::new(Mutex::new(conn)),
        }))
    }

    /// Returns the identity_id for a known sender, or None if unpaired.
    pub fn find_identity(&self, channel_id: &str, sender_id: &str) -> Result<Option<i64>> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT identity_id FROM channel_identities
             WHERE channel_id = ?1 AND sender_id = ?2",
        )?;
        let mut rows = stmt.query(params![channel_id, sender_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    /// Returns (identity_id, access_level, relation) for a known sender, or None if unpaired.
    /// access_level and relation are stored on the identity (cross-channel).
    pub fn find_identity_full(
        &self,
        channel_id: &str,
        sender_id: &str,
    ) -> Result<Option<(i64, Option<String>, Option<String>)>> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT ci.identity_id, i.access_level, i.relation
             FROM channel_identities ci
             JOIN identities i ON i.id = ci.identity_id
             WHERE ci.channel_id = ?1 AND ci.sender_id = ?2",
        )?;
        let mut rows = stmt.query(params![channel_id, sender_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some((row.get(0)?, row.get(1)?, row.get(2)?)))
        } else {
            Ok(None)
        }
    }

    /// Set the access level for a user (applies across all their channels).
    pub fn set_access_level(
        &self,
        channel_id: &str,
        sender_id: &str,
        level: &str,
    ) -> Result<()> {
        let conn = self.db.lock().unwrap();
        let rows = conn.execute(
            "UPDATE identities SET access_level = ?1
             WHERE id = (SELECT identity_id FROM channel_identities
                         WHERE channel_id = ?2 AND sender_id = ?3)",
            params![level, channel_id, sender_id],
        )?;
        if rows == 0 {
            bail!("No paired user found for channel={} sender={}", channel_id, sender_id);
        }
        Ok(())
    }

    /// Set the relation for a user (applies across all their channels).
    pub fn set_relation(
        &self,
        channel_id: &str,
        sender_id: &str,
        relation: &str,
    ) -> Result<()> {
        let conn = self.db.lock().unwrap();
        let rows = conn.execute(
            "UPDATE identities SET relation = ?1
             WHERE id = (SELECT identity_id FROM channel_identities
                         WHERE channel_id = ?2 AND sender_id = ?3)",
            params![relation, channel_id, sender_id],
        )?;
        if rows == 0 {
            bail!("No paired user found for channel={} sender={}", channel_id, sender_id);
        }
        Ok(())
    }

    /// Returns true if there is already a pending pairing request for this sender.
    pub fn has_pending_request(&self, channel_id: &str, sender_id: &str) -> Result<bool> {
        let conn = self.db.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pairing_requests
             WHERE channel_id = ?1 AND sender_id = ?2 AND state = 'pending'",
            params![channel_id, sender_id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Create a new pairing request and return the generated code.
    pub fn create_request(
        &self,
        channel_id: &str,
        sender_id: &str,
        chat_id: i64,
        display_name: &str,
    ) -> Result<String> {
        let code = generate_code();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let conn = self.db.lock().unwrap();
        conn.execute(
            "INSERT INTO pairing_requests (code, channel_id, sender_id, chat_id, display_name, state, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 'pending', ?6)",
            params![code, channel_id, sender_id, chat_id, display_name, now],
        )
        .context("Failed to insert pairing request")?;

        info!(code, channel_id, sender_id, display_name, "Created pairing request");
        Ok(code)
    }

    /// List all pending pairing requests.
    pub fn list_pending(&self) -> Result<Vec<PairingRequest>> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, code, channel_id, sender_id, chat_id, display_name, created_at
             FROM pairing_requests
             WHERE state = 'pending'
             ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map(params![], |row| {
            Ok(PairingRequest {
                id: row.get(0)?,
                code: row.get(1)?,
                channel_id: row.get(2)?,
                sender_id: row.get(3)?,
                chat_id: row.get(4)?,
                display_name: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    }

    /// Approve a pairing request by code: atomically creates identity + channel_identity.
    pub fn approve(&self, code: &str, is_owner: bool) -> Result<ApprovedInfo> {
        let conn = self.db.lock().unwrap();

        // Fetch the pending request
        let (channel_id, sender_id, chat_id, display_name): (String, String, i64, String) = conn
            .query_row(
                "SELECT channel_id, sender_id, chat_id, display_name
                 FROM pairing_requests
                 WHERE code = ?1 AND state = 'pending'",
                params![code],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .context("Pairing request not found or not pending")?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let access_level = if is_owner { "owner" } else { "member" };

        // Create identity (access_level lives here, cross-channel)
        conn.execute(
            "INSERT INTO identities (display_name, created_at, access_level) VALUES (?1, ?2, ?3)",
            params![display_name, now, access_level],
        )?;
        let identity_id = conn.last_insert_rowid();

        // Create channel_identity
        conn.execute(
            "INSERT INTO channel_identities (identity_id, channel_id, sender_id, display_name)
             VALUES (?1, ?2, ?3, ?4)",
            params![identity_id, channel_id, sender_id, display_name],
        )?;

        // Mark request as approved (not yet notified)
        conn.execute(
            "UPDATE pairing_requests SET state = 'approved', decided_at = ?1
             WHERE code = ?2",
            params![now, code],
        )?;

        info!(code, display_name, identity_id, "Approved pairing request");

        Ok(ApprovedInfo {
            identity_id,
            display_name,
            channel_id,
            chat_id,
        })
    }

    /// Reject a pairing request by code.
    pub fn reject(&self, code: &str) -> Result<()> {
        let conn = self.db.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let rows = conn.execute(
            "UPDATE pairing_requests SET state = 'rejected', decided_at = ?1
             WHERE code = ?2 AND state = 'pending'",
            params![now, code],
        )?;

        if rows == 0 {
            bail!("Pairing request '{}' not found or not pending", code);
        }

        info!(code, "Rejected pairing request");
        Ok(())
    }

    /// If there is an approved-but-unnotified pairing for this sender, mark it notified
    /// and return (target, message). Called on every message from the sender.
    pub fn pop_pending_notification(
        &self,
        channel_id: &str,
        sender_id: &str,
    ) -> Result<Option<(ChannelTarget, String)>> {
        let conn = self.db.lock().unwrap();

        let row = conn
            .query_row(
                "SELECT id, chat_id, display_name
                 FROM pairing_requests
                 WHERE channel_id = ?1 AND sender_id = ?2 AND state = 'approved' AND notified = 0",
                params![channel_id, sender_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;

        if let Some((req_id, chat_id, display_name)) = row {
            conn.execute(
                "UPDATE pairing_requests SET notified = 1 WHERE id = ?1",
                params![req_id],
            )?;
            let target = ChannelTarget {
                channel_id: channel_id.to_string(),
                chat_id,
            };
            let msg = format!(
                "You've been approved, welcome {}! You can now chat with me normally.",
                display_name
            );
            Ok(Some((target, msg)))
        } else {
            Ok(None)
        }
    }
}

/// Generate a 6-character uppercase hex code.
fn generate_code() -> String {
    let micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_micros();
    // Mix with a pseudo-random value from the address of a stack local
    let rand_bits = {
        let x: u32 = 0;
        (&x as *const u32 as usize) as u32
    };
    let combined = (micros ^ rand_bits) & 0x00FF_FFFF;
    format!("{:06X}", combined)
}

/// Extension trait on `rusqlite::Statement` for optional row queries.
trait OptionalExt<T> {
    fn optional(self) -> rusqlite::Result<Option<T>>;
}

impl<T> OptionalExt<T> for rusqlite::Result<T> {
    fn optional(self) -> rusqlite::Result<Option<T>> {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}
