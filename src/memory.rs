use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use std::sync::Once;
use tracing::{info, warn};

extern "C" {
    // The symbol exported by vec0.so is sqlite3_vec_init (not sqlite3_vec0_init).
    fn sqlite3_vec_init(
        db: *mut rusqlite::ffi::sqlite3,
        pz_err_msg: *mut *mut std::ffi::c_char,
        p_api: *const rusqlite::ffi::sqlite3_api_routines,
    ) -> std::ffi::c_int;
}

static SQLITE_VEC_REGISTERED: Once = Once::new();

fn register_sqlite_vec() {
    SQLITE_VEC_REGISTERED.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(sqlite3_vec_init));
    });
}

const RECALL_HALF_LIFE_DAYS: f64 = 7.0;
const CORE_SCORE_BOOST: f64 = 0.3;
const SUPERSEDED_SCORE_GATE: f64 = 1.0;
const RETRACTED_SCORE_GATE: f64 = 1.3;
const RECALL_OVER_FETCH: usize = 10;
const RECALL_LIMIT: usize = 5;
const RECALL_BOOST_MAX: f64 = 0.4;
const RECALL_BOOST_HALF_DAYS: f64 = 14.0; // inflection: boost = MAX/2 at this age
const RECALL_BOOST_STEEPNESS: f64 = 0.15;

pub struct MemoryStore {
    conn: Arc<Mutex<Connection>>,
    embedder: Mutex<TextEmbedding>,
}

impl MemoryStore {
    pub fn new(data_dir: &str) -> Result<Arc<Self>> {
        std::fs::create_dir_all(data_dir)?;
        let db_path = format!("{data_dir}/nina.db");

        // Must register before Connection::open so auto_extension fires on the new connection
        register_sqlite_vec();

        let conn = Connection::open(&db_path)?;

        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .context("Failed to set WAL mode")?;

        // New schema
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories_meta (
                id              INTEGER PRIMARY KEY,
                content         TEXT NOT NULL,
                category        TEXT NOT NULL DEFAULT 'conversation',
                status          TEXT NOT NULL DEFAULT 'active',
                retraction_note TEXT,
                superseded_by   INTEGER REFERENCES memories_meta(id),
                source          TEXT NOT NULL DEFAULT 'user',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content='memories_meta',
                content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories_meta BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories_meta BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories_meta BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[384]);",
        )
        .context("Failed to initialize memory database")?;

        // Migration: if old FTS5 `memories` table exists but `memories_meta` is empty, migrate
        let needs_migration: bool = {
            let old_exists: i64 = conn.query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memories'",
                [],
                |row| row.get(0),
            )?;
            let new_count: i64 = conn.query_row(
                "SELECT count(*) FROM memories_meta",
                [],
                |row| row.get(0),
            )?;
            old_exists > 0 && new_count == 0
        };

        if needs_migration {
            info!("Migrating old memories table to memories_meta...");
            conn.execute_batch(
                "INSERT OR IGNORE INTO memories_meta (id, content, category, source, created_at, updated_at)
                 SELECT rowid, content, 'conversation', COALESCE(source, 'user'), created_at, created_at
                 FROM memories;
                 DROP TABLE memories;",
            )
            .context("Failed to migrate old memories table")?;
            info!("Migration complete");
        }

        // Migration: move any rows from the old memories_vec blob table into vec_memories
        let old_vec_exists: i64 = conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memories_vec'",
            [],
            |row| row.get(0),
        )?;
        if old_vec_exists > 0 {
            info!("Migrating memories_vec → vec_memories...");
            let mut stmt = conn.prepare("SELECT memory_id, embedding FROM memories_vec")?;
            let rows: Vec<(i64, Vec<u8>)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .flatten()
                .collect();
            for (memory_id, blob) in rows {
                let _ = conn.execute(
                    "INSERT OR IGNORE INTO vec_memories(rowid, embedding) VALUES (?1, ?2)",
                    params![memory_id, blob],
                );
            }
            conn.execute_batch("DROP TABLE memories_vec;")?;
            info!("memories_vec migration complete");
        }

        // Additive migration: identity scoping for memories.
        // Ignores "duplicate column" errors on re-runs.
        let _ = conn.execute("ALTER TABLE memories_meta ADD COLUMN identity_id INTEGER", []);
        let _ = conn.execute("ALTER TABLE memories_meta ADD COLUMN last_recalled_at TEXT", []);
        // Assign existing unscoped memories to the owner identity.
        let _ = conn.execute(
            "UPDATE memories_meta
               SET identity_id = (SELECT id FROM identities WHERE access_level = 'owner' LIMIT 1)
             WHERE identity_id IS NULL",
            [],
        );

        info!("Loading embedding model (AllMiniLmL6V2)...");
        let mut init_opts = InitOptions::default();
        init_opts.model_name = EmbeddingModel::AllMiniLML6V2;
        init_opts.show_download_progress = true;
        let embedder = TextEmbedding::try_new(init_opts)
            .context("Failed to load embedding model")?;
        info!("Embedding model loaded");

        Ok(Arc::new(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: Mutex::new(embedder),
        }))
    }

    pub async fn remember(&self, content: &str, category: &str, source: &str, identity_id: Option<i64>) -> Result<()> {
        let content = content.to_string();
        let category = category.to_string();
        let source = source.to_string();
        let embedding = self.embed(&content)?;
        let now = chrono::Utc::now().to_rfc3339();

        let conn = self.conn.clone();
        let content_clone = content.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO memories_meta(content, category, source, identity_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
                params![content_clone, category, source, identity_id, now],
            )?;
            let memory_id = conn.last_insert_rowid();

            let blob = embedding_to_bytes(&embedding);
            conn.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES (?1, ?2)",
                params![memory_id, blob],
            )?;

            info!("Stored memory (id={memory_id}): {}", truncate_str(&content_clone, 80));
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("spawn_blocking join error")??;

        Ok(())
    }

    /// Correct or retract a memory by fuzzy search.
    /// If `new_content` is Some, supersede old with new (correct).
    /// If `new_content` is None, mark old as retracted.
    pub async fn update_memory(&self, query: &str, new_content: Option<String>, identity_id: Option<i64>) -> Result<String> {
        let query_embedding = self.embed(query)?;
        // Pre-generate the new embedding here (TextEmbedding is !Send, can't cross into spawn_blocking)
        let new_embedding: Option<Vec<f32>> = match &new_content {
            Some(text) => Some(self.embed(text)?),
            None => None,
        };
        let query_str = query.to_string();

        let conn = self.conn.clone();
        let new_content_clone = new_content.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();

            // Find best active match via hybrid search
            let escaped = format!("\"{}\"", query_str.replace('"', "\"\""));
            let mut scored: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();

            // FTS5 search restricted to active entries for this identity
            {
                let stmt_result = conn.prepare(
                    "SELECT m.id, fts.rank
                     FROM memories_fts fts
                     JOIN memories_meta m ON fts.rowid = m.id
                     WHERE memories_fts MATCH ?1 AND m.status = 'active'
                       AND (?2 IS NULL OR m.identity_id = ?2 OR m.identity_id IS NULL)
                     ORDER BY rank LIMIT 10",
                );
                if let Ok(mut stmt) = stmt_result {
                    if let Ok(rows) = stmt.query_map(params![escaped, identity_id], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
                    }) {
                        for row in rows.flatten() {
                            let score = 1.0 / (1.0 + (-row.1));
                            *scored.entry(row.0).or_default() += score;
                        }
                    }
                }
            }

            // Semantic search via sqlite-vec KNN (replaces full BLOB scan)
            {
                let blob = embedding_to_bytes(&query_embedding);
                let stmt_result = conn.prepare(
                    "SELECT rowid, distance FROM vec_memories
                     WHERE embedding MATCH ?1 ORDER BY distance LIMIT 20",
                );
                if let Ok(mut stmt) = stmt_result {
                    if let Ok(rows) = stmt.query_map(params![blob], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
                    }) {
                        for row in rows.flatten() {
                            let sim = 1.0 / (1.0 + row.1);
                            *scored.entry(row.0).or_default() += sim;
                        }
                    }
                }
            }

            // Filter to active + identity-scoped entries
            scored.retain(|id, _| {
                conn.query_row(
                    "SELECT 1 FROM memories_meta WHERE id = ?1 AND status = 'active'
                       AND (?2 IS NULL OR identity_id = ?2 OR identity_id IS NULL)",
                    params![id, identity_id],
                    |_| Ok(()),
                ).is_ok()
            });

            let mut ranked: Vec<(i64, f64)> = scored.into_iter().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let best = match ranked.first() {
                Some(&(id, _)) => id,
                None => return Ok("No matching memory found.".into()),
            };

            let (old_content, old_category): (String, String) = conn.query_row(
                "SELECT content, category FROM memories_meta WHERE id = ?1",
                params![best],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;

            let now = chrono::Utc::now().to_rfc3339();

            if let Some(ref new_text) = new_content_clone {
                // Correct: insert new memory, supersede old
                conn.execute(
                    "INSERT INTO memories_meta(content, category, source, identity_id, created_at, updated_at)
                     VALUES (?1, ?2, 'agent', ?3, ?4, ?4)",
                    params![new_text, old_category, identity_id, now],
                )?;
                let new_id = conn.last_insert_rowid();

                conn.execute(
                    "UPDATE memories_meta SET status='superseded', superseded_by=?1, updated_at=?2 WHERE id=?3",
                    params![new_id, now, best],
                )?;

                // Store the new embedding — new_id is clean from last_insert_rowid()
                if let Some(ref emb) = new_embedding {
                    let blob = embedding_to_bytes(emb);
                    let _ = conn.execute(
                        "INSERT INTO vec_memories(rowid, embedding) VALUES (?1, ?2)",
                        params![new_id, blob],
                    );
                }

                Ok(format!("Corrected: '{}' → '{}'", old_content, new_text))
            } else {
                // Retract
                conn.execute(
                    "UPDATE memories_meta SET status='retracted', retraction_note='retracted', updated_at=?1 WHERE id=?2",
                    params![now, best],
                )?;

                Ok(format!("Retracted: '{}'", old_content))
            }
        })
        .await
        .context("spawn_blocking join error")?
    }

    pub async fn recall(&self, query: &str, identity_id: Option<i64>) -> Result<String> {
        let query_embedding = self.embed(query)?;
        let query_str = query.to_string();

        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = chrono::Utc::now();

            // --- FTS5 search with identity filter ---
            let escaped = format!("\"{}\"", query_str.replace('"', "\"\""));
            let mut fts_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
            {
                let stmt_result = conn.prepare(
                    "SELECT m.id, fts.rank
                     FROM memories_fts fts
                     JOIN memories_meta m ON fts.rowid = m.id
                     WHERE memories_fts MATCH ?1
                       AND (?3 IS NULL OR m.identity_id = ?3 OR m.identity_id IS NULL)
                     ORDER BY rank LIMIT ?2",
                );
                match stmt_result {
                    Ok(mut stmt) => {
                        if let Ok(rows) = stmt.query_map(params![escaped, RECALL_OVER_FETCH as i64, identity_id], |row| {
                            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
                        }) {
                            for row in rows.flatten() {
                                let score = 1.0 / (1.0 + (-row.1));
                                fts_scores.insert(row.0, score);
                            }
                        }
                    }
                    Err(e) => warn!("FTS5 prepare failed: {e}"),
                }
            }

            // --- Semantic search via sqlite-vec KNN (replaces full BLOB scan) ---
            // Over-fetch to absorb identity filtering in post-processing.
            // AllMiniLML6V2 produces unit-normalised vectors so L2 ≈ cosine distance.
            let mut semantic_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
            {
                let blob = embedding_to_bytes(&query_embedding);
                let knn_limit = (RECALL_OVER_FETCH * 4) as i64;
                let stmt_result = conn.prepare(
                    "SELECT rowid, distance FROM vec_memories
                     WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
                );
                match stmt_result {
                    Ok(mut stmt) => {
                        if let Ok(rows) = stmt.query_map(params![blob, knn_limit], |row| {
                            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
                        }) {
                            for row in rows.flatten() {
                                // distance ∈ [0, 2] for unit vectors → similarity ∈ (0, 1]
                                let sim = 1.0 / (1.0 + row.1);
                                semantic_scores.insert(row.0, sim);
                            }
                        }
                    }
                    Err(e) => warn!("vec0 KNN prepare failed: {e}"),
                }
            }

            // Merge raw scores
            let mut raw_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
            for (id, s) in &fts_scores {
                *raw_scores.entry(*id).or_default() += s;
            }
            for (id, s) in &semantic_scores {
                *raw_scores.entry(*id).or_default() += s;
            }

            // Batch-fetch metadata for all candidates (single query, not N+1)
            struct Candidate {
                id: i64,
                content: String,
                _category: String,
                status: String,
                source: String,
                created_at: String,
                superseded_by: Option<i64>,
                final_score: f64,
            }

            let all_ids: Vec<i64> = raw_scores.keys().copied().collect();
            let mut meta_map: std::collections::HashMap<
                i64,
                (String, String, String, String, String, Option<i64>, Option<String>),
            > = std::collections::HashMap::new();
            if !all_ids.is_empty() {
                let placeholders = all_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql = format!(
                    "SELECT id, content, category, status, source, created_at, superseded_by, last_recalled_at
                     FROM memories_meta WHERE id IN ({placeholders})"
                );
                if let Ok(mut stmt) = conn.prepare(&sql) {
                    if let Ok(rows) = stmt.query_map(
                        rusqlite::params_from_iter(all_ids.iter()),
                        |row| {
                            Ok((
                                row.get::<_, i64>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, String>(3)?,
                                row.get::<_, String>(4)?,
                                row.get::<_, String>(5)?,
                                row.get::<_, Option<i64>>(6)?,
                                row.get::<_, Option<String>>(7)?,
                            ))
                        },
                    ) {
                        for row in rows.flatten() {
                            // identity filter applied here, covering the vec0 over-fetch
                            let passes_identity = identity_id.is_none()
                                || row.6.is_none()
                                || row.6 == identity_id;
                            if passes_identity {
                                meta_map.insert(row.0, (row.1, row.2, row.3, row.4, row.5, row.6, row.7));
                            }
                        }
                    }
                }
            }

            let mut candidates: Vec<Candidate> = Vec::new();
            for (id, raw) in &raw_scores {
                if let Some((content, category, status, source, created_at, superseded_by, last_recalled_at)) =
                    meta_map.get(id)
                {
                    let (content, category, status, source, created_at, superseded_by, last_recalled_at) = (
                        content.clone(), category.clone(), status.clone(),
                        source.clone(), created_at.clone(), *superseded_by, last_recalled_at.clone(),
                    );
                    // Time decay for non-core (from creation date — unchanged)
                    let age_days = parse_age_days(&created_at, &now);
                    let mut score = *raw;
                    if category != "core" {
                        score *= 2f64.powf(-age_days / RECALL_HALF_LIFE_DAYS);
                    }
                    // Core boost
                    if category == "core" {
                        score += CORE_SCORE_BOOST;
                    }
                    // Recall recency boost: S-curve bounded to RECALL_BOOST_MAX
                    // boost = MAX / (1 + exp(k * (days_since_recall - T_half)))
                    // → ~MAX when recalled recently, → 0 when recalled long ago or never
                    if let Some(ref lr) = last_recalled_at {
                        let days_since = parse_age_days(lr, &now);
                        let boost = RECALL_BOOST_MAX
                            / (1.0 + (RECALL_BOOST_STEEPNESS * (days_since - RECALL_BOOST_HALF_DAYS)).exp());
                        score += boost;
                    }
                    // Status gating
                    let passes = match status.as_str() {
                        "active" => true,
                        "superseded" => score >= SUPERSEDED_SCORE_GATE,
                        "retracted" => score >= RETRACTED_SCORE_GATE,
                        _ => false,
                    };
                    if passes {
                        candidates.push(Candidate {
                            id: *id,
                            content,
                            _category: category,
                            status,
                            source,
                            created_at,
                            superseded_by,
                            final_score: score,
                        });
                    }
                }
            }

            // Sort by score descending, take top RECALL_OVER_FETCH then trim to RECALL_LIMIT
            candidates.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(RECALL_OVER_FETCH);

            // Stamp last_recalled_at for top 25% of scored candidates (at least 1)
            let stamp_count = ((candidates.len() as f64 * 0.25).ceil() as usize).max(1);
            let now_str = now.to_rfc3339();
            for c in candidates.iter().take(stamp_count) {
                let _ = conn.execute(
                    "UPDATE memories_meta SET last_recalled_at = ?1 WHERE id = ?2",
                    params![now_str, c.id],
                );
            }

            if candidates.is_empty() {
                return Ok("No memories found.".into());
            }

            // Ensure superseded entries have their active replacement present
            let active_ids: std::collections::HashSet<i64> = candidates.iter()
                .filter(|c| c.status == "active")
                .map(|c| c.id)
                .collect();

            let superseded_needing_parent: Vec<i64> = candidates.iter()
                .filter(|c| c.status == "superseded")
                .filter_map(|c| c.superseded_by)
                .filter(|pid| !active_ids.contains(pid))
                .collect();

            // Fetch missing active parents (just for rendering, not scored)
            struct Extra {
                id: i64,
                content: String,
                source: String,
                created_at: String,
            }
            let mut extras: Vec<Extra> = Vec::new();
            for pid in superseded_needing_parent {
                if let Ok((content, source, created_at)) = conn.query_row(
                    "SELECT content, source, created_at FROM memories_meta WHERE id = ?1",
                    params![pid],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)),
                ) {
                    extras.push(Extra { id: pid, content, source, created_at });
                }
            }

            // Build a lookup: superseded_by -> superseded entry content+date
            // For each active entry, find if any candidate is its superseded predecessor
            let superseded_lookup: std::collections::HashMap<i64, &Candidate> = candidates.iter()
                .filter(|c| c.status == "superseded")
                .filter_map(|c| c.superseded_by.map(|pid| (pid, c)))
                .collect();

            let extra_lookup: std::collections::HashMap<i64, &Extra> = extras.iter()
                .map(|e| (e.id, e))
                .collect();

            // Collect active candidates (limited) + handle superseded predecessors
            let active: Vec<&Candidate> = candidates.iter()
                .filter(|c| c.status == "active")
                .take(RECALL_LIMIT)
                .collect();

            let retracted: Vec<&Candidate> = candidates.iter()
                .filter(|c| c.status == "retracted")
                .collect();

            // Render output
            let mut lines: Vec<String> = Vec::new();
            lines.push("[memory]".into());

            // Active entries + their superseded predecessors
            for entry in &active {
                let date = format_date(&entry.created_at);
                let header = if entry.source == "agent" {
                    format!("[{}, agent] {}", date, entry.content)
                } else {
                    format!("[{}] {}", date, entry.content)
                };
                lines.push(header);

                // Look for superseded predecessor: a candidate whose superseded_by == this entry's id
                let pred = superseded_lookup.get(&entry.id);
                if let Some(pred) = pred {
                    lines.push(format!("  [{}, was] {}", format_date(&pred.created_at), pred.content));
                }
            }

            // Extra active parents for orphaned superseded entries (add them if not already listed)
            // These are active entries fetched just to pair with superseded children
            let already_rendered: std::collections::HashSet<i64> = active.iter().map(|c| c.id).collect();
            for (pid, extra) in &extra_lookup {
                if !already_rendered.contains(pid) {
                    let date = format_date(&extra.created_at);
                    let header = if extra.source == "agent" {
                        format!("[{}, agent] {}", date, extra.content)
                    } else {
                        format!("[{}] {}", date, extra.content)
                    };
                    lines.push(header);
                    // Find its superseded child
                    for c in &candidates {
                        if c.status == "superseded" && c.superseded_by == Some(*pid) {
                            lines.push(format!("  [{}, was] {}", format_date(&c.created_at), c.content));
                        }
                    }
                }
            }

            // Retracted section
            if !retracted.is_empty() {
                lines.push(String::new());
                lines.push("[retracted]".into());
                for entry in retracted {
                    lines.push(format!("[{}] {}", format_date(&entry.created_at), entry.content));
                }
            }

            Ok(lines.join("\n"))
        })
        .await
        .context("spawn_blocking join error")?
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut embedder = self.embedder.lock().unwrap();
        let mut embeddings = embedder
            .embed(vec![text], None)
            .context("Embedding failed")?;
        embeddings.pop().context("No embedding returned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // --- embedding_to_bytes / bytes_to_embedding ---

    #[test]
    fn embedding_roundtrip() {
        let emb: Vec<f32> = vec![0.1, -0.5, 0.9, 1.0, -1.0, 0.0];
        let bytes = embedding_to_bytes(&emb);
        let back = bytes_to_embedding(&bytes);
        assert_eq!(emb.len(), back.len());
        for (a, b) in emb.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn embedding_to_bytes_correct_length() {
        let emb: Vec<f32> = vec![1.0, 2.0, 3.0];
        assert_eq!(embedding_to_bytes(&emb).len(), 12); // 3 * 4 bytes
    }

    #[test]
    fn embedding_empty_roundtrip() {
        let emb: Vec<f32> = vec![];
        assert!(bytes_to_embedding(&embedding_to_bytes(&emb)).is_empty());
    }

    // --- cosine_similarity ---

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let a = vec![0.0f32, 0.0];
        let b = vec![1.0f32, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_length_mismatch_returns_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_range_minus_one_to_one() {
        let a = vec![0.3f32, 0.4, 0.5];
        let b = vec![0.1f32, 0.9, 0.2];
        let sim = cosine_similarity(&a, &b);
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    // --- parse_age_days ---

    #[test]
    fn parse_age_days_just_created() {
        let now = Utc::now();
        let ts = now.to_rfc3339();
        let age = parse_age_days(&ts, &now);
        assert!(age < 0.01, "Expected near-zero age, got {age}");
    }

    #[test]
    fn parse_age_days_one_day_ago() {
        let now = Utc::now();
        let yesterday = now - chrono::Duration::days(1);
        let age = parse_age_days(&yesterday.to_rfc3339(), &now);
        assert!((age - 1.0).abs() < 0.01, "Expected ~1.0, got {age}");
    }

    #[test]
    fn parse_age_days_seven_days() {
        let now = Utc::now();
        let week_ago = now - chrono::Duration::days(7);
        let age = parse_age_days(&week_ago.to_rfc3339(), &now);
        assert!((age - 7.0).abs() < 0.01, "Expected ~7.0, got {age}");
    }

    #[test]
    fn parse_age_days_invalid_returns_zero() {
        let now = Utc::now();
        assert_eq!(parse_age_days("not-a-date", &now), 0.0);
        assert_eq!(parse_age_days("", &now), 0.0);
    }

    #[test]
    fn parse_age_days_future_clamped_to_zero() {
        let now = Utc::now();
        let future = now + chrono::Duration::days(1);
        let age = parse_age_days(&future.to_rfc3339(), &now);
        assert_eq!(age, 0.0, "Future timestamps should return 0");
    }

    // --- format_date ---

    #[test]
    fn format_date_extracts_yyyy_mm_dd() {
        assert_eq!(format_date("2024-01-15T10:30:00Z"), "2024-01-15");
    }

    #[test]
    fn format_date_with_offset() {
        assert_eq!(format_date("2025-12-31T23:59:59+05:30"), "2025-12-31");
    }

    #[test]
    fn format_date_exact_ten_chars() {
        assert_eq!(format_date("2024-01-15"), "2024-01-15");
    }

    #[test]
    fn format_date_short_string_returned_as_is() {
        assert_eq!(format_date("2024-01"), "2024-01");
    }

    #[test]
    fn format_date_empty_returned_as_is() {
        assert_eq!(format_date(""), "");
    }
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[allow(dead_code)]
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a * norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let end = s.floor_char_boundary(max);
        &s[..end]
    }
}

/// Parse ISO8601/RFC3339 created_at and return age in days from `now`.
fn parse_age_days(created_at: &str, now: &chrono::DateTime<chrono::Utc>) -> f64 {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_at) {
        let duration = now.signed_duration_since(dt.with_timezone(&chrono::Utc));
        duration.num_seconds().max(0) as f64 / 86400.0
    } else {
        0.0
    }
}

/// Format a created_at string to YYYY-MM-DD.
fn format_date(created_at: &str) -> String {
    if created_at.len() >= 10 {
        created_at[..10].to_string()
    } else {
        created_at.to_string()
    }
}
