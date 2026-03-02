use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

const RECALL_HALF_LIFE_DAYS: f64 = 7.0;
const CORE_SCORE_BOOST: f64 = 0.3;
const SUPERSEDED_SCORE_GATE: f64 = 1.0;
const RETRACTED_SCORE_GATE: f64 = 1.3;
const RECALL_OVER_FETCH: usize = 10;
const RECALL_LIMIT: usize = 5;

pub struct MemoryStore {
    conn: Arc<Mutex<Connection>>,
    embedder: Mutex<TextEmbedding>,
}

impl MemoryStore {
    /// Get a reference to the shared database connection for use by other stores.
    pub fn connection(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    pub fn new(data_dir: &str) -> Result<Arc<Self>> {
        std::fs::create_dir_all(data_dir)?;
        let db_path = format!("{data_dir}/nina.db");
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
            CREATE TABLE IF NOT EXISTS memories_vec (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                memory_id INTEGER NOT NULL
            );",
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

    pub async fn remember(&self, content: &str, category: &str, source: &str) -> Result<()> {
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
                "INSERT INTO memories_meta(content, category, source, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?4)",
                params![content_clone, category, source, now],
            )?;
            let memory_id = conn.last_insert_rowid();

            let blob = embedding_to_bytes(&embedding);
            conn.execute(
                "INSERT INTO memories_vec(embedding, memory_id) VALUES (?1, ?2)",
                params![blob, memory_id],
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
    pub async fn update_memory(&self, query: &str, new_content: Option<String>) -> Result<String> {
        let query_embedding = self.embed(query)?;
        let query_str = query.to_string();

        let conn = self.conn.clone();
        let new_content_clone = new_content.clone();

        let result: Result<String> = tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();

            // Find best active match via hybrid search
            let escaped = format!("\"{}\"", query_str.replace('"', "\"\""));
            let mut scored: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();

            // FTS5 search restricted to active entries
            {
                let stmt_result = conn.prepare(
                    "SELECT m.id, fts.rank
                     FROM memories_fts fts
                     JOIN memories_meta m ON fts.rowid = m.id
                     WHERE memories_fts MATCH ?1 AND m.status = 'active'
                     ORDER BY rank LIMIT 10",
                );
                if let Ok(mut stmt) = stmt_result {
                    if let Ok(rows) = stmt.query_map(params![escaped], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
                    }) {
                        for row in rows.flatten() {
                            let score = 1.0 / (1.0 + (-row.1));
                            *scored.entry(row.0).or_default() += score;
                        }
                    }
                }
            }

            // Semantic search
            {
                let mut stmt = conn.prepare(
                    "SELECT mv.memory_id, mv.embedding
                     FROM memories_vec mv
                     JOIN memories_meta m ON mv.memory_id = m.id
                     WHERE m.status = 'active'",
                )?;
                let rows = stmt.query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
                })?;
                for row in rows.flatten() {
                    let emb = bytes_to_embedding(&row.1);
                    let sim = cosine_similarity(&query_embedding, &emb) as f64;
                    *scored.entry(row.0).or_default() += sim;
                }
            }

            let mut ranked: Vec<(i64, f64)> = scored.into_iter().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let best = match ranked.first() {
                Some(&(id, _)) => id,
                None => return Ok("No matching memory found.".into()),
            };

            // Fetch old memory details
            let (old_content, old_category): (String, String) = conn.query_row(
                "SELECT content, category FROM memories_meta WHERE id = ?1",
                params![best],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;

            let now = chrono::Utc::now().to_rfc3339();

            if let Some(ref new_text) = new_content_clone {
                // Correct: insert new, supersede old
                conn.execute(
                    "INSERT INTO memories_meta(content, category, source, created_at, updated_at)
                     VALUES (?1, ?2, 'agent', ?3, ?3)",
                    params![new_text, old_category, now],
                )?;
                let new_id = conn.last_insert_rowid();

                conn.execute(
                    "UPDATE memories_meta SET status='superseded', superseded_by=?1, updated_at=?2 WHERE id=?3",
                    params![new_id, now, best],
                )?;

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
        .context("spawn_blocking join error")?;

        // Embed and store the new memory if correction
        if let Some(ref new_text) = new_content {
            if result.as_ref().map(|s| s.starts_with("Corrected")).unwrap_or(false) {
                let embedding = self.embed(new_text)?;
                let blob = embedding_to_bytes(&embedding);
                // Get the new memory id (last inserted)
                let new_id: i64 = {
                    let conn = self.conn.lock().unwrap();
                    conn.query_row(
                        "SELECT id FROM memories_meta WHERE source='agent' ORDER BY id DESC LIMIT 1",
                        [],
                        |row| row.get(0),
                    )?
                };
                let conn = self.conn.lock().unwrap();
                conn.execute(
                    "INSERT INTO memories_vec(embedding, memory_id) VALUES (?1, ?2)",
                    params![blob, new_id],
                )?;
            }
        }

        result
    }

    pub async fn recall(&self, query: &str) -> Result<String> {
        let query_embedding = self.embed(query)?;
        let query_str = query.to_string();

        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            let now = chrono::Utc::now();

            // --- FTS5 search with metadata join ---
            let escaped = format!("\"{}\"", query_str.replace('"', "\"\""));
            let mut fts_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
            {
                let stmt_result = conn.prepare(
                    "SELECT m.id, fts.rank
                     FROM memories_fts fts
                     JOIN memories_meta m ON fts.rowid = m.id
                     WHERE memories_fts MATCH ?1
                     ORDER BY rank LIMIT ?2",
                );
                match stmt_result {
                    Ok(mut stmt) => {
                        if let Ok(rows) = stmt.query_map(params![escaped, RECALL_OVER_FETCH as i64], |row| {
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

            // --- Semantic search with metadata join ---
            let mut semantic_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
            {
                let mut stmt = conn.prepare(
                    "SELECT mv.memory_id, mv.embedding
                     FROM memories_vec mv
                     JOIN memories_meta m ON mv.memory_id = m.id",
                )?;
                let rows = stmt.query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
                })?;
                for row in rows.flatten() {
                    let emb = bytes_to_embedding(&row.1);
                    let sim = cosine_similarity(&query_embedding, &emb) as f64;
                    semantic_scores.insert(row.0, sim);
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

            // Fetch metadata for all candidates, apply decay + boost + gating
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

            let mut candidates: Vec<Candidate> = Vec::new();
            for (id, raw) in &raw_scores {
                let row = conn.query_row(
                    "SELECT content, category, status, source, created_at, superseded_by
                     FROM memories_meta WHERE id = ?1",
                    params![id],
                    |row| Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, Option<i64>>(5)?,
                    )),
                );
                if let Ok((content, category, status, source, created_at, superseded_by)) = row {
                    // Time decay for non-core
                    let age_days = parse_age_days(&created_at, &now);
                    let mut score = *raw;
                    if category != "core" {
                        score *= 2f64.powf(-age_days / RECALL_HALF_LIFE_DAYS);
                    }
                    // Core boost
                    if category == "core" {
                        score += CORE_SCORE_BOOST;
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

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

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
