mod agent;
mod channel;
mod config;
mod conversation;
mod gateway;
mod identity;
mod llm;
mod memory;
mod pairing;
mod reminders;
mod scheduler;
mod self_awareness;
mod session;
mod setup;
mod skills;
mod telegram;
mod timer;
mod tools;
mod tts;

use agent::Agent;
use channel::{CallContext, ChannelRouter, IncomingMessage};
use clap::{Parser, Subcommand};
use config::Config;
use conversation::ConversationStore;
use gateway::GatewayMap;
use pairing::PairingStore;
use reminders::{ReminderStore, spawn_reminder_task};
use session::SessionManager;
use skills::SkillRegistry;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;
use tools::ToolRegistry;
use tracing::{error, info};

// ---- CLI ----------------------------------------------------------------

#[derive(Parser)]
#[command(name = "nina", about = "AI assistant")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Manage pairing requests
    Pair {
        #[command(subcommand)]
        action: PairAction,
    },
    /// Run the interactive setup wizard
    Setup,
}

#[derive(Subcommand)]
enum PairAction {
    /// List pending pairing requests
    List,
    /// Approve a pairing request by code
    Approve {
        code: String,
        /// Grant owner-level access
        #[arg(long)]
        owner: bool,
    },
    /// Reject a pairing request by code
    Reject { code: String },
    /// Set access level for an already-paired user
    SetAccess {
        channel_id: String,
        sender_id: String,
        /// Access level: owner or member
        level: String,
    },
    /// Set relation disposition for an already-paired user
    SetRelation {
        channel_id: String,
        sender_id: String,
        /// Relation: warm, neutral, cold, or hostile
        relation: String,
    },
}

// ---- Active-processing state --------------------------------------------

struct ActiveProcessing {
    message_text: String,
    branch_id: u64,
}

// ---- Entry point ---------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nina=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();
    let config = Config::load()?;
    let db_path = config.data_dir().join("nina.db");

    match cli.command {
        Some(Command::Pair { action }) => run_pair_command(action, &db_path),
        Some(Command::Setup) => {
            std::fs::create_dir_all(&config.home_dir)?;
            std::fs::create_dir_all(config.context_dir())?;
            setup::run_setup(&config).await
        }
        None => run_server(config).await,
    }
}

// ---- CLI sub-command -----------------------------------------------------

fn run_pair_command(action: PairAction, db_path: &Path) -> anyhow::Result<()> {
    let store = PairingStore::new(db_path)?;

    match action {
        PairAction::List => {
            let requests = store.list_pending()?;
            if requests.is_empty() {
                println!("No pending pairing requests.");
            } else {
                println!("{:<8} {:<10} {:<20} {}", "CODE", "CHANNEL", "SENDER", "NAME");
                println!("{}", "-".repeat(60));
                for r in requests {
                    println!(
                        "{:<8} {:<10} {:<20} {}",
                        r.code, r.channel_id, r.sender_id, r.display_name
                    );
                }
            }
            Ok(())
        }
        PairAction::Approve { code, owner } => {
            let info = store.approve(&code, owner)?;
            let level = if owner { "owner" } else { "member" };
            println!(
                "Approved: {} (identity #{}) on {}:{} as {}",
                info.display_name, info.identity_id, info.channel_id, info.chat_id, level
            );
            Ok(())
        }
        PairAction::Reject { code } => {
            store.reject(&code)?;
            println!("Rejected pairing request {code}.");
            Ok(())
        }
        PairAction::SetAccess { channel_id, sender_id, level } => {
            store.set_access_level(&channel_id, &sender_id, &level)?;
            println!("Access level for {sender_id} on {channel_id} set to {level}.");
            Ok(())
        }
        PairAction::SetRelation { channel_id, sender_id, relation } => {
            store.set_relation(&channel_id, &sender_id, &relation)?;
            println!("Relation for {sender_id} on {channel_id} set to {relation}.");
            Ok(())
        }
    }
}

// ---- Helpers -------------------------------------------------------------

fn build_weighted(providers: &[(config::ResolvedProvider, u32)]) -> std::sync::Arc<dyn llm::LlmProvider> {
    let pool = providers
        .iter()
        .map(|(p, w)| {
            let provider: std::sync::Arc<dyn llm::LlmProvider> =
                std::sync::Arc::new(llm::OpenAiProvider::new(&p.base_url, &p.api_key, &p.model));
            (provider, *w)
        })
        .collect();
    std::sync::Arc::new(llm::WeightedProvider::new(pool))
}

/// Build the LLM handle for a section. If a fallback tier is provided, wraps both
/// in a `TieredProvider` (primary exhausted → fallback). Otherwise returns the
/// primary `WeightedProvider` directly.
fn build_llm(
    primary: &[(config::ResolvedProvider, u32)],
    fallback: &[(config::ResolvedProvider, u32)],
) -> std::sync::Arc<dyn llm::LlmProvider> {
    let primary_tier = build_weighted(primary);
    if fallback.is_empty() {
        return primary_tier;
    }
    std::sync::Arc::new(llm::TieredProvider::new(vec![primary_tier, build_weighted(fallback)]))
}

// ---- Server --------------------------------------------------------------

async fn run_server(config: Config) -> anyhow::Result<()> {
    let data_dir = config.data_dir();
    let soul_file = config.soul_file();
    let context_dir = config.context_dir();
    let scheduler_file = config.scheduler_file();

    // Ensure home subdirectories exist.
    std::fs::create_dir_all(&config.home_dir)?;
    std::fs::create_dir_all(&data_dir)?;
    std::fs::create_dir_all(&context_dir)?;

    // Ensure workspace exists and set CWD.
    std::fs::create_dir_all(&config.workspace_dir)?;
    std::env::set_current_dir(&config.workspace_dir)?;

    // Auto-bootstrap if no soul.md exists
    if !soul_file.exists() {
        eprintln!(
            "No soul.md found at {} — bootstrapping...",
            soul_file.display()
        );
        setup::run_setup(&config).await?;
    }

    let agent_name = config.agent_name();

    info!(
        home = %config.home_dir.display(),
        workspace = %config.workspace_dir.display(),
        name = %agent_name,
        "Directories initialized"
    );

    let data_dir_str = data_dir.to_string_lossy().to_string();
    let soul_file_str = soul_file.to_string_lossy().to_string();
    let context_dir_str = context_dir.to_string_lossy().to_string();

    info!("Loading memory store...");
    let memory = memory::MemoryStore::new(&data_dir_str)?;

    info!("Loading conversation store...");
    let conv_store = ConversationStore::new(&data_dir_str)?;

    // Reminder + pairing stores share the same DB file.
    let db_path = data_dir.join("nina.db");
    let reminder_store = ReminderStore::new(&db_path)?;
    let pairing_store = PairingStore::new(&db_path)?;

    let http_client = Arc::new(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()?,
    );

    // Build channel router
    let mut router = ChannelRouter::new();
    router.register(Box::new(
        channel::telegram::TelegramBot::new(&config.telegram_bot_token),
    ));
    let router = Arc::new(RwLock::new(router));

    // Shared OnceLock so ReminderSkill can get the agent after it's constructed.
    let agent_cell: Arc<OnceLock<Arc<Agent>>> = Arc::new(OnceLock::new());
    // Shared OnceLock so FilesSkill can refresh session prompts after context writes.
    let sessions_cell: Arc<OnceLock<Arc<SessionManager>>> = Arc::new(OnceLock::new());

    // Build skills
    let mut skill_registry = SkillRegistry::new();
    skill_registry.register(Box::new(skills::shell::ShellSkill::new()));
    skill_registry.register(Box::new(skills::web::WebSkill::new(http_client.clone())));
    skill_registry.register(Box::new(skills::memory::MemorySkill::new(
        memory.clone(),
        context_dir.clone(),
    )));

    if let Some((tts_url, tts_voice, tts_model)) = &config.tts {
        let tts_client = Arc::new(tts::TtsClient::new(tts_url, tts_voice, tts_model));
        skill_registry.register(Box::new(skills::tts::TtsSkill::new(tts_client, router.clone())));
    }
    skill_registry.register(Box::new(skills::sessions_send::SendToSessionSkill::new(router.clone(), sessions_cell.clone())));

    skill_registry.register(Box::new(skills::current_time::CurrentTimeSkill::new(pairing_store.clone())));
    skill_registry.register(Box::new(skills::lua::LuaSkill::new()));
    skill_registry.register(Box::new(skills::files::FilesSkill::new(
        vec![
            data_dir.clone(),
            context_dir.clone(),
            config.workspace_dir.clone(),
        ],
        context_dir.clone(),
        sessions_cell.clone(),
    )));

    // Markdown-based knowledge skills: repo bundled → home → workspace (ascending precedence)
    let skill_dirs = vec![
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        config.home_dir.clone(),
        config.workspace_dir.clone(),
    ];
    for skill in skills::markdown::load_markdown_skills(&skill_dirs) {
        skill_registry.register(skill);
    }

    skill_registry.register(Box::new(
        skills::relation::RelationSkill::new(pairing_store.clone())
    ));

    // Reminder skill — agent reference filled in after agent construction.
    let reminder_skill = skills::reminders::ReminderSkill::new(
        reminder_store.clone(),
        agent_cell.clone(),
        router.clone(),
    );
    skill_registry.register(Box::new(reminder_skill));

    // Build tool registry (delegates to skills)
    let tool_registry = ToolRegistry::new(skill_registry);
    let skill_additions = tool_registry.system_prompt_additions();

    let workspace_dir_str = config.workspace_dir.to_string_lossy().to_string();
    let model_name = config
        .agent_providers
        .iter()
        .map(|(p, _)| p.model.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let tool_summaries = tool_registry.tool_summaries().await;

    let soul = identity::build_system_prompt(
        &soul_file_str,
        &context_dir_str,
        &skill_additions,
        &workspace_dir_str,
        &model_name,
        None,
        &tool_summaries,
        &agent_name,
    )?;

    if config.debug {
        info!("[debug] System prompt ({} chars):\n{}", soul.len(), soul);
    }

    let llm = build_llm(&config.agent_providers, &config.agent_fallback);

    let tool_registry = Arc::new(tool_registry);
    let sessions = Arc::new(SessionManager::new(
        soul,
        conv_store,
        soul_file_str,
        context_dir_str,
        skill_additions,
        workspace_dir_str,
        model_name,
        tool_summaries,
        agent_name.clone(),
    ));

    if config.debug {
        info!("Debug mode enabled — model thoughts and tool calls will be logged");
    }

    let agent = Arc::new(Agent::new(llm, tool_registry.clone(), sessions.clone(), memory.clone(), config.debug));

    // Publish agent and sessions to their respective skills now that both exist.
    let _ = agent_cell.set(agent.clone());
    let _ = sessions_cell.set(sessions.clone());

    // Reminder startup: crash-recover stale tasks, then re-schedule pending ones.
    let stale = reminder_store.recover_stale()?;
    if stale > 0 {
        info!(stale, "Recovered stale reminders");
    }
    let pending = reminder_store.load_pending()?;
    info!(count = pending.len(), "Scheduling pending reminders from DB");
    for reminder in pending {
        spawn_reminder_task(
            reminder_store.clone(),
            agent.clone(),
            router.clone(),
            reminder,
        );
    }

    // Build scheduler and load tasks
    let mut sched = scheduler::Scheduler::new().await?;
    sched
        .load_from_file(&scheduler_file.to_string_lossy(), agent.clone(), router.clone())
        .await?;

    sched.start().await?;
    info!("Scheduler started");

    // Self-awareness pass — persistent random interval, crash-recovery aware.
    {
        let timers = timer::open(&db_path)?;
        let awareness = Arc::new(self_awareness::SelfAwareness::new(
            agent.clone(),
            pairing_store.clone(),
        ));
        info!("Starting self-awareness loop (4–6h persistent random interval)");
        let sa_min = config.self_awareness_interval_hours.saturating_sub(1).max(1);
        let sa_max = config.self_awareness_interval_hours + 1;
        let sa_trigger = Arc::new(tokio::sync::Notify::new());
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let trigger = sa_trigger.clone();
            tokio::spawn(async move {
                let mut sigusr1 = signal(SignalKind::user_defined1())
                    .expect("failed to register SIGUSR1 handler");
                loop {
                    sigusr1.recv().await;
                    info!("SIGUSR1 received — triggering self-awareness pass");
                    trigger.notify_one();
                }
            });
        }
        self_awareness::spawn(awareness, sa_min, sa_max, timers, sa_trigger);
    }

    // Gateway map for branch & merge orchestration
    let gateway_map = Arc::new(GatewayMap::new(build_llm(&config.gateway_providers, &config.gateway_fallback)));

    // Track which message is actively being processed per session.
    let active_processing: Arc<RwLock<HashMap<String, ActiveProcessing>>> =
        Arc::new(RwLock::new(HashMap::new()));

    info!("{agent_name} is online — polling for updates...");

    let telegram_bot_token = config.telegram_bot_token.clone();

    // --- Graceful shutdown ---
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let shutdown_signal = shutdown.clone();
    tokio::spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM");
            tokio::select! {
                _ = ctrl_c => info!("Received Ctrl+C, shutting down..."),
                _ = sigterm.recv() => info!("Received SIGTERM, shutting down..."),
            }
        }
        #[cfg(not(unix))]
        {
            ctrl_c.await.ok();
            info!("Received Ctrl+C, shutting down...");
        }
        shutdown_signal.notify_waiters();
    });

    loop {
        let poll_fut = async {
            let guard = router.read().await;
            guard.poll_all().await
        };
        tokio::select! {
            biased;
            _ = shutdown.notified() => {
                info!("Shutdown signal received, stopping polling loop");
                break;
            }
            updates = poll_fut => {
                for incoming in updates {
                    let session_key = incoming.source.session_key();

                    info!(
                        session_key = %session_key,
                        user = %incoming.user_name,
                        "Message: {}",
                        incoming.text
                    );

                    // --- Pairing gate ---
                    let (identity_id, access_level_opt, relation_opt) = match pairing_store
                        .find_identity_full(&incoming.source.channel_id, &incoming.sender_id)
                    {
                        Err(e) => {
                            error!("Pairing DB error: {e}");
                            continue;
                        }
                        Ok(None) => {
                            handle_unpaired_sender(&pairing_store, &router, &incoming).await;
                            continue;
                        }
                        Ok(Some((identity_id, access_level_opt, relation_opt))) => {
                            // Notify on first post-approval message
                            match pairing_store.pop_pending_notification(
                                &incoming.source.channel_id,
                                &incoming.sender_id,
                            ) {
                                Ok(Some((target, msg))) => {
                                    router.read().await.send_text(&target, &msg).await;
                                }
                                Ok(None) => {}
                                Err(e) => error!("Notification check error: {e}"),
                            }
                            (identity_id, access_level_opt, relation_opt)
                        }
                    };

                    // --- Compute display_text with sender attribution prefix ---
                    let access = access_level_opt.as_deref().unwrap_or("member");
                    let prefix = match relation_opt.as_deref() {
                        Some(r) => format!("role={}, relation={}", access, r),
                        None    => format!("role={}", access),
                    };
                    let is_group = incoming.source.chat_id < 0;
                    let reply_part = incoming.reply_to.as_ref().map(|r| {
                        let quoted = truncate_both_ends(&r.text, 80);
                        if is_group {
                            format!(" replying to {}(id:{}) \"{}\"", r.sender_name, r.sender_id, quoted)
                        } else {
                            format!(" replying to {} \"{}\"", r.sender_name, quoted)
                        }
                    }).unwrap_or_default();
                    let display_text = if is_group {
                        format!(
                            "[{prefix}] {name} (id:{sender_id}){reply_part}: {text}",
                            name      = incoming.user_name,
                            sender_id = incoming.sender_id,
                            text      = incoming.text,
                        )
                    } else {
                        format!("[{prefix}]{reply_part}: {text}", text = incoming.text)
                    };

                    // --- Command handling ---
                    if incoming.text.starts_with('/') {
                        let cmd = incoming.text.split_whitespace().next().unwrap_or("");
                        match cmd {
                            "/reset" => {
                                sessions.reset(&session_key).await;
                                router
                                    .read()
                                    .await
                                    .send_text(&incoming.source, "Session reset.")
                                    .await;
                            }
                            "/start" => {
                                router
                                    .read()
                                    .await
                                    .send_text(
                                        &incoming.source,
                                        &format!("Hey, I'm {agent_name} — your AI assistant. Just talk to me."),
                                    )
                                    .await;
                            }
                            "/help" => {
                                router
                                    .read()
                                    .await
                                    .send_text(
                                        &incoming.source,
                                        "I can:\n\
                                         • Chat and answer questions\n\
                                         • Search the web\n\
                                         • Read and write files\n\
                                         • Run shell commands\n\
                                         • Set reminders\n\
                                         • Send voice notes\n\
                                         • Remember things about you\n\n\
                                         Commands:\n\
                                         /reset — clear conversation history\n\
                                         /help — this message",
                                    )
                                    .await;
                            }
                            _ => {
                                router
                                    .read()
                                    .await
                                    .send_text(&incoming.source, "Unknown command. Try /help.")
                                    .await;
                            }
                        }
                        continue;
                    }

                    // --- Branch & Merge: check if there's an active branch for this session ---
                    let orchestrator = gateway_map.get(&session_key).await;

                    let active = active_processing.read().await;
                    let active_info = active.get(&session_key);

                    if let Some(active_info) = active_info {
                        let recent_messages = sessions.get_messages(&session_key).await;
                        let message_being_processed = active_info.message_text.clone();
                        let branch_id = active_info.branch_id;
                        drop(active);

                        let is_related = orchestrator
                            .classify(
                                &recent_messages,
                                &message_being_processed,
                                &incoming.text,
                            )
                            .await;

                        if is_related {
                            info!(
                                session_key = %session_key,
                                branch_id,
                                "Absorbing related message into active branch"
                            );
                            orchestrator
                                .absorb_message(branch_id, incoming.text, &sessions, &session_key)
                                .await;
                            continue;
                        }

                        info!(
                            session_key = %session_key,
                            "Unrelated message, starting new concurrent branch"
                        );
                    } else {
                        drop(active);
                    }

                    // --- Start a new branch ---
                    let (branch_id, mut absorb_rx) = orchestrator.start_branch().await;

                    active_processing.write().await.insert(
                        session_key.clone(),
                        ActiveProcessing {
                            message_text: incoming.text.clone(),
                            branch_id,
                        },
                    );

                    let call_ctx = CallContext {
                        target: incoming.source.clone(),
                        identity_id: Some(identity_id),
                    };

                    let agent = agent.clone();
                    let router = router.clone();
                    let source = incoming.source.clone();
                    let text = incoming.text;
                    let display_text = display_text;
                    let orchestrator = orchestrator.clone();
                    let active_processing = active_processing.clone();
                    let tool_registry = tool_registry.clone();
                    let token = telegram_bot_token.clone();

                    tokio::spawn(async move {
                        let session_key = source.session_key();

                        let typing_handle = if source.channel_id == "telegram" {
                            let chat_id = source.chat_id;
                            let token = token.clone();
                            Some(tokio::spawn(async move {
                                let bot = channel::telegram::TelegramBot::new(&token);
                                loop {
                                    bot.send_typing(chat_id).await;
                                    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
                                }
                            }))
                        } else {
                            None
                        };

                        let router_cb = router.clone();
                        let source_cb = source.clone();
                        let orchestrator_cb = orchestrator.clone();
                        let user_text = text.clone();
                        let on_interim_text = move |t: String| {
                            let router = router_cb.clone();
                            let source = source_cb.clone();
                            let orchestrator = orchestrator_cb.clone();
                            let user_msg = user_text.clone();
                            async move {
                                if t.trim() == "SILENT" || is_empty_response(&t) {
                                    return;
                                }
                                match orchestrator
                                    .send_or_merge(branch_id, &t, &user_msg)
                                    .await
                                {
                                    Some(merged) => {
                                        router.read().await.send_text(&source, &merged).await;
                                    }
                                    None => {}
                                }
                            }
                        };

                        let timeout = tokio::time::timeout(
                            std::time::Duration::from_secs(300),
                            agent.process_message(&session_key, &text, &display_text, on_interim_text, &mut absorb_rx, &call_ctx),
                        )
                        .await;

                        if let Some(handle) = typing_handle {
                            handle.abort();
                        }

                        let response = match timeout {
                            Ok(Ok(Some(resp))) => Some(resp),
                            Ok(Ok(None)) => None,
                            Ok(Err(e)) => {
                                error!(session_key = %session_key, "Agent error: {e}");
                                Some(format!("Error: {e}"))
                            }
                            Err(_) => Some("Response timed out after 300s.".into()),
                        };

                        if let Some(resp) = response {
                            if !resp.is_empty() && resp.trim() != "SILENT" && !is_empty_response(&resp) {
                                match orchestrator
                                    .send_or_merge(branch_id, &resp, &text)
                                    .await
                                {
                                    Some(merged) => {
                                        router.read().await.send_text(&source, &merged).await;
                                    }
                                    None => {}
                                }
                            }
                        }

                        orchestrator.end_branch(branch_id).await;
                        active_processing.write().await.remove(&session_key);
                    });

                    // suppress unused warning — tool_registry is used in the spawn above via closure capture
                    let _ = &tool_registry;
                }
            }
        }
    }

    info!("{agent_name} shutting down gracefully");
    Ok(())
}

/// Returns true if the response is a vacuous meta-response that shouldn't be sent.
/// Catches things like "(Empty response)", "(No response)", etc. that models sometimes emit.
fn is_empty_response(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    // Parenthesized meta-responses
    if t.starts_with('(') && t.ends_with(')') && t.len() < 40 {
        let inner = &t[1..t.len() - 1];
        return inner.contains("empty") || inner.contains("no response") || inner.contains("nothing");
    }
    false
}

/// Truncate text from both ends, keeping start and end with "…" in the middle.
fn truncate_both_ends(text: &str, max_len: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max_len {
        return text.to_string();
    }
    let side = max_len / 2;
    let start: String = text.chars().take(side).collect();
    let end: String = text.chars().rev().take(side).collect();
    let end: String = end.chars().rev().collect();
    format!("{}…{}", start.trim_end(), end.trim_start())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_both_ends_short_string() {
        assert_eq!(truncate_both_ends("hello", 10), "hello");
    }

    #[test]
    fn truncate_both_ends_exact_length() {
        let s = "a".repeat(10);
        assert_eq!(truncate_both_ends(&s, 10), s);
    }

    #[test]
    fn truncate_both_ends_long_string_has_ellipsis() {
        let result = truncate_both_ends("hello world foo bar baz", 10);
        assert!(result.contains('…'));
    }

    #[test]
    fn truncate_both_ends_preserves_start_and_end() {
        let result = truncate_both_ends("abcdefghijklmnopqrstuvwxyz", 10);
        assert!(result.starts_with("abcde"));
        assert!(result.ends_with("vwxyz"));
    }

    #[test]
    fn truncate_both_ends_trims_surrounding_whitespace() {
        assert_eq!(truncate_both_ends("  hello  ", 10), "hello");
    }

    #[test]
    fn truncate_both_ends_empty() {
        assert_eq!(truncate_both_ends("", 10), "");
    }
}

/// Handle a message from an unpaired sender: issue a pairing code or remind them it's pending.
async fn handle_unpaired_sender(
    pairing_store: &Arc<PairingStore>,
    router: &Arc<RwLock<ChannelRouter>>,
    incoming: &IncomingMessage,
) {
    let channel_id = &incoming.source.channel_id;
    let sender_id = &incoming.sender_id;
    let chat_id = incoming.source.chat_id;
    let display_name = &incoming.user_name;

    match pairing_store.has_pending_request(channel_id, sender_id) {
        Ok(true) => {
            router
                .read()
                .await
                .send_text(
                    &incoming.source,
                    "Your pairing request is still pending approval. Please wait.",
                )
                .await;
        }
        Ok(false) => {
            match pairing_store.create_request(channel_id, sender_id, chat_id, display_name) {
                Ok(code) => {
                    info!(
                        code,
                        sender_id,
                        channel_id,
                        "New pairing request — run: nina pair approve {code}"
                    );
                    router
                        .read()
                        .await
                        .send_text(
                            &incoming.source,
                            &format!(
                                "Hi! This bot requires approval before use.\n\
                                 Your pairing code is: *{code}*\n\n\
                                 Ask the bot owner to run:\n`nina pair approve {code}`"
                            ),
                        )
                        .await;
                }
                Err(e) => {
                    error!("Failed to create pairing request: {e}");
                }
            }
        }
        Err(e) => {
            error!("Failed to check pending request: {e}");
        }
    }
}
