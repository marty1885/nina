use crate::agent::Agent;
use crate::channel::{CallContext, ChannelTarget};
use crate::pairing::PairingStore;
use crate::timer::{self, TimerStore};
use std::sync::Arc;
use tokio::sync::{mpsc, Notify};
use tracing::{info, warn};

pub struct SelfAwareness {
    agent: Arc<Agent>,
    pairing: Arc<PairingStore>,
}

impl SelfAwareness {
    pub fn new(agent: Arc<Agent>, pairing: Arc<PairingStore>) -> Self {
        Self { agent, pairing }
    }

    pub async fn run(&self) {
        let identities = match self.pairing.get_all_identities_with_channels() {
            Ok(ids) => ids,
            Err(e) => {
                warn!("Self-awareness: failed to load identities: {e}");
                return;
            }
        };

        info!(
            identity_count = identities.len(),
            "Self-awareness pass starting"
        );

        for (identity_id, channels) in identities {
            self.run_for_identity(identity_id, channels).await;
        }
    }

    async fn run_for_identity(&self, identity_id: i64, channels: Vec<(String, i64)>) {
        let channel_list = channels
            .iter()
            .map(|(ch, chat_id)| format!("- {ch}:{chat_id}"))
            .collect::<Vec<_>>()
            .join("\n");

        info!(
            identity_id,
            channels = %channel_list,
            "Self-awareness: running for identity"
        );

        let framing = format!(
            "[Self-awareness pass — ephemeral session, not bound to any channel]\n\
             \n\
             You have no conversation history. Call recall() to get context about this person.\n\
             \n\
             Their known channels (use with send_to_session):\n\
             {channel_list}\n\
             \n\
             After recalling: judge whether there's something genuinely worth saying to \
             this person right now — not by a checklist, but by whether it actually serves them. \
             If yes, use send_to_session(). If uncertain, don't.\n\
             \n\
             Default: SILENT. One message that adds no value costs more than ten good ones earn."
        );

        let session_key = format!("self:{identity_id}");
        let dummy_target = ChannelTarget {
            channel_id: "self".to_string(),
            chat_id: identity_id,
        };
        let ctx = CallContext {
            target: dummy_target,
            identity_id: Some(identity_id),
        };

        let (_absorb_tx, mut absorb_rx) = mpsc::channel::<String>(1);

        // TEMP DEBUG
        info!(identity_id, framing = %framing, "Self-awareness: LLM prompt");

        match self
            .agent
            .process_message(&session_key, &framing, &framing, |_| async {}, &mut absorb_rx, &ctx)
            .await
        {
            Ok(Some(response)) => {
                let silent = response.trim().to_ascii_uppercase().contains("SILENT")
                    || response.trim().is_empty();
                // TEMP DEBUG
                info!(identity_id, response = %response, "Self-awareness: LLM response");
                info!(
                    identity_id,
                    silent,
                    "Self-awareness: pass complete"
                );
            }
            Ok(None) => {
                info!(identity_id, "Self-awareness: pass suppressed (no response)");
            }
            Err(e) => {
                warn!(identity_id, "Self-awareness: pass error: {e}");
            }
        }

        self.agent.reset_session(&session_key).await;
        info!(identity_id, "Self-awareness: ephemeral session reset");
    }
}

/// Spawn a background task that runs the self-awareness pass on a random interval
/// between min_hours and max_hours. The next deadline is persisted so restarts resume
/// the saved countdown rather than resetting it, and crashes mid-run are self-healed.
///
/// `trigger` can be notified (e.g. via SIGUSR1) to fire an immediate out-of-band pass
/// without affecting the timer schedule.
pub fn spawn(awareness: Arc<SelfAwareness>, min_hours: u64, max_hours: u64, timers: Arc<TimerStore>, trigger: Arc<Notify>) {
    let awareness_timer = awareness.clone();
    timer::spawn(
        "self_awareness",
        timers,
        min_hours * 3600,
        max_hours * 3600,
        move || {
            let awareness = awareness_timer.clone();
            async move { awareness.run().await }
        },
    );

    // Out-of-band trigger (e.g. SIGUSR1) — runs immediately, no timer interaction.
    tokio::spawn(async move {
        loop {
            trigger.notified().await;
            info!("Self-awareness: out-of-band trigger received, running pass now");
            awareness.run().await;
        }
    });
}
