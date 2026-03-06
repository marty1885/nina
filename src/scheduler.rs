use crate::agent::Agent;
use crate::channel::{CallContext, ChannelRouter, ChannelTarget};
use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info};

#[derive(Debug, Deserialize)]
pub struct ScheduledTask {
    pub name: String,
    pub cron: String,
    pub message: String,
    pub channel: String,
    pub chat_id: i64,
}

#[derive(Debug, Deserialize)]
struct SchedulerConfig {
    #[serde(default)]
    tasks: Vec<ScheduledTask>,
}

pub struct Scheduler {
    sched: JobScheduler,
}

impl Scheduler {
    pub async fn new() -> Result<Self> {
        let sched = JobScheduler::new().await?;
        Ok(Self { sched })
    }

    pub async fn add_task(
        &mut self,
        task: ScheduledTask,
        agent: Arc<Agent>,
        router: Arc<RwLock<ChannelRouter>>,
    ) -> Result<()> {
        let target = ChannelTarget {
            channel_id: task.channel.clone(),
            chat_id: task.chat_id,
        };
        let message = task.message.clone();
        let name = task.name.clone();

        let job = Job::new_async(task.cron.as_str(), move |_uuid, _lock| {
            let agent = agent.clone();
            let router = router.clone();
            let target = target.clone();
            let message = message.clone();
            let name = name.clone();
            Box::pin(async move {
                info!(task = %name, "Scheduled task firing");
                let session_key = target.session_key();
                let call_ctx = CallContext { target: target.clone(), identity_id: None };
                // Scheduled tasks don't participate in branch/merge — use a dummy absorb channel.
                let (_absorb_tx, mut absorb_rx) = tokio::sync::mpsc::channel::<String>(1);
                match agent.process_message(&session_key, &message, &message, |_| async {}, &mut absorb_rx, &call_ctx).await {
                    Ok(Some(response)) => {
                        if !response.is_empty() {
                            router.read().await.send_text(&target, &response).await;
                        }
                    }
                    Ok(None) => {} // Suppressed
                    Err(e) => error!(task = %name, "Scheduled task error: {e}"),
                }
            })
        })?;

        self.sched.add(job).await?;
        info!(task = %task.name, cron = %task.cron, "Scheduled task registered");
        Ok(())
    }

    pub async fn start(&self) -> Result<()> {
        self.sched.start().await?;
        Ok(())
    }

    /// Load tasks from scheduler.toml if it exists.
    pub async fn load_from_file(
        &mut self,
        path: &str,
        agent: Arc<Agent>,
        router: Arc<RwLock<ChannelRouter>>,
    ) -> Result<()> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                info!("No scheduler.toml found, skipping scheduled tasks");
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        let config: SchedulerConfig = toml::from_str(&content)?;
        info!(count = config.tasks.len(), "Loading scheduled tasks");

        for task in config.tasks {
            self.add_task(task, agent.clone(), router.clone()).await?;
        }

        Ok(())
    }
}
