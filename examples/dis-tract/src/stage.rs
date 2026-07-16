//! Transport-agnostic stage execution: owns the (`!Send`) [`StageState`] and its
//! resident KV on a dedicated thread, driven over a channel so any transport
//! (zenoh, HTTP, in-process) can feed it owned `Tensor`s.

use anyhow::{Result, anyhow};
use tokio::sync::{mpsc, oneshot};
use tract_core::prelude::*;

use crate::llm::StageState;
use crate::protocol::IoSpec;

#[allow(clippy::large_enum_variant)]
pub enum Command {
    Load {
        model: TypedModel,
        backend: String,
        inputs: Vec<IoSpec>,
        outputs: Vec<IoSpec>,
        reply: oneshot::Sender<Result<()>>,
    },
    Step {
        inputs: TVec<Tensor>,
        reply: oneshot::Sender<Result<TVec<Tensor>>>,
    },
    /// Clear the resident KV cache to start a fresh sequence.
    Reset {
        reply: oneshot::Sender<Result<()>>,
    },
}

/// Spawn the stage-owning thread; returns the command channel to it. The
/// `StageState` (and its growing KV cache) lives on this one thread for its
/// whole lifetime — the coordinator's single-machine reference reuses it too.
pub fn spawn_stage_thread() -> mpsc::UnboundedSender<Command> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Command>();
    std::thread::Builder::new()
        .name("distract-stage".into())
        .spawn(move || {
            let mut stage: Option<StageState> = None;
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    Command::Load { model, backend, inputs, outputs, reply } => {
                        let r = StageState::new(model, &backend, inputs, outputs)
                            .map(|s| stage = Some(s));
                        let _ = reply.send(r);
                    }
                    Command::Step { inputs, reply } => {
                        let out = match stage.as_mut() {
                            Some(s) => s.step(inputs),
                            None => Err(anyhow!("stage not loaded")),
                        };
                        let _ = reply.send(out);
                    }
                    Command::Reset { reply } => {
                        let out = match stage.as_mut() {
                            Some(s) => s.reset(),
                            None => Err(anyhow!("stage not loaded")),
                        };
                        let _ = reply.send(out);
                    }
                }
            }
        })
        .expect("spawn stage thread");
    tx
}

/// Load a model into a fresh stage thread and block until it is ready.
pub async fn load_stage(
    model: TypedModel,
    backend: String,
    inputs: Vec<IoSpec>,
    outputs: Vec<IoSpec>,
) -> Result<mpsc::UnboundedSender<Command>> {
    let tx = spawn_stage_thread();
    let (reply, done) = oneshot::channel();
    tx.send(Command::Load { model, backend, inputs, outputs, reply }).ok();
    done.await.map_err(|_| anyhow!("stage thread gone"))??;
    Ok(tx)
}

/// Run one step on a stage thread and await the wire outputs.
pub async fn step(
    tx: &mpsc::UnboundedSender<Command>,
    inputs: TVec<Tensor>,
) -> Result<TVec<Tensor>> {
    let (reply, done) = oneshot::channel();
    tx.send(Command::Step { inputs, reply }).ok();
    done.await.map_err(|_| anyhow!("stage thread gone"))?
}

/// Clear a stage thread's resident KV cache and await completion.
pub async fn reset(tx: &mpsc::UnboundedSender<Command>) -> Result<()> {
    let (reply, done) = oneshot::channel();
    tx.send(Command::Reset { reply }).ok();
    done.await.map_err(|_| anyhow!("stage thread gone"))?
}
