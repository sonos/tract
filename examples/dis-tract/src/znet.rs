//! Zenoh transport glue: key expressions and error mapping.
//!
//! Detection is zenoh **scouting** (peers auto-discover, no addresses); the
//! activation hot path is pub/sub; one-time model+config distribution is a
//! queryable the coordinator serves and each worker pulls.

use anyhow::{Result, anyhow};

/// Map any zenoh error (its error type isn't `std::error::Error`) into `anyhow`.
pub fn zerr<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("zenoh: {e}")
}

/// Star topology: the coordinator runs as a zenoh **router**, workers as
/// **clients** that route all pub/sub/queries through it. This avoids
/// worker-to-worker meshing (which, on a multi-homed host, floods connection
/// attempts across every interface address). On a real LAN, clients discover the
/// router by scouting; on loopback we hand them the bootstrap endpoint.
const BOOTSTRAP: &str = r#"["tcp/127.0.0.1:7447"]"#;

fn base() -> zenoh::Config {
    zenoh::Config::default()
}

pub fn coordinator_config() -> Result<zenoh::Config> {
    let mut c = base();
    let set = |c: &mut zenoh::Config, k: &str, v: &str| {
        c.insert_json5(k, v).map_err(|e| anyhow!("zenoh config {k}: {e}"))
    };
    set(&mut c, "mode", r#""router""#)?;
    set(&mut c, "listen/endpoints", BOOTSTRAP)?;
    Ok(c)
}

pub fn worker_config() -> Result<zenoh::Config> {
    let mut c = base();
    let set = |c: &mut zenoh::Config, k: &str, v: &str| {
        c.insert_json5(k, v).map_err(|e| anyhow!("zenoh config {k}: {e}"))
    };
    set(&mut c, "mode", r#""client""#)?;
    set(&mut c, "connect/endpoints", BOOTSTRAP)?;
    Ok(c)
}

/// A worker publishes its `NodeCaps` here on join; coordinator + dashboard sub.
pub fn caps_key(node_id: &str) -> String {
    format!("distract/node/{node_id}/caps")
}
pub const CAPS_WILDCARD: &str = "distract/node/*/caps";

/// A worker publishes periodic `NodeStats` here; the dashboard subscribes.
pub fn stats_key(node_id: &str) -> String {
    format!("distract/node/{node_id}/stats")
}

/// Everything under a node (caps + stats) — the dashboard's subscription.
pub const NODE_WILDCARD: &str = "distract/node/**";

/// Cluster-level `RunStats` the coordinator publishes each token; dashboard subs.
pub const RUN_KEY: &str = "distract/run";

/// Liveliness token a worker declares on join. Zenoh undeclares it (a `Delete`
/// sample to subscribers) the instant the worker's session drops — clean exit or
/// crash — so the dashboard can evict a dead node immediately, no polling.
pub fn live_key(node_id: &str) -> String {
    format!("distract/live/{node_id}")
}
pub const LIVE_WILDCARD: &str = "distract/live/*";

/// Per-node assignment queryable the coordinator serves after planning: a JSON
/// [`crate::protocol::AssignSpec`]. The worker pulls it by its own id and builds
/// its shard locally, so no model bytes cross the wire.
pub fn assign_key(node_id: &str) -> String {
    format!("distract/assign/{node_id}")
}

/// Where a stage receives its input activations (`frame(StepMeta) + tensors`).
pub fn in_key(stage: usize) -> String {
    format!("distract/stage/{stage}/in")
}

/// A worker publishes here once loaded and subscribed (payload = stage index).
pub fn ready_key(stage: usize) -> String {
    format!("distract/stage/{stage}/ready")
}

pub const READY_WILDCARD: &str = "distract/stage/*/ready";

/// Tail stage publishes final logits here; the coordinator subscribes.
pub const RESULT_KEY: &str = "distract/result";

/// The coordinator publishes here before each prompt; every worker resets its
/// resident KV cache so the new sequence starts from empty context.
pub const RESET_KEY: &str = "distract/reset";

/// A worker publishes here (payload = stage index) once its KV reset has landed.
/// The coordinator waits for one ack per stage before decoding, so no fixed sleep
/// is needed and the reset is guaranteed to precede the prefill step.
pub fn reset_ack_key(stage: usize) -> String {
    format!("distract/resetack/{stage}")
}
pub const RESET_ACK_WILDCARD: &str = "distract/resetack/*";

/// Queryable the coordinator serves as a persistent generation server: the
/// payload is a JSON `GenerateRequest`, the reply a JSON `GenerateReply`.
pub const GENERATE_KEY: &str = "distract/generate";

/// Base of the partial-token stream the coordinator publishes per decode step, so
/// the dashboard can render a generation live. See [`crate::protocol::StreamMsg`].
pub const STREAM_KEY: &str = "distract/stream";

/// Per-request stream key: isolates one generation's live partials from another's.
pub fn stream_key(id: u64) -> String {
    format!("{STREAM_KEY}/{id}")
}
