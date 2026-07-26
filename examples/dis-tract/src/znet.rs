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

/// The coordinator publishes here to clear one sequence's KV, so the sequence
/// starts from empty context without disturbing others in flight. The key must
/// carry the sequence id: `RESET_SEQ_WILDCARD` matches exactly one chunk after
/// the prefix, so a bare `distract/reset` reaches no worker.
pub const RESET_SEQ_PREFIX: &str = "distract/reset/";
pub const RESET_SEQ_WILDCARD: &str = "distract/reset/*";

pub fn reset_key(seq_id: u64) -> String {
    format!("{RESET_SEQ_PREFIX}{seq_id}")
}

/// The sequence slot the one-prompt-at-a-time coordinator uses, and the slot a
/// [`crate::protocol::StepMeta`] naming no sequence falls back to.
pub const SINGLE_SEQ: u64 = 0;

/// Sequence id carried by a `RESET_SEQ_WILDCARD` / `FREE_SEQ_WILDCARD` sample.
pub fn seq_of_key(key: &str, prefix: &str) -> Option<u64> {
    key.strip_prefix(prefix)?.parse().ok()
}

/// The coordinator publishes here when a sequence ends, so the worker drops its
/// KV instead of holding it for a sequence that will never step again. A worker
/// also expires caches left idle by a coordinator that died mid-sequence.
pub const FREE_SEQ_PREFIX: &str = "distract/free/";
pub const FREE_SEQ_WILDCARD: &str = "distract/free/*";

pub fn free_key(seq_id: u64) -> String {
    format!("{FREE_SEQ_PREFIX}{seq_id}")
}

/// A worker publishes here once a sequence's KV reset has landed. The coordinator
/// waits for one ack per stage before decoding, so no fixed sleep is needed and the
/// reset is guaranteed to precede the prefill step.
pub fn reset_ack_key(stage: usize) -> String {
    format!("distract/resetack/{stage}")
}
pub const RESET_ACK_WILDCARD: &str = "distract/resetack/*";

/// Reset-ack payload: which stage acked, and for which sequence. Both sides go
/// through this pair so the layout cannot drift between them — a coordinator that
/// ignored the sequence id would count another sequence's ack as its own.
pub fn reset_ack(stage: usize, seq_id: u64) -> Vec<u8> {
    let mut b = vec![stage as u8];
    b.extend_from_slice(&seq_id.to_le_bytes());
    b
}

pub fn parse_reset_ack(payload: &[u8]) -> Option<(usize, u64)> {
    let stage = *payload.first()? as usize;
    let seq = u64::from_le_bytes(payload.get(1..9)?.try_into().ok()?);
    Some((stage, seq))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use zenoh::key_expr::KeyExpr;

    /// What a publisher sends must be what a subscriber's wildcard selects. A `*`
    /// matches exactly one chunk, so a per-sequence subscriber never sees a
    /// sequence-less publication — the two sides drift silently, with no error
    /// anywhere, and the stages simply stop resetting.
    fn matches(wildcard: &str, key: &str) -> bool {
        KeyExpr::new(wildcard).unwrap().intersects(&KeyExpr::new(key).unwrap())
    }

    #[test]
    fn sequence_wildcards_select_the_keys_that_are_published() {
        for seq in [SINGLE_SEQ, 1, u64::MAX] {
            assert!(matches(RESET_SEQ_WILDCARD, &reset_key(seq)), "reset {seq}");
            assert!(matches(FREE_SEQ_WILDCARD, &free_key(seq)), "free {seq}");
            assert_eq!(seq_of_key(&reset_key(seq), RESET_SEQ_PREFIX), Some(seq));
            assert_eq!(seq_of_key(&free_key(seq), FREE_SEQ_PREFIX), Some(seq));
        }
        assert!(!matches(RESET_SEQ_WILDCARD, "distract/reset"));
        assert!(!matches(RESET_SEQ_WILDCARD, &free_key(0)));
    }

    #[test]
    fn stage_wildcards_select_their_own_keys_only() {
        assert!(matches(READY_WILDCARD, &ready_key(3)));
        assert!(matches(RESET_ACK_WILDCARD, &reset_ack_key(3)));
        assert!(!matches(RESET_ACK_WILDCARD, &ready_key(3)));
    }

    #[test]
    fn reset_ack_round_trips() {
        assert_eq!(parse_reset_ack(&reset_ack(2, 7)), Some((2, 7)));
        assert_eq!(parse_reset_ack(&reset_ack(0, u64::MAX)), Some((0, u64::MAX)));
        // The pre-sequence ack was a lone stage byte; it must not read as sequence 0.
        assert_eq!(parse_reset_ack(&[2]), None);
        assert_eq!(parse_reset_ack(&[]), None);
    }
}
