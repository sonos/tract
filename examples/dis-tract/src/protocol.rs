//! Control- and data-channel message types exchanged between the coordinator
//! and workers.

use serde::{Deserialize, Serialize};

/// Pinned dtype + shape of one boundary tensor (dims as strings so symbolic
/// dims survive). Informational (used for logging the boundary contract).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TensorContract {
    pub dt: String,
    pub shape: Vec<String>,
}

/// Role of one stage input/output slot.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Crosses the machine boundary (residual activation, token ids, logits).
    Wire,
    /// Stays resident in the worker and loops step→step (KV cache). Never sent.
    Cache,
}

/// Describes one stage I/O slot: its role, the cache pairing key (for `Cache`
/// slots, shared between the `in_*`/`out_*` pair), and — for cache inputs — the
/// empty (P=0) dtype+shape used to seed it. `-1` in `shape` marks a symbolic dim.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IoSpec {
    pub role: Role,
    pub slot: Option<String>,
    pub dt: String,
    pub shape: Vec<i64>,
}

impl IoSpec {
    pub fn wire() -> Self {
        IoSpec { role: Role::Wire, slot: None, dt: String::new(), shape: vec![] }
    }
}

/// Control message: coordinator → worker, once, over `POST /load`. The NNEF
/// model bytes travel in a separate frame after the JSON header.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoadMeta {
    pub stage_index: usize,
    /// Runtime name: `"cpu"`, `"metal"`, or `"cuda"`.
    pub backend: String,
    /// Next stage's base URL, or `None` for the tail stage.
    pub next_hop: Option<String>,
    /// Per-input-outlet roles, in stage `input_outlets()` order.
    pub inputs: Vec<IoSpec>,
    /// Per-output-outlet roles, in stage `output_outlets()` order.
    pub outputs: Vec<IoSpec>,
    /// Weight bytes of this shard, as the coordinator's plan counted them — so
    /// the dashboard's per-node figure matches the aggregated total exactly.
    pub weights_bytes: u64,
}

/// Assignment the coordinator sends a worker. The worker builds its shard
/// locally (load `model_path`, transform, `partition_stages`), so no serialized
/// sub-model crosses the wire — `cut_layers` + `stage_index` pin which shard,
/// and `model_path` is resolvable on the worker (co-located or shared store).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AssignSpec {
    pub stage_index: usize,
    pub cut_layers: Vec<usize>,
    pub backend: String,
    pub next_hop: Option<String>,
    pub model_path: String,
    pub n_layers: usize,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

/// Data message header sent ahead of the activation tensors each step.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StepMeta {
    pub turn: u64,
    pub phase: Phase,
}

/// What a worker advertises on join, so the coordinator can plan a
/// memory-weighted split and assign it a stage.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeCaps {
    pub node_id: String,
    pub hostname: String,
    /// Backend this worker will run its shard on: "cpu" | "metal" | "cuda".
    pub backend: String,
    pub total_mem: u64,
    pub avail_mem: u64,
    /// How much memory the planner may use for this node's shard (weights + KV).
    pub mem_budget: u64,
    pub cpus: usize,
}

/// Cluster-level run telemetry, published by the coordinator. Throughput here is
/// the **end-to-end** token rate (this is pipeline parallelism — every node
/// processes every token — so it is *not* the sum of per-node `tok_s`). Memory
/// aggregation is done against `node_ids` so dead nodes never inflate the total.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RunStats {
    /// Model filename being served.
    pub model: String,
    pub n_stages: usize,
    pub n_layers: usize,
    /// Sum of the shard weights distributed across all stages, in bytes.
    pub total_weight_bytes: u64,
    /// Prefill latency: prompt submit → first generated token, in ms.
    pub ttft_ms: f64,
    /// Steady-state end-to-end decode rate (tokens/s), averaged over decode steps.
    pub decode_tok_s: f64,
    pub tokens: u64,
    pub prompt_tokens: usize,
    /// Node ids participating in this run (for ghost-free memory aggregation).
    pub node_ids: Vec<String>,
}

/// A generation request to the coordinator's `distract/generate` queryable.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenerateRequest {
    pub prompt: Vec<i64>,
    pub max_tokens: usize,
    /// Correlates the live partial stream to this request. The coordinator publishes
    /// `StreamMsg`s on `stream_key(stream_id)`; `0` means the base key (no streaming
    /// consumer). Isolates concurrent/overlapping requests so their streams don't cross.
    #[serde(default)]
    pub stream_id: u64,
    /// Ids that end the generation (EOS / end of an assistant turn). Only the
    /// tokenizer knows them, so the caller supplies them; empty runs to `max_tokens`.
    #[serde(default)]
    pub stop: Vec<i64>,
}

/// Live partial-generation update published on [`crate::znet::STREAM_KEY`] once per
/// decode step: the full token sequence so far, and whether generation is complete.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StreamMsg {
    pub tokens: Vec<i64>,
    pub done: bool,
}

/// The coordinator's reply: the generated token ids and the end-to-end timing.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenerateReply {
    pub tokens: Vec<i64>,
    pub ttft_ms: f64,
    pub decode_tok_s: f64,
}

/// Periodic per-node telemetry for the dashboard.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeStats {
    pub node_id: String,
    pub hostname: String,
    pub stage: usize,
    pub backend: String,
    pub tokens: u64,
    pub last_step_ms: f64,
    pub tok_s: f64,
    pub host_cpu: f32,
    pub host_mem_used: u64,
    pub host_mem_total: u64,
    /// Physical memory this worker process occupies (macOS: `phys_footprint`, not
    /// the resident-set size, which omits compressed and GPU-driver pages).
    pub mem_footprint: u64,
    /// This node's advertised memory budget (bytes) — carried in the heartbeat
    /// so the dashboard shows it even if it missed the pre-assignment caps.
    pub mem_budget: u64,
    /// Bytes of model weights this node's shard holds.
    pub weights_bytes: u64,
}
