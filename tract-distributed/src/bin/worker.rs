//! Dis-tract worker: joins by zenoh scouting, advertises its capabilities, is
//! assigned a stage by the coordinator's memory-weighted plan, then serves
//! activations and publishes a periodic stats heartbeat for the dashboard.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use tract_distributed::caps::HostSampler;
use tract_distributed::llm::{StageSpec, full_io_roles, load_model};
use tract_distributed::partition::const_bytes;
use tract_distributed::protocol::AssignSpec;
use tract_distributed::shard_graph::{load_shard, shard_io_roles, shard_range};
use tract_distributed::stage::{load_stage, reset, step};
use tract_distributed::{codec, znet};

#[derive(Parser, Debug)]
#[command(about = "Dis-tract pipeline-stage worker (zenoh)")]
struct Args {
    /// Backend this node runs its shard on: cpu | metal | cuda.
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// Fraction of available memory the planner may fill for this node's shard.
    #[arg(long, default_value = "0.8")]
    mem_frac: f64,
    /// Absolute memory budget in MiB. Overrides `--mem-frac` when set, so a node
    /// can advertise a fixed capacity (e.g. `--mem-mb 4096` for a 4 GiB node).
    #[arg(long)]
    mem_mb: Option<u64>,
    /// Stable node name, reused across restarts so the dashboard reclaims this
    /// node's card. Give co-located workers distinct names; omit for a persisted
    /// per-host id.
    #[arg(long)]
    name: Option<String>,
}

#[derive(Default)]
struct RunStats {
    tokens: u64,
    last_step_ms: f64,
    tok_s: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let mut sampler = HostSampler::new();
    let node_id = HostSampler::node_id(args.name.as_deref());
    let mut caps = sampler.caps(node_id.clone(), args.backend.clone(), args.mem_frac);
    if let Some(mb) = args.mem_mb {
        caps.mem_budget = mb * 1024 * 1024;
    }

    let session = zenoh::open(znet::worker_config()?).await.map_err(znet::zerr)?;
    // Liveliness token: zenoh undeclares it when this process dies, so the
    // dashboard evicts our card immediately. Held for the process lifetime.
    let _liveness =
        session.liveliness().declare_token(znet::live_key(&node_id)).await.map_err(znet::zerr)?;
    log::info!(
        "node {node_id}: joined via scouting — backend {}, {} MiB budget",
        caps.backend,
        caps.mem_budget / (1024 * 1024)
    );

    // Advertise caps until the coordinator plans and serves our assignment.
    let caps_bytes = serde_json::to_vec(&caps)?;
    let assign_key = znet::assign_key(&node_id);
    let cfg_bytes = loop {
        session.put(znet::caps_key(&node_id), caps_bytes.clone()).await.map_err(znet::zerr)?;
        if let Ok(replies) = session.get(&assign_key).await
            && let Ok(reply) = replies.recv_async().await
            && let Ok(sample) = reply.result()
        {
            break sample.payload().to_bytes().to_vec();
        }
        log::info!("node {node_id}: advertising caps, awaiting assignment...");
        tokio::time::sleep(Duration::from_millis(700)).await;
    };

    let spec: AssignSpec = serde_json::from_slice(&cfg_bytes)?;
    let next_key = spec.next_hop.clone().unwrap_or_else(|| znet::RESULT_KEY.to_string());
    let stage = spec.stage_index;

    // Build this shard locally by pruning the graph to its layer range and reading
    // only its .dat weights — the full model is NEVER materialised (EXO-style), so a
    // model too big for one node still fits per-shard. The whole-model path (no cuts)
    // stays for the single-node reference.
    let st = {
        if spec.cut_layers.is_empty() {
            let (full, n_regular) = load_model(&spec.model_path)?;
            let (inputs, outputs) = full_io_roles(&full, n_regular)?;
            StageSpec { model: full, inputs, outputs }
        } else {
            let (start, end) = shard_range(&spec.cut_layers, stage, spec.n_layers);
            let (model, _io) = load_shard(&spec.model_path, start, end, spec.n_layers)?;
            // Derive I/O roles from an optimized clone (a middle shard drops input_ids
            // there), but hand the RAW model to the backend's prepare — matching the
            // partition_stages flow, so Metal doesn't choke uploading block-quant consts.
            let (inputs, outputs) = shard_io_roles(&model.clone().into_optimized()?)?;
            StageSpec { model, inputs, outputs }
        }
    };
    let weights_bytes = const_bytes(&st.model) as u64;
    let tx = load_stage(st.model, spec.backend.clone(), st.inputs, st.outputs).await?;
    let sub = session.declare_subscriber(znet::in_key(stage)).await.map_err(znet::zerr)?;
    session.put(znet::ready_key(stage), vec![stage as u8]).await.map_err(znet::zerr)?;
    log::info!("node {node_id}: assigned stage {stage} on {} -> {next_key}", spec.backend);

    // Stats heartbeat — always carries budget + shard weight so the dashboard
    // shows them even if it missed this node's pre-assignment caps.
    let stats = Arc::new(Mutex::new(RunStats::default()));
    {
        let (hb_stats, hb_session, hb_id, hb_backend) =
            (stats.clone(), session.clone(), node_id.clone(), spec.backend.clone());
        let mem_budget = caps.mem_budget;
        tokio::spawn(async move {
            let mut sampler = HostSampler::new();
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let (tokens, last_ms, tok_s) = {
                    let s = hb_stats.lock().unwrap();
                    (s.tokens, s.last_step_ms, s.tok_s)
                };
                let ns = sampler.stats(
                    hb_id.clone(),
                    stage,
                    hb_backend.clone(),
                    tokens,
                    last_ms,
                    tok_s,
                    mem_budget,
                    weights_bytes,
                );
                if let Ok(bytes) = serde_json::to_vec(&ns) {
                    let _ = hb_session.put(znet::stats_key(&hb_id), bytes).await;
                }
            }
        });
    }

    // KV reset: clear this stage's cache whenever the coordinator starts a prompt.
    {
        let reset_tx = tx.clone();
        let ack_session = session.clone();
        let ack_key = znet::reset_ack_key(stage);
        let reset_sub = session.declare_subscriber(znet::RESET_KEY).await.map_err(znet::zerr)?;
        tokio::spawn(async move {
            while reset_sub.recv_async().await.is_ok() {
                let _ = reset(&reset_tx).await;
                let _ = ack_session.put(&ack_key, vec![stage as u8]).await;
            }
        });
    }

    while let Ok(sample) = sub.recv_async().await {
        let t0 = Instant::now();
        let bytes = sample.payload().to_bytes().to_vec();
        let mut r = std::io::Cursor::new(bytes.as_slice());
        let step_meta = codec::read_frame(&mut r)?;
        let inputs = codec::read_tensors(&mut r)?;

        let outputs = step(&tx, inputs).await?;

        let mut body = Vec::new();
        codec::write_frame(&mut body, &step_meta)?;
        codec::write_tensors(&mut body, &outputs)?;
        session.put(&next_key, body).await.map_err(znet::zerr)?;

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut s = stats.lock().unwrap();
        s.tokens += 1;
        s.last_step_ms = ms;
        s.tok_s = if ms > 0.0 { 1000.0 / ms } else { 0.0 };
    }
    Ok(())
}
