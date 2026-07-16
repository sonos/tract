//! Dis-tract coordinator (zenoh): discovers workers by their advertised caps,
//! plans a memory-weighted layer split, assigns + serves each worker its shard,
//! then runs as a persistent generation server — each `distract/generate` query
//! resets the stages' KV and greedily decodes the prompt over the warm pipeline.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use clap::Parser;
use tract_core::prelude::*;
use tract_distributed::plan::{layer_weight_profile, memory_weighted_cuts};
use tract_distributed::protocol::{
    AssignSpec, GenerateReply, GenerateRequest, NodeCaps, Phase, RunStats, StepMeta, StreamMsg,
};
use tract_distributed::{codec, znet};
use zenoh::Wait;

#[derive(Parser, Debug)]
#[command(about = "Dis-tract LLM coordinator (zenoh, memory-weighted planning)")]
struct Args {
    #[arg(long)]
    model: String,
    /// Number of workers to wait for before planning.
    #[arg(long, default_value = "2")]
    workers: usize,
}

fn argmax(t: &Tensor) -> Result<i64> {
    let f = t.cast_to::<f32>()?;
    let v = f.view();
    let s = v.as_slice::<f32>()?;
    let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
    let mut nans = 0usize;
    for (i, &v) in s.iter().enumerate() {
        if v.is_nan() {
            nans += 1;
        } else if v > best_v {
            (best, best_v) = (i, v);
        }
    }
    if nans > 0 {
        log::warn!("argmax saw {nans}/{} NaN logits", s.len());
    }
    Ok(best as i64)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let session = zenoh::open(znet::coordinator_config()?).await.map_err(znet::zerr)?;

    let (full, _n_regular) = tract_distributed::llm::load_model(&args.model)?;
    let n_layers = full
        .input_outlets()?
        .iter()
        .filter(|o| full.node(o.node).name.contains("cache_key"))
        .count();

    // Discover workers via their advertised caps.
    let caps_sub = session.declare_subscriber(znet::CAPS_WILDCARD).await.map_err(znet::zerr)?;
    let mut caps: HashMap<String, NodeCaps> = HashMap::new();
    log::info!("waiting for {} workers to advertise caps...", args.workers);
    while caps.len() < args.workers {
        if let Ok(s) = caps_sub.recv_async().await
            && let Ok(c) = serde_json::from_slice::<NodeCaps>(&s.payload().to_bytes())
            && caps.insert(c.node_id.clone(), c.clone()).is_none()
        {
            log::info!(
                "discovered {} ({}), {} MiB budget",
                c.node_id,
                c.backend,
                c.mem_budget / (1024 * 1024)
            );
        }
    }

    // Order nodes (cpu first for a stable pipeline), plan a memory-weighted split.
    let mut nodes: Vec<NodeCaps> = caps.into_values().collect();
    nodes.sort_by_key(|c| (c.backend != "cpu", c.node_id.clone()));
    let budgets: Vec<u64> = nodes.iter().map(|c| c.mem_budget).collect();
    let profile = layer_weight_profile(&full, n_layers);
    let cut_layers = memory_weighted_cuts(&profile, &budgets)?;
    let n = cut_layers.len() + 1;
    ensure!(n == nodes.len(), "planned {n} stages for {} nodes", nodes.len());
    let owner = |l: usize| cut_layers.iter().filter(|&&c| c <= l).count();
    let stage_weight = |s: usize| -> u64 {
        (0..n_layers).filter(|&l| owner(l) == s).filter_map(|l| profile.get(l)).sum()
    };

    println!("plan ({n_layers} layers, cuts at {cut_layers:?}):");
    for (i, node) in nodes.iter().enumerate() {
        println!(
            "  stage {i} -> {} on {} : {} MiB weights ({} MiB budget)",
            node.node_id,
            node.backend,
            stage_weight(i) / (1024 * 1024),
            node.mem_budget / (1024 * 1024)
        );
    }

    // Assign each node its shard spec; the worker builds the shard locally
    // (load + transform + partition), so no serialized sub-model crosses the wire.
    let mut queryables = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let next_hop = (i + 1 < n).then(|| znet::in_key(i + 1));
        let spec = AssignSpec {
            stage_index: i,
            cut_layers: cut_layers.clone(),
            backend: node.backend.clone(),
            next_hop,
            model_path: args.model.clone(),
            n_layers,
        };
        let cfg = Arc::new(serde_json::to_vec(&spec)?);
        let key = znet::assign_key(&node.node_id);
        let reply_key = key.clone();
        let q = session
            .declare_queryable(&key)
            .callback(move |query| {
                let _ = query.reply(reply_key.clone(), (*cfg).clone()).wait();
            })
            .await
            .map_err(znet::zerr)?;
        queryables.push(q);
    }

    drop(full); // planning is done; workers hold the shards from here on.

    // Wait for all workers to load.
    let ready_sub = session.declare_subscriber(znet::READY_WILDCARD).await.map_err(znet::zerr)?;
    let mut ready = std::collections::HashSet::new();
    while ready.len() < n {
        if let Ok(s) = ready_sub.recv_async().await
            && let Some(&idx) = s.payload().to_bytes().first()
            && ready.insert(idx as usize)
        {
            log::info!("stage {idx} ready ({}/{n})", ready.len());
        }
    }

    let result_sub = session.declare_subscriber(znet::RESULT_KEY).await.map_err(znet::zerr)?;
    let reset_ack_sub =
        session.declare_subscriber(znet::RESET_ACK_WILDCARD).await.map_err(znet::zerr)?;
    let in0 = znet::in_key(0);
    let ids = |toks: &[i64]| Tensor::from_shape(&[1, toks.len()], toks).unwrap();

    // Static run facts for the dashboard's aggregated section.
    let model_name = std::path::Path::new(&args.model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&args.model)
        .to_string();
    let total_weight_bytes: u64 = profile.iter().sum();
    let node_ids: Vec<String> = nodes.iter().map(|c| c.node_id.clone()).collect();

    // Persistent generation server: greedily decode one prompt at a time on the
    // warm cluster, resetting every stage's KV before each. Never exits.
    let gen_q = session.declare_queryable(znet::GENERATE_KEY).await.map_err(znet::zerr)?;
    log::info!("generation server ready on {} — awaiting prompts", znet::GENERATE_KEY);
    while let Ok(query) = gen_q.recv_async().await {
        let Some(req) = query
            .payload()
            .and_then(|p| serde_json::from_slice::<GenerateRequest>(&p.to_bytes()).ok())
        else {
            continue;
        };

        // Fresh context: clear every worker's KV and wait for each stage to ack,
        // so prefill can't race ahead of a reset (bounded fallback if an ack drops).
        session.put(znet::RESET_KEY, vec![]).await.map_err(znet::zerr)?;
        {
            let mut acked = std::collections::HashSet::new();
            let deadline = tokio::time::sleep(Duration::from_millis(500));
            tokio::pin!(deadline);
            while acked.len() < n {
                tokio::select! {
                    s = reset_ack_sub.recv_async() => match s {
                        Ok(s) => {
                            if let Some(&idx) = s.payload().to_bytes().first() {
                                acked.insert(idx as usize);
                            }
                        }
                        Err(_) => break,
                    },
                    _ = &mut deadline => {
                        log::warn!("reset ack timeout: {}/{n} stages", acked.len());
                        break;
                    }
                }
            }
        }

        let skey = if req.stream_id != 0 {
            znet::stream_key(req.stream_id)
        } else {
            znet::STREAM_KEY.to_string()
        };
        let mut out_toks = Vec::with_capacity(req.max_tokens);
        let mut dist_tok = 0i64;
        let (mut ttft_ms, mut dec_sum_ms, mut dec_cnt) = (0.0f64, 0.0f64, 0u64);
        for turn in 0..req.max_tokens as u64 {
            let (phase, tok) = if turn == 0 {
                (Phase::Prefill, ids(&req.prompt))
            } else {
                (Phase::Decode, ids(&[dist_tok]))
            };
            let t0 = Instant::now();
            let mut body = Vec::new();
            codec::write_frame(&mut body, &serde_json::to_vec(&StepMeta { turn, phase })?)?;
            codec::write_tensors(&mut body, &[tok])?;
            session.put(&in0, body).await.map_err(znet::zerr)?;
            let sample = result_sub.recv_async().await.map_err(znet::zerr)?;
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let mut rc = Cursor::new(sample.payload().to_bytes().to_vec());
            let _ = codec::read_frame(&mut rc)?;
            dist_tok = argmax(&codec::read_tensors(&mut rc)?[0].clone())?;
            let stop_hit = req.stop.contains(&dist_tok);
            if !stop_hit {
                out_toks.push(dist_tok);
            }

            if turn == 0 {
                ttft_ms = elapsed;
            } else {
                dec_sum_ms += elapsed;
                dec_cnt += 1;
            }
            let decode_tok_s = if dec_cnt > 0 { 1000.0 * dec_cnt as f64 / dec_sum_ms } else { 0.0 };
            let run = RunStats {
                model: model_name.clone(),
                n_stages: n,
                n_layers,
                total_weight_bytes,
                ttft_ms,
                decode_tok_s,
                tokens: out_toks.len() as u64,
                prompt_tokens: req.prompt.len(),
                node_ids: node_ids.clone(),
            };
            let _ = session.put(znet::RUN_KEY, serde_json::to_vec(&run)?).await;
            // Live partial for the dashboard's streaming chat, on this request's key.
            let stream = StreamMsg { tokens: out_toks.clone(), done: false };
            let _ = session.put(&skey, serde_json::to_vec(&stream)?).await;

            if stop_hit {
                log::info!("stop token {dist_tok} after {} tokens", out_toks.len());
                break;
            }
        }

        let _ = session
            .put(&skey, serde_json::to_vec(&StreamMsg { tokens: out_toks.clone(), done: true })?)
            .await;
        let decode_tok_s = if dec_cnt > 0 { 1000.0 * dec_cnt as f64 / dec_sum_ms } else { 0.0 };
        log::info!(
            "generated {} tokens (ttft {:.0}ms, {:.1} tok/s)",
            out_toks.len(),
            ttft_ms,
            decode_tok_s
        );
        let reply = GenerateReply { tokens: out_toks, ttft_ms, decode_tok_s };
        let _ = query.reply(znet::GENERATE_KEY, serde_json::to_vec(&reply)?).await;
    }
    Ok(())
}
