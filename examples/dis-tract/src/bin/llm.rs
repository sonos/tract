//! Dis-tract coordinator (zenoh): discovers workers by their advertised caps,
//! plans a memory-weighted layer split, assigns + serves each worker its shard,
//! then runs as a persistent generation server.
//!
//! Each `distract/generate` query becomes a [`Sequence`] with its own KV slot on every
//! stage. Up to one sequence per stage is admitted and their steps are interleaved, so a
//! stage that has handed its step on works on another sequence instead of idling until
//! the token comes back. Results carry their sequence id, so they are matched rather
//! than assumed to arrive in order.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use clap::Parser;
use tract_core::prelude::*;
use tract_distributed::plan::{memory_weighted_cuts, stage_weights};
use tract_distributed::protocol::{
    AssignSpec, GenerateReply, GenerateRequest, NodeCaps, RunStats, StepMeta, StreamMsg,
};
use tract_distributed::schedule::{Progress, Scheduler, Sequence, Step};
use tract_distributed::shard_graph::{
    ast_layer_weight_profile, count_layers, read_graph_and_sizes,
};
use tract_distributed::{codec, znet};
use zenoh::Wait;
use zenoh::query::Query;
use zenoh::sample::SampleKind;

/// How long one stage-chain traversal may take before the generation is abandoned.
/// A step is tens of ms and a long prefill a few seconds, so this only fires when a
/// stage is gone or wedged — the case that used to block the coordinator forever.
const STEP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Parser, Debug)]
#[command(about = "Dis-tract LLM coordinator (zenoh, memory-weighted planning)")]
struct Args {
    #[arg(long)]
    model: String,
    /// Number of workers to wait for before planning.
    #[arg(long, default_value = "2")]
    workers: usize,
    /// Sequences allowed to hold a KV slot at once. Defaults to the number of
    /// stages, which is what it takes to keep every stage busy. Higher lets more
    /// clients progress together at the cost of KV on every worker.
    #[arg(long)]
    max_sequences: Option<usize>,
    /// Prompt tokens per prefill step; 0 feeds the whole prompt in one step. Chunking
    /// stops a long prompt from holding a stage for its entire prefill.
    #[arg(long, default_value = "0")]
    prefill_chunk: usize,
}

/// Close a sequence out: final stream message, release its KV on every stage, reply.
async fn finish(session: &zenoh::Session, s: Sequence<Query>, error: Option<String>) -> Result<()> {
    let done = StreamMsg { tokens: s.out_toks.clone(), done: true };
    let _ = session.put(&s.stream_key, serde_json::to_vec(&done)?).await;
    let _ = session.put(znet::free_key(s.id), vec![]).await;
    let reply = GenerateReply {
        decode_tok_s: s.decode_tok_s(),
        tokens: s.out_toks,
        ttft_ms: s.ttft_ms,
        error,
    };
    let _ = s.reply.reply(znet::GENERATE_KEY, serde_json::to_vec(&reply)?).await;
    Ok(())
}

/// Give up on every sequence riding a pipeline that has stopped answering, replying to
/// each with what it managed to generate. Their stages' KV is out of step with them, so
/// each is freed; the next sequence resets its own slot anyway.
async fn abandon(
    session: &zenoh::Session,
    sched: &mut Scheduler<Query>,
    why: String,
) -> Result<()> {
    let lost = sched.drain();
    log::error!("pipeline stalled, abandoning {} sequence(s): {why}", lost.len());
    for s in lost {
        log::warn!("seq {} abandoned after {} tokens", s.id, s.out_toks.len());
        finish(session, s, Some(why.clone())).await?;
    }
    Ok(())
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

    let (graph_text, dat_sizes) = read_graph_and_sizes(&args.model)?;
    let doc = tract_nnef::ast::parse::parse_document(&graph_text)?;
    let n_layers = count_layers(&doc);

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
    let profile = ast_layer_weight_profile(&doc, &dat_sizes, n_layers)?;
    let cut_layers = memory_weighted_cuts(&profile, &budgets)?;
    let n = cut_layers.len() + 1;
    ensure!(n == nodes.len(), "planned {n} stages for {} nodes", nodes.len());
    let weights = stage_weights(&profile, &cut_layers);

    println!("plan ({n_layers} layers, cuts at {cut_layers:?}):");
    for (i, node) in nodes.iter().enumerate() {
        println!(
            "  stage {i} -> {} on {} : {} MiB weights ({} MiB budget)",
            node.node_id,
            node.backend,
            weights[i] / (1024 * 1024),
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
    // A stage that dies mid-token takes its `distract/result` publication with it, so
    // the wait below must end on something other than a reply: zenoh retracts the dead
    // worker's liveliness token, and the deadline covers a stage that is merely stuck.
    let live_sub =
        session.liveliness().declare_subscriber(znet::LIVE_WILDCARD).await.map_err(znet::zerr)?;
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

    // Persistent generation server. Never exits.
    let gen_q = session.declare_queryable(znet::GENERATE_KEY).await.map_err(znet::zerr)?;
    log::info!("generation server ready on {} — up to {n} sequences in flight", znet::GENERATE_KEY);

    let mut sched: Scheduler<Query> =
        Scheduler::new(n, args.max_sequences.unwrap_or(n), args.prefill_chunk);
    // When each step went out, so a result can be timed. Kept here rather than in the
    // scheduler, which stays a pure function of the results it is told about.
    let mut sent: HashMap<u64, Instant> = HashMap::new();
    // Decode steps the cluster has completed across all sequences in the current busy
    // period, and the rate that gives. This is the number interleaving moves: a single
    // sequence decodes no faster, the cluster gets through more of them at once.
    let mut busy: Option<(Instant, u64)> = None;
    let mut cluster_tok_s;
    loop {
        for Step { seq_id, meta, ids: step_ids } in sched.dispatch() {
            let mut body = Vec::new();
            codec::write_frame(&mut body, &serde_json::to_vec(&meta)?)?;
            codec::write_tensors(&mut body, &[ids(&step_ids)])?;
            sent.insert(seq_id, Instant::now());
            session.put(&in0, body).await.map_err(znet::zerr)?;
        }
        if sched.active() == 0 {
            busy = None;
        }

        let waiting = sched.waiting();
        tokio::select! {
            // Results first: draining the pipeline matters more than admitting work.
            biased;

            r = result_sub.recv_async() => {
                let Ok(sample) = r else { continue };
                let mut rc = Cursor::new(sample.payload().to_bytes().to_vec());
                let meta: StepMeta = match codec::read_frame(&mut rc)
                    .map_err(|e| e.to_string())
                    .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
                {
                    Ok(m) => m,
                    Err(e) => {
                        log::error!("undecodable result meta, dropping: {e}");
                        continue;
                    }
                };
                // Results carry the sequence they belong to, so they need not come back
                // in the order the steps went out.
                let Some(seq_id) = meta.seq_ids.first().copied() else {
                    log::warn!("result names no sequence, dropping");
                    continue;
                };
                let tok = argmax(&codec::read_tensors(&mut rc)?[0])?;
                let elapsed = sent
                    .remove(&seq_id)
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);

                match sched.on_result(seq_id, tok, elapsed) {
                    None => log::warn!("result for a sequence no longer active: {seq_id}"),
                    Some(Progress::Prefilling) => {}
                    Some(progress) => {
                        let (since, steps) = busy.get_or_insert((Instant::now(), 0));
                        *steps += 1;
                        let secs = since.elapsed().as_secs_f64();
                        cluster_tok_s = if secs > 0.0 { *steps as f64 / secs } else { 0.0 };

                        let (toks, ttft, prompt_tokens, skey) = match &progress {
                            Progress::Done(s) => (
                                s.out_toks.clone(),
                                s.ttft_ms,
                                s.prompt_tokens,
                                s.stream_key.clone(),
                            ),
                            _ => {
                                let s = sched.sequence(seq_id).expect("stepped sequence");
                                (
                                    s.out_toks.clone(),
                                    s.ttft_ms,
                                    s.prompt_tokens,
                                    s.stream_key.clone(),
                                )
                            }
                        };
                        let partial = StreamMsg { tokens: toks.clone(), done: false };
                        let _ = session.put(&skey, serde_json::to_vec(&partial)?).await;
                        let run = RunStats {
                            model: model_name.clone(),
                            n_stages: n,
                            n_layers,
                            total_weight_bytes,
                            ttft_ms: ttft,
                            decode_tok_s: cluster_tok_s,
                            tokens: toks.len() as u64,
                            prompt_tokens,
                            node_ids: node_ids.clone(),
                        };
                        let _ = session.put(znet::RUN_KEY, serde_json::to_vec(&run)?).await;

                        if let Progress::Done(s) = progress {
                            log::info!(
                                "seq {} done: {} tokens (ttft {:.0}ms, {:.1} tok/s), \
                                 {} still in flight",
                                s.id,
                                s.out_toks.len(),
                                s.ttft_ms,
                                s.decode_tok_s(),
                                sched.active()
                            );
                            finish(&session, *s, None).await?;
                        }
                    }
                }
            }

            l = live_sub.recv_async() => {
                // Liveliness replays the live tokens as Puts on declare; only a Delete
                // for a stage of *this* run ends the wait.
                if let Ok(sample) = l
                    && sample.kind() == SampleKind::Delete
                    && let Some(id) = sample.key_expr().as_str().rsplit('/').next()
                    && node_ids.iter().any(|nid| nid == id)
                {
                    abandon(&session, &mut sched, format!("node {id} left the cluster")).await?;
                }
            }

            q = gen_q.recv_async(), if sched.has_room() => {
                let Ok(query) = q else { continue };
                let Some(req) = query
                    .payload()
                    .and_then(|p| serde_json::from_slice::<GenerateRequest>(&p.to_bytes()).ok())
                else {
                    log::warn!("undecodable generate request");
                    continue;
                };
                let skey = if req.stream_id != 0 {
                    znet::stream_key(req.stream_id)
                } else {
                    znet::STREAM_KEY.to_string()
                };
                let prompt_tokens = req.prompt.len();
                let id = sched.admit(req, skey, query);

                // Fresh context: clear this sequence's KV on every stage and wait for
                // each to ack, so its prefill cannot race ahead of its own reset. Other
                // sequences keep their caches, and their results queue up meanwhile.
                session.put(znet::reset_key(id), vec![]).await.map_err(znet::zerr)?;
                let mut acked = std::collections::HashSet::new();
                let deadline = tokio::time::sleep(Duration::from_millis(500));
                tokio::pin!(deadline);
                while acked.len() < n {
                    tokio::select! {
                        a = reset_ack_sub.recv_async() => match a {
                            Ok(a) => {
                                if let Some((stage, seq)) =
                                    znet::parse_reset_ack(&a.payload().to_bytes())
                                    && seq == id
                                {
                                    acked.insert(stage);
                                }
                            }
                            Err(_) => break,
                        },
                        _ = &mut deadline => {
                            log::warn!("seq {id}: reset ack timeout, {}/{n} stages", acked.len());
                            break;
                        }
                    }
                }
                log::info!("seq {id} admitted ({prompt_tokens} prompt tokens)");
            }

            _ = tokio::time::sleep(STEP_TIMEOUT), if waiting => {
                // The stages are shared, so a wedged one takes down every sequence
                // riding the pipeline, not just the one whose step was outstanding.
                sent.clear();
                abandon(
                    &session,
                    &mut sched,
                    format!("no stage reply in {}s", STEP_TIMEOUT.as_secs()),
                )
                .await?;
            }
        }
    }
}
