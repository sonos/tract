//! Dis-tract dashboard: a zenoh client that subscribes to every node's caps +
//! stats and the coordinator's run record, serves a live web page (aggregated
//! run stats + per-node grid), and hosts an in-page chat box that drives the
//! coordinator's `distract/generate` server. Liveliness evicts dead nodes
//! immediately.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::extract::{Query, State};
use axum::response::Html;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tract_distributed::chat::ChatTokenizer;
use tract_distributed::protocol::{
    GenerateReply, GenerateRequest, NodeCaps, NodeStats, RunStats, StreamMsg,
};
use tract_distributed::znet;
use zenoh::sample::SampleKind;

/// Grace period before the staleness sweep drops a node that stopped updating
/// without a liveliness `Delete` (e.g. a paused process).
const STALE_AFTER: Duration = Duration::from_secs(6);

#[derive(Default, Clone, Serialize)]
struct NodeView {
    caps: Option<NodeCaps>,
    stats: Option<NodeStats>,
    #[serde(skip)]
    last_seen: Option<Instant>,
}

#[derive(Default)]
struct Dash {
    nodes: HashMap<String, NodeView>,
    run: Option<RunStats>,
}
type Shared = Arc<Mutex<Dash>>;

#[derive(Clone)]
struct AppState {
    dash: Shared,
    session: zenoh::Session,
    tok: Arc<ChatTokenizer>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let session = zenoh::open(znet::worker_config()?).await.map_err(znet::zerr)?;
    let state: Shared = Arc::new(Mutex::new(Dash::default()));
    let tok_path = std::env::var("DISTRACT_TOKENIZER")
        .map_err(|_| anyhow::anyhow!("set DISTRACT_TOKENIZER to the model's tokenizer.json"))?;
    let tok = Arc::new(ChatTokenizer::from_file(&tok_path)?);

    let node_sub = session.declare_subscriber(znet::NODE_WILDCARD).await.map_err(znet::zerr)?;
    {
        let st = state.clone();
        tokio::spawn(async move {
            while let Ok(sample) = node_sub.recv_async().await {
                let key = sample.key_expr().as_str().to_string();
                let bytes = sample.payload().to_bytes();
                let mut d = st.lock().unwrap();
                if key.ends_with("/caps") {
                    if let Ok(c) = serde_json::from_slice::<NodeCaps>(&bytes) {
                        let v = d.nodes.entry(c.node_id.clone()).or_default();
                        v.caps = Some(c);
                        v.last_seen = Some(Instant::now());
                    }
                } else if key.ends_with("/stats")
                    && let Ok(s) = serde_json::from_slice::<NodeStats>(&bytes)
                {
                    let v = d.nodes.entry(s.node_id.clone()).or_default();
                    v.stats = Some(s);
                    v.last_seen = Some(Instant::now());
                }
            }
        });
    }

    let run_sub = session.declare_subscriber(znet::RUN_KEY).await.map_err(znet::zerr)?;
    {
        let st = state.clone();
        tokio::spawn(async move {
            while let Ok(sample) = run_sub.recv_async().await {
                if let Ok(r) = serde_json::from_slice::<RunStats>(&sample.payload().to_bytes()) {
                    st.lock().unwrap().run = Some(r);
                }
            }
        });
    }

    // Liveliness: evict a node's card the moment zenoh reports its token gone.
    let live_sub =
        session.liveliness().declare_subscriber(znet::LIVE_WILDCARD).await.map_err(znet::zerr)?;
    {
        let st = state.clone();
        tokio::spawn(async move {
            while let Ok(sample) = live_sub.recv_async().await {
                if sample.kind() == SampleKind::Delete {
                    let id =
                        sample.key_expr().as_str().rsplit('/').next().unwrap_or("").to_string();
                    st.lock().unwrap().nodes.remove(&id);
                }
            }
        });
    }

    // Staleness sweep: fallback eviction for nodes that went silent.
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                st.lock()
                    .unwrap()
                    .nodes
                    .retain(|_, v| v.last_seen.map(|t| t.elapsed() < STALE_AFTER).unwrap_or(true));
            }
        });
    }

    let app_state = AppState { dash: state, session, tok };
    let app = Router::new()
        .route("/", get(index))
        .route("/api/nodes", get(api_nodes))
        .route("/api/run", get(api_run))
        .route("/api/chat", post(api_chat))
        .route("/api/chat/stream", get(api_chat_stream))
        .with_state(app_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8088").await?;
    println!("dashboard: http://127.0.0.1:8088");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn api_nodes(State(st): State<AppState>) -> Json<Vec<NodeView>> {
    let d = st.dash.lock().unwrap();
    let mut v: Vec<NodeView> = d.nodes.values().cloned().collect();
    v.sort_by_key(|n| n.stats.as_ref().map(|s| s.stage).unwrap_or(usize::MAX));
    Json(v)
}

async fn api_run(State(st): State<AppState>) -> Json<Option<RunStats>> {
    Json(st.dash.lock().unwrap().run.clone())
}

#[derive(Deserialize)]
struct ChatReq {
    text: String,
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct ChatResp {
    reply: String,
    ttft_ms: f64,
    tok_s: f64,
    error: Option<String>,
}

fn chat_err(msg: String) -> Json<ChatResp> {
    Json(ChatResp { reply: String::new(), ttft_ms: 0.0, tok_s: 0.0, error: Some(msg) })
}

/// Tokenize the prompt, run it on the `distract/generate` server over zenoh,
/// then detokenize the reply.
async fn api_chat(State(st): State<AppState>, Json(body): Json<ChatReq>) -> Json<ChatResp> {
    let ids: Vec<i64> = match st.tok.encode_prompt(&body.text) {
        Ok(v) => v,
        Err(e) => return chat_err(format!("tokenize: {e}")),
    };
    if ids.is_empty() {
        return chat_err("empty prompt".into());
    }
    let req = GenerateRequest {
        prompt: ids,
        max_tokens: body.max_tokens.unwrap_or(768),
        stream_id: 0,
        stop: st.tok.stop_ids(),
    };
    let payload = serde_json::to_vec(&req).unwrap();

    let replies = match st
        .session
        .get(znet::GENERATE_KEY)
        .payload(payload)
        .timeout(Duration::from_secs(600))
        .await
    {
        Ok(r) => r,
        Err(e) => return chat_err(format!("cluster query: {e}")),
    };
    let Ok(reply) = replies.recv_async().await else {
        return chat_err("no reply — is a coordinator running?".into());
    };
    let sample = match reply.result() {
        Ok(s) => s,
        Err(_) => return chat_err("generation error".into()),
    };
    let gr: GenerateReply = match serde_json::from_slice(&sample.payload().to_bytes()) {
        Ok(g) => g,
        Err(e) => return chat_err(format!("bad reply: {e}")),
    };
    let text = st.tok.decode_reply(&gr.tokens).unwrap_or_default();
    Json(ChatResp { reply: text, ttft_ms: gr.ttft_ms, tok_s: gr.decode_tok_s, error: gr.error })
}

#[derive(Deserialize)]
struct StreamParams {
    text: String,
    max_tokens: Option<usize>,
}

/// Server-sent-events version of the chat: kicks off generation, then relays the
/// coordinator's per-step partial token stream ([`znet::STREAM_KEY`]) as decoded
/// text so the answer appears live. A `done` event ends the stream.
async fn api_chat_stream(
    State(st): State<AppState>,
    Query(params): Query<StreamParams>,
) -> impl axum::response::IntoResponse {
    let tok = st.tok.clone();
    let session = st.session.clone();
    let text = params.text;
    let max_tokens = params.max_tokens.unwrap_or(768);

    let stream = async_stream::stream! {
        let ids: Vec<i64> = match tok.encode_prompt(&text) {
            Ok(v) => v,
            Err(e) => {
                yield Ok::<_, std::convert::Infallible>(Event::default().event("error").data(format!("tokenize: {e}")));
                return;
            }
        };
        if ids.is_empty() {
            yield Ok(Event::default().event("error").data("empty prompt"));
            return;
        }
        // Unique per-request key so overlapping generations never cross streams.
        static STREAM_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = STREAM_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Subscribe BEFORE kicking off so no early tokens are missed.
        let sub = match session.declare_subscriber(znet::stream_key(id)).await {
            Ok(s) => s,
            Err(e) => {
                yield Ok(Event::default().event("error").data(format!("subscribe: {e}")));
                return;
            }
        };
        let payload = serde_json::to_vec(&GenerateRequest {
            prompt: ids,
            max_tokens,
            stream_id: id,
            stop: tok.stop_ids(),
        })
        .unwrap();
        let gen_session = session.clone();
        tokio::spawn(async move {
            if let Ok(replies) =
                gen_session.get(znet::GENERATE_KEY).payload(payload).timeout(Duration::from_secs(600)).await
            {
                let _ = replies.recv_async().await; // drain final reply so the query completes
            }
        });
        while let Ok(sample) = sub.recv_async().await {
            let Ok(msg) = serde_json::from_slice::<StreamMsg>(&sample.payload().to_bytes())
            else { continue };
            let text = tok.decode_reply(&msg.tokens).unwrap_or_default();
            yield Ok(Event::default().data(text));
            if msg.done {
                // Must carry data: EventSource discards an event whose data buffer is
                // empty, so the client would never see `done`, never close, and would
                // silently reconnect — re-running the generation.
                yield Ok(Event::default().event("done").data("end"));
                break;
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

const INDEX_HTML: &str = r#"<!doctype html><html><head><meta charset=utf-8>
<title>Dis-tract</title><style>
body{background:#0e1116;color:#d6deeb;font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:18px;margin:0 0 4px}.sub{color:#7a8699;margin-bottom:20px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.card{background:#161b22;border:1px solid #232c39;border-radius:10px;padding:14px}
.row{display:flex;justify-content:space-between;margin:3px 0}
.k{color:#7a8699}.badge{padding:1px 8px;border-radius:6px;font-size:12px;font-weight:600}
.info{display:inline-flex;align-items:center;justify-content:center;width:13px;height:13px;margin-left:5px;
 border:1px solid #3d4757;border-radius:50%;color:#7a8699;font:700 9px/1 georgia,serif;font-style:italic;
 cursor:help;position:relative;vertical-align:1px}
.info:hover{border-color:#8899aa;color:#dfe6ee}
.info:hover::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 7px);left:50%;
 transform:translateX(-50%);width:230px;background:#0b1017;color:#cfd8e3;border:1px solid #2f3a4a;
 border-radius:7px;padding:8px 10px;font:400 11.5px/1.5 -apple-system,system-ui,sans-serif;font-style:normal;
 text-align:left;white-space:normal;z-index:30;box-shadow:0 8px 24px rgba(0,0,0,.6);pointer-events:none}
.info:hover::before{content:'';position:absolute;bottom:calc(100% + 2px);left:50%;transform:translateX(-50%);
 border:5px solid transparent;border-top-color:#2f3a4a;z-index:31;pointer-events:none}
.card:last-child .info:hover::after,.card:nth-child(2n) .info:hover::after{left:auto;right:-8px;transform:none}
.card:last-child .info:hover::before,.card:nth-child(2n) .info:hover::before{left:auto;right:0;transform:none}
.cpu{background:#1f3b5c;color:#7fb4ff}.metal{background:#3d2b52;color:#c79bff}.cuda{background:#123f2e;color:#68d391}
.bar{height:6px;background:#232c39;border-radius:4px;overflow:hidden;margin-top:3px}
.fill{height:100%;background:linear-gradient(90deg,#4f8cff,#8b5cf6)}
.big{font-size:22px;font-weight:700}.title{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.mono{font-family:ui-monospace,monospace;font-size:12px;color:#9aa7b8}
.agg{background:linear-gradient(135deg,#12203a,#1a1430);border:1px solid #2b3550;border-radius:12px;padding:18px 20px;margin-bottom:20px}
.agghead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px}
.model{font-size:16px;font-weight:700;color:#e8eefc}
.tiles{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.tile{background:#0e1420;border:1px solid #222c3e;border-radius:9px;padding:12px 14px}
.tlabel{color:#7a8699;font-size:12px;margin-bottom:4px}
.tval{font-size:24px;font-weight:700;color:#e8eefc}.tunit{font-size:12px;color:#7a8699;font-weight:400;margin-left:3px}
.chat{background:#161b22;border:1px solid #232c39;border-radius:12px;padding:16px;margin-bottom:20px}
.transcript{max-height:320px;overflow-y:auto;margin-bottom:12px}
.msg{margin:8px 0;display:flex}.msg.u{justify-content:flex-end}
.bubble{max-width:80%;padding:9px 13px;border-radius:12px;white-space:pre-wrap}
.u .bubble{background:#1f3b5c;color:#dbe9ff}.a .bubble{background:#1a2130;color:#d6deeb}
.meta{font-size:11px;color:#5f6b7e;margin-top:3px}
.inputrow{display:flex;gap:8px}
#msg{flex:1;background:#0e1420;border:1px solid #2b3550;border-radius:9px;color:#e8eefc;padding:10px 12px;font-size:14px}
#send{background:#3457d5;border:0;border-radius:9px;color:#fff;font-weight:600;padding:0 18px;cursor:pointer}
#send:disabled{opacity:.5;cursor:default}
</style></head><body>
<h1>Dis-tract</h1><div class=sub>distributed tract — live cluster view</div>
<div id=agg></div>
<div class=chat>
  <div class=transcript id=transcript><div class=sub>Ask the distributed model something…</div></div>
  <div class=inputrow>
    <input id=msg placeholder="The capital of France is" autocomplete=off>
    <button id=send>Send</button>
  </div>
</div>
<div class=grid id=grid></div>
<script>
const fmtG=b=>(b/1073741824).toFixed(2)+' GB', fmtM=b=>(b/1048576).toFixed(0)+' MB';
const tile=(label,val,unit)=>`<div class=tile><div class=tlabel>${label}</div><div class=tval>${val}<span class=tunit>${unit||''}</span></div></div>`;
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const info=t=>`<i class=info data-tip="${t.replace(/"/g,'&quot;')}">i</i>`;

const transcript=document.getElementById('transcript'), msg=document.getElementById('msg'), send=document.getElementById('send');
function bubble(role,text,meta){
  const d=document.createElement('div'); d.className='msg '+(role==='u'?'u':'a');
  d.innerHTML=`<div><div class=bubble>${esc(text)}</div>${meta?`<div class=meta>${meta}</div>`:''}</div>`;
  transcript.appendChild(d); transcript.scrollTop=transcript.scrollHeight; return d;
}
async function ask(){
  const text=msg.value.trim(); if(!text) return;
  if(transcript.querySelector('.sub')) transcript.innerHTML='';
  msg.value=''; send.disabled=true; msg.disabled=true;
  bubble('u',text);
  const pend=bubble('a','…','generating across the cluster');
  const b=pend.querySelector('.bubble'), m=pend.querySelector('.meta');
  const done=()=>{ send.disabled=false; msg.disabled=false; msg.focus(); };
  const es=new EventSource('/api/chat/stream?text='+encodeURIComponent(text)+'&max_tokens=768');
  es.onmessage=e=>{ if(e.data){ b.textContent=e.data; m.textContent='generating…'; } };
  es.addEventListener('done',()=>{ es.close(); m.textContent=''; done(); });
  es.addEventListener('error',e=>{ b.textContent='⚠ '+(e.data||'stream error'); m.textContent=''; es.close(); done(); });
  es.onerror=()=>{ es.close(); if(!b.textContent||b.textContent==='…'){ b.textContent='⚠ connection lost'; } m.textContent=''; done(); };
}
send.onclick=ask; msg.addEventListener('keydown',e=>{if(e.key==='Enter')ask();});

async function tick(){
  let n=[],r=null;
  try{n=await (await fetch('/api/nodes')).json()}catch(e){}
  try{r=await (await fetch('/api/run')).json()}catch(e){}
  if(r){
    const set=new Set(r.node_ids||[]);
    const live=n.filter(v=>set.has((v.stats&&v.stats.node_id)||(v.caps&&v.caps.node_id)));
    const rss=live.reduce((a,v)=>a+((v.stats&&v.stats.mem_footprint)||0),0);
    // A token traverses every stage in turn, so the stages' step times ADD; the
    // live end-to-end rate is their reciprocal sum, never a node's own rate.
    const stepSum=live.reduce((a,v)=>a+((v.stats&&v.stats.last_step_ms)||0),0);
    const nowTps=stepSum>0?1000/stepSum:0;
    document.getElementById('agg').innerHTML=`<div class=agg>
      <div class=agghead><div class=model>▶ ${r.model}</div>
        <span class=mono>${r.n_layers} layers · ${r.n_stages} nodes (pipeline: every token traverses all) · prompt ${r.prompt_tokens} tok</span></div>
      <div class=tiles>
        ${tile('time to first token', (r.ttft_ms||0).toFixed(0), 'ms')}
        ${tile('throughput (end-to-end)', (r.decode_tok_s||0).toFixed(1), `tok/s avg · now ${nowTps.toFixed(1)}`)}
        ${tile('tokens generated', r.tokens||0, '')}
        ${tile('model weights', fmtG(r.total_weight_bytes||0), 'distributed')}
        ${tile('aggregate memory', fmtG(rss), `across ${live.length} nodes`)}
      </div></div>`;
  } else { document.getElementById('agg').innerHTML=''; }
  document.getElementById('grid').innerHTML = n.map(v=>{
    const c=v.caps||{}, s=v.stats||{}; const be=(c.backend||s.backend||'?');
    const memPct = s.host_mem_total? 100*s.host_mem_used/s.host_mem_total : 0;
    return `<div class=card>
      <div class=title><div><span class="badge ${be}">${be.toUpperCase()}</span>
        <b style="margin-left:8px">stage ${s.stage??'—'}</b></div>
        <span class=mono>${c.hostname||s.hostname||''}</span></div>
      <div class=mono>${c.node_id||s.node_id||''}</div>
      <div class=row><span class=k>stage rate${info('Wall-clock time this node took for its most recent step, for its own slice of layers only. Step times ADD across stages: a token passes through every stage in turn, so the cluster rate is 1000 / (sum of every stage rate).')}</span><span class=big>${(s.last_step_ms||0).toFixed(1)}<span style="font-size:12px;color:#7a8699"> ms/step</span></span></div>
      <div class=row><span class=k>this stage alone${info('1000 / stage rate — the rate this node could sustain if its layers were the whole model. NOT the cluster rate, and not addable across nodes: a token still needs every other stage. Two balanced stages land near half this.')}</span><span>${(s.tok_s||0).toFixed(1)} tok/s</span></div>
      <div class=row><span class=k>tokens${info('Steps this node has processed since it started (it counts prefill and decode steps alike). This is pipeline parallelism, so every node sees every token — these counts track each other across nodes.')}</span><span>${s.tokens||0}</span></div>
      <div class=row><span class=k>host CPU${info('Whole-machine CPU utilisation sampled by this worker — the entire host, every process, not just this shard. On a Metal node the work is on the GPU, so this stays low even at full tilt.')}</span><span>${(s.host_cpu||0).toFixed(0)}%</span></div>
      <div class=bar><div class=fill style="width:${Math.min(100,s.host_cpu||0)}%"></div></div>
      <div class=row style="margin-top:8px"><span class=k>host memory${info('Whole-machine RAM used / total, sampled by this worker — every process on the box, not just this shard. Co-located workers share one host, so they report the same figure and it must not be summed across them.')}</span><span>${s.host_mem_used?fmtG(s.host_mem_used):'—'} / ${s.host_mem_total?fmtG(s.host_mem_total):'—'}</span></div>
      <div class=bar><div class=fill style="width:${memPct}%"></div></div>
      <div class=row style="margin-top:8px"><span class=k>shard weights${info('Bytes of model weights this shard holds, as actually stored — block-quant (q40) weights are counted packed, not as their dequantized f32 size. This is the split working: each node loads only its own layers, and these sum to the full model across the cluster.')}</span><span>${s.weights_bytes?fmtM(s.weights_bytes):'—'}</span></div>
      <div class=row><span class=k>mem budget${info('The memory cap this node advertised on join (--mem-mb, or --mem-frac of free RAM). The coordinator plans a memory-weighted split from these, so a bigger budget earns a node more layers.')}</span><span>${(c.mem_budget||s.mem_budget)?fmtG(c.mem_budget||s.mem_budget):'—'}</span></div>
      <div class=row><span class=k>process memory${info('Physical memory this worker process actually occupies — packed weights + KV cache + runtime. On macOS this is phys_footprint (what Activity Monitor shows), not resident-set size, which omits compressed and GPU-driver pages. This is the honest per-node cost, and the proof of the split: no node holds the whole model.')}</span><span>${s.mem_footprint?fmtM(s.mem_footprint):'—'}</span></div>
    </div>`}).join('') || '<div class=sub>waiting for nodes…</div>';
}
tick(); setInterval(tick,1000);
</script></body></html>"#;
