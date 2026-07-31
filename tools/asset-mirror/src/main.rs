//! LAN caching reverse-proxy for tract's CI/bench model assets.
//!
//! Serves plain HTTP (fine on a trusted LAN). On `GET /<key>` it serves the file
//! from the local cache dir if present, otherwise starts a background download of
//! it from the R2 origin (`https://tract-test-assets.tract.rs/<key>`) into a
//! `.part` file and streams that to the client as it grows. The download runs on
//! its own thread and always runs to completion, so a client that disconnects
//! (e.g. a bench watchdog kill) still leaves the model fully cached for next time
//! -- the mirror is self-warming. The `.part` is published (atomic rename) only
//! once the whole Content-Length has landed, so a truncated fetch is never cached.
//! Keys are immutable (generation-tagged), so a cached file is never revalidated.
//! `HEAD` is proxied to the origin so a size probe does not pull the body.
//!
//!   tract-asset-mirror [--listen 0.0.0.0:8080] [--cache-dir ./asset-cache]
//!                      [--origin https://tract-test-assets.tract.rs] [--threads 16]
//!
//! Each flag also reads an env fallback: LISTEN, CACHE_DIR, ORIGIN, THREADS.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

struct Config {
    listen: String,
    cache_dir: PathBuf,
    origin: String,
    threads: usize,
}

const FILL_RUNNING: u8 = 0;
const FILL_DONE: u8 = 1;
const FILL_FAILED: u8 = 2;

/// Progress of one background download, shared with every client tailing its
/// `.part`. `written` is bytes flushed so far; `state` moves RUNNING -> DONE/FAILED.
struct Fill {
    written: AtomicU64,
    state: AtomicU8,
    len: Option<u64>,
}

/// Downloads in flight, so concurrent requests for the same key share one download
/// (and its `.part`) instead of racing.
type Inflight = Mutex<HashMap<String, Arc<Fill>>>;

fn main() {
    let cfg = Arc::new(parse_config());
    fs::create_dir_all(&cfg.cache_dir).unwrap_or_else(|e| {
        eprintln!("cannot create cache dir {}: {e}", cfg.cache_dir.display());
        std::process::exit(1);
    });
    let server = Arc::new(Server::http(&cfg.listen).unwrap_or_else(|e| {
        eprintln!("cannot bind {}: {e}", cfg.listen);
        std::process::exit(1);
    }));
    let agent = ureq::AgentBuilder::new().build();
    let inflight: Arc<Inflight> = Arc::new(Mutex::new(HashMap::new()));

    eprintln!(
        "tract-asset-mirror: http://{} -> {} (cache {}, {} threads)",
        cfg.listen,
        cfg.origin,
        cfg.cache_dir.display(),
        cfg.threads
    );

    let mut handles = Vec::new();
    for _ in 0..cfg.threads {
        let server = server.clone();
        let cfg = cfg.clone();
        let agent = agent.clone();
        let inflight = inflight.clone();
        handles.push(thread::spawn(move || {
            while let Ok(req) = server.recv() {
                handle(req, &cfg, &agent, &inflight);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn handle(req: Request, cfg: &Config, agent: &ureq::Agent, inflight: &Arc<Inflight>) {
    let method = req.method().clone();
    let key = match req.url().split('?').next().unwrap_or("").trim_start_matches('/') {
        "" => return reply(req, 400, "empty key"),
        k => k.to_string(),
    };
    let rel = match safe_rel(&key) {
        Some(r) => r,
        None => return reply(req, 400, "bad path"),
    };
    let cache_path = cfg.cache_dir.join(&rel);

    match method {
        Method::Get => get(req, cfg, agent, inflight, &key, &cache_path),
        Method::Head => head(req, cfg, agent, &key, &cache_path),
        _ => reply(req, 405, "method not allowed"),
    }
}

fn get(
    req: Request,
    cfg: &Config,
    agent: &ureq::Agent,
    inflight: &Arc<Inflight>,
    key: &str,
    cache_path: &Path,
) {
    if cache_path.is_file() {
        return serve_file(req, cache_path, "HIT");
    }
    let part = part_path(cache_path);
    // Join an in-flight download for this key, or become the one that starts it.
    let fill = {
        let mut map = inflight.lock().unwrap();
        if cache_path.is_file() {
            drop(map);
            return serve_file(req, cache_path, "HIT");
        }
        if let Some(f) = map.get(key) {
            f.clone()
        } else {
            let resp = match origin_get(agent, &cfg.origin, key) {
                Ok(r) => r,
                Err((code, msg)) => {
                    drop(map);
                    eprintln!("{code} {key} (upstream: {msg})");
                    return reply(req, code, &msg);
                }
            };
            let fill = Arc::new(Fill {
                written: AtomicU64::new(0),
                state: AtomicU8::new(FILL_RUNNING),
                len: content_length(&resp).map(|l| l as u64),
            });
            if let Some(parent) = cache_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if fs::File::create(&part).is_err() {
                drop(map);
                return reply(req, 500, "cannot create cache temp");
            }
            map.insert(key.to_string(), fill.clone());
            eprintln!("GET 200 {key} (streaming + background fill)");
            spawn_downloader(
                resp,
                cache_path.to_path_buf(),
                part.clone(),
                key.to_string(),
                fill.clone(),
                inflight.clone(),
            );
            fill
        }
    };
    serve_tail(req, &part, cache_path, fill);
}

/// Background thread: pull the whole body into `part`, publish on completion. Runs
/// to the end regardless of whether any client is still reading, so the cache warms
/// even when the requesting client (a bench) is killed mid-stream.
fn spawn_downloader(
    resp: ureq::Response,
    cache_path: PathBuf,
    part: PathBuf,
    key: String,
    fill: Arc<Fill>,
    inflight: Arc<Inflight>,
) {
    thread::spawn(move || {
        let ok = download(resp, &part, &fill);
        if ok && fs::rename(&part, &cache_path).is_ok() {
            fill.state.store(FILL_DONE, Ordering::SeqCst);
            eprintln!("fill {key}: cached ({} bytes)", fill.written.load(Ordering::SeqCst));
        } else {
            fill.state.store(FILL_FAILED, Ordering::SeqCst);
            let _ = fs::remove_file(&part);
            eprintln!("fill {key}: FAILED (incomplete upstream / cache io)");
        }
        inflight.lock().unwrap().remove(&key);
    });
}

/// Copy the origin body into `part`, bumping `fill.written` as bytes are flushed.
/// Returns whether the full Content-Length landed (always true when length unknown).
fn download(resp: ureq::Response, part: &Path, fill: &Fill) -> bool {
    let mut reader = resp.into_reader();
    let mut file = match fs::OpenOptions::new().write(true).truncate(true).open(part) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if file.write_all(&buf[..n]).is_err() {
                    return false;
                }
                fill.written.fetch_add(n as u64, Ordering::Release);
            }
            Err(_) => return false,
        }
    }
    let _ = file.flush();
    let _ = file.sync_all();
    match fill.len {
        Some(e) => fill.written.load(Ordering::Acquire) >= e,
        None => true,
    }
}

/// Serve a client by tailing the `.part` as the background download fills it.
fn serve_tail(req: Request, part: &Path, cache_path: &Path, fill: Arc<Fill>) {
    let file = match fs::File::open(part) {
        Ok(f) => f,
        // The download finished and renamed `part` away between join and open.
        Err(_) if cache_path.is_file() => return serve_file(req, cache_path, "HIT"),
        Err(e) => return reply(req, 502, &format!("fill unavailable: {e}")),
    };
    let ct = Header::from_bytes(&b"Content-Type"[..], &b"application/octet-stream"[..]).unwrap();
    let len = fill.len.map(|l| l as usize);
    let reader = TailReader { file, fill, pos: 0 };
    let resp = Response::new(StatusCode(200), vec![ct], reader, len, None)
        .with_chunked_threshold(usize::MAX);
    let _ = req.respond(resp);
}

/// `Read` over a `.part` being written by a background download: yields whatever
/// has been flushed, waits for more, ends at EOF when the fill completes, and
/// errors if the fill failed (so a truncated download is never served as complete).
struct TailReader {
    file: fs::File,
    fill: Arc<Fill>,
    pos: u64,
}

impl Read for TailReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            let written = self.fill.written.load(Ordering::Acquire);
            if written > self.pos {
                let want = ((written - self.pos) as usize).min(buf.len());
                let n = self.file.read(&mut buf[..want])?;
                self.pos += n as u64;
                return Ok(n);
            }
            match self.fill.state.load(Ordering::SeqCst) {
                FILL_DONE => return Ok(0),
                FILL_FAILED => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "upstream fill failed",
                    ))
                }
                _ => thread::sleep(Duration::from_millis(20)),
            }
        }
    }
}

fn head(req: Request, cfg: &Config, agent: &ureq::Agent, key: &str, cache_path: &Path) {
    if let Ok(meta) = fs::metadata(cache_path) {
        if meta.is_file() {
            eprintln!("HEAD 200 {key} (cached, {} bytes)", meta.len());
            let resp = Response::new(
                StatusCode(200),
                vec![],
                io::empty(),
                Some(meta.len() as usize),
                None,
            )
            .with_chunked_threshold(usize::MAX);
            let _ = req.respond(resp);
            return;
        }
    }
    let url = join(&cfg.origin, key);
    match agent.head(&url).call() {
        Ok(r) => {
            let len: Option<usize> = content_length(&r);
            eprintln!("HEAD {} {key} (origin)", r.status());
            let resp = Response::new(StatusCode(r.status()), vec![], io::empty(), len, None)
                .with_chunked_threshold(usize::MAX);
            let _ = req.respond(resp);
        }
        Err(ureq::Error::Status(code, _)) => reply(req, code, "upstream status"),
        Err(e) => reply(req, 502, &format!("upstream error: {e}")),
    }
}

/// Open the origin GET for `key`; the body is not read yet.
fn origin_get(
    agent: &ureq::Agent,
    origin: &str,
    key: &str,
) -> Result<ureq::Response, (u16, String)> {
    match agent.get(&join(origin, key)).call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::Status(code, _)) => Err((code, format!("upstream {code}"))),
        Err(e) => Err((502, format!("upstream error: {e}"))),
    }
}

fn serve_file(req: Request, path: &Path, tag: &str) {
    match fs::File::open(path) {
        Ok(f) => {
            eprintln!("GET 200 {} ({tag})", path.display());
            let ct =
                Header::from_bytes(&b"Content-Type"[..], &b"application/octet-stream"[..]).unwrap();
            let resp = Response::from_file(f).with_header(ct).with_chunked_threshold(usize::MAX);
            let _ = req.respond(resp);
        }
        Err(e) => reply(req, 500, &format!("open cache: {e}")),
    }
}

fn reply(req: Request, code: u16, msg: &str) {
    let _ =
        req.respond(Response::from_string(format!("{msg}\n")).with_status_code(StatusCode(code)));
}

/// Sibling `.part` path for a cache entry (deterministic per key so concurrent
/// tailers open the same one).
fn part_path(cache_path: &Path) -> PathBuf {
    let mut s = cache_path.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Reject anything that would escape the cache dir (`..`, absolute paths); keep
/// only normal path components.
fn safe_rel(key: &str) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for comp in Path::new(key).components() {
        match comp {
            Component::Normal(c) => out.push(c),
            _ => return None,
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Case-insensitive Content-Length lookup (ureq lowercases header names, and
/// origins vary in casing).
fn content_length(r: &ureq::Response) -> Option<usize> {
    r.headers_names()
        .iter()
        .find(|n| n.eq_ignore_ascii_case("content-length"))
        .and_then(|n| r.header(n))
        .and_then(|s| s.parse().ok())
}

fn join(origin: &str, key: &str) -> String {
    format!("{}/{}", origin.trim_end_matches('/'), key)
}

fn parse_config() -> Config {
    let mut listen = std::env::var("LISTEN").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let mut cache_dir = std::env::var("CACHE_DIR").unwrap_or_else(|_| "./asset-cache".into());
    let mut origin =
        std::env::var("ORIGIN").unwrap_or_else(|_| "https://tract-test-assets.tract.rs".into());
    let mut threads: usize =
        std::env::var("THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_default();
        match a.as_str() {
            "--listen" => listen = val(),
            "--cache-dir" => cache_dir = val(),
            "--origin" => origin = val(),
            "--threads" => threads = val().parse().unwrap_or(threads),
            "-h" | "--help" => {
                eprintln!(
                    "usage: tract-asset-mirror [--listen ADDR] [--cache-dir DIR] [--origin URL] [--threads N]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    Config { listen, cache_dir: PathBuf::from(cache_dir), origin, threads: threads.max(1) }
}
