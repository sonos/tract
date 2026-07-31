//! LAN caching reverse-proxy for tract's CI/bench model assets. `GET /<key>`
//! serves the cached file, or streams it from the R2 origin while a background
//! thread fills the cache to completion (so a client that disconnects still warms
//! it). `HEAD` is proxied to the origin. Plain HTTP; keys are immutable.
//!
//!   tract-asset-mirror [--listen 0.0.0.0:8080] [--cache-dir ./asset-cache]
//!                      [--origin https://tract-test-assets.tract.rs] [--threads 16]
//! Flags also read env: LISTEN, CACHE_DIR, ORIGIN, THREADS.

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

/// Progress of one background download, shared with the clients tailing its `.part`.
struct Fill {
    written: AtomicU64,
    state: AtomicU8,
    len: Option<u64>,
}

/// In-flight downloads, so concurrent requests for a key share one download.
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
                len: content_length(&resp),
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

/// Pull the whole body into `part` and publish it (atomic rename) on completion,
/// regardless of whether any client is still reading — this is what warms the cache
/// even when the requesting client is killed mid-stream.
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

fn serve_tail(req: Request, part: &Path, cache_path: &Path, fill: Arc<Fill>) {
    let file = match fs::File::open(part) {
        Ok(f) => f,
        // Race: the fill completed and renamed `part` away between join and open.
        Err(_) if cache_path.is_file() => return serve_file(req, cache_path, "HIT"),
        Err(e) => return reply(req, 502, &format!("fill unavailable: {e}")),
    };
    let len = fill.len;
    let reader = TailReader { file, fill, pos: 0 };
    match len {
        Some(len) => respond_body(req, len, reader),
        // Origin sent no Content-Length: fall back to tiny_http's chunked encoding.
        None => {
            let ct =
                Header::from_bytes(&b"Content-Type"[..], &b"application/octet-stream"[..]).unwrap();
            let resp = Response::new(StatusCode(200), vec![ct], reader, None, None)
                .with_chunked_threshold(usize::MAX);
            let _ = req.respond(resp);
        }
    }
}

/// Reads a `.part` as a background download fills it: yields flushed bytes, waits
/// for more, EOF on completion, error if the fill failed (never a silent truncation).
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
            return respond_head(req, meta.len());
        }
    }
    let url = join(&cfg.origin, key);
    match agent.head(&url).call() {
        Ok(r) => match content_length(&r) {
            Some(len) if (200..300).contains(&r.status()) => {
                eprintln!("HEAD {} {key} (origin, {len} bytes)", r.status());
                respond_head(req, len);
            }
            _ => {
                eprintln!("HEAD {} {key} (origin)", r.status());
                let resp =
                    Response::new(StatusCode(r.status()), vec![], io::empty(), Some(0), None);
                let _ = req.respond(resp);
            }
        },
        Err(ureq::Error::Status(code, _)) => reply(req, code, "upstream status"),
        Err(e) => reply(req, 502, &format!("upstream error: {e}")),
    }
}

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
    let f = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) => return reply(req, 500, &format!("open cache: {e}")),
    };
    let len = match f.metadata() {
        Ok(m) => m.len(),
        Err(e) => return reply(req, 500, &format!("stat cache: {e}")),
    };
    eprintln!("GET 200 {} ({tag}, {len} bytes)", path.display());
    respond_body(req, len, f);
}

/// The status line + headers for a 200 with a real `u64` content length. Written
/// straight to the socket via `Request::into_writer`, bypassing tiny_http's `usize`
/// `Content-Length` (which wraps any length >= 4 GiB on the 32-bit Pi). `Connection:
/// close` keeps the raw-write framing simple.
fn head_200(len: u64) -> String {
    format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: application/octet-stream\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\r\n"
    )
}

/// Stream `body` as a 200 with a `u64` content length (for GET).
fn respond_body(req: Request, len: u64, mut body: impl Read) {
    let mut w = req.into_writer();
    if w.write_all(head_200(len).as_bytes()).is_ok() {
        let _ = io::copy(&mut body, &mut w);
    }
    let _ = w.flush();
}

/// Send the headers of a 200 with a `u64` content length and no body (for HEAD).
fn respond_head(req: Request, len: u64) {
    let mut w = req.into_writer();
    let _ = w.write_all(head_200(len).as_bytes());
    let _ = w.flush();
}

fn reply(req: Request, code: u16, msg: &str) {
    let _ =
        req.respond(Response::from_string(format!("{msg}\n")).with_status_code(StatusCode(code)));
}

fn part_path(cache_path: &Path) -> PathBuf {
    let mut s = cache_path.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Keep only normal path components, rejecting `..` / absolute paths.
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

/// Case-insensitive: ureq lowercases header names and its `.header()` is exact-match.
/// Parsed as `u64` (not `usize`): the Pi is 32-bit, and a `usize` would wrap any
/// length >= 4 GiB.
fn content_length(r: &ureq::Response) -> Option<u64> {
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
