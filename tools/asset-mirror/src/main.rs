//! LAN caching reverse-proxy for tract's CI/bench model assets.
//!
//! Serves plain HTTP (fine on a trusted LAN). On `GET /<key>` it serves the file
//! from the local cache dir if present, otherwise streams it from the R2 origin
//! (`https://tract-test-assets.tract.rs/<key>`) to the client while teeing it into
//! the cache, so the first byte is immediate even on a cold miss.
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

struct Config {
    listen: String,
    cache_dir: PathBuf,
    origin: String,
    threads: usize,
}

/// Coalesces concurrent first-misses for the same key so it is fetched once.
type KeyLocks = Mutex<HashMap<String, Arc<Mutex<()>>>>;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

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
    let locks: Arc<KeyLocks> = Arc::new(Mutex::new(HashMap::new()));

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
        let locks = locks.clone();
        handles.push(thread::spawn(move || {
            while let Ok(req) = server.recv() {
                handle(req, &cfg, &agent, &locks);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn handle(req: Request, cfg: &Config, agent: &ureq::Agent, locks: &KeyLocks) {
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
        Method::Get => get(req, cfg, agent, locks, &key, &cache_path),
        Method::Head => head(req, cfg, agent, &key, &cache_path),
        _ => reply(req, 405, "method not allowed"),
    }
}

fn get(
    req: Request,
    cfg: &Config,
    agent: &ureq::Agent,
    locks: &KeyLocks,
    key: &str,
    cache_path: &Path,
) {
    if cache_path.is_file() {
        return serve_file(req, cache_path, "HIT");
    }
    let lock = key_lock(locks, key);
    let _guard = lock.lock().unwrap();
    // Another thread may have fetched it while we waited on the lock.
    if cache_path.is_file() {
        return serve_file(req, cache_path, "HIT");
    }
    match origin_get(agent, &cfg.origin, key) {
        Ok(resp) => stream_and_cache(req, resp, cache_path, key),
        Err((code, msg)) => {
            eprintln!("{code} {key} (upstream: {msg})");
            reply(req, code, &msg);
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

/// Stream the origin response to the client while teeing it into the cache. The
/// first byte reaches the client immediately (no fetch-then-serve stall), and the
/// cache file is only published (temp + atomic rename) once the full body has
/// been written, so an interrupted transfer never leaves a partial file. If the
/// cache write fails (e.g. mirror disk full) it degrades to plain pass-through.
fn stream_and_cache(req: Request, resp: ureq::Response, cache_path: &Path, key: &str) {
    let expected = content_length(&resp).map(|l| l as u64);
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = cache_path.with_file_name(format!(
        ".{}.tmp.{}.{}",
        cache_path.file_name().and_then(|s| s.to_str()).unwrap_or("part"),
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed),
    ));
    let tee = TeeReader {
        src: Box::new(resp.into_reader()),
        tmp: fs::File::create(&tmp).ok(),
        tmp_path: tmp,
        final_path: cache_path.to_path_buf(),
        expected,
        written: 0,
        done: false,
    };
    eprintln!("GET 200 {key} (streaming)");
    let ct = Header::from_bytes(&b"Content-Type"[..], &b"application/octet-stream"[..]).unwrap();
    let resp = Response::new(StatusCode(200), vec![ct], tee, expected.map(|l| l as usize), None)
        .with_chunked_threshold(usize::MAX);
    let _ = req.respond(resp);
}

/// Reader that copies everything it yields into a cache temp file, renaming it
/// into place once the whole body is seen. See [`stream_and_cache`].
struct TeeReader {
    src: Box<dyn Read + Send>,
    tmp: Option<fs::File>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    expected: Option<u64>,
    written: u64,
    done: bool,
}

impl Read for TeeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.src.read(buf)?;
        if n > 0 {
            if let Some(f) = self.tmp.as_mut() {
                if f.write_all(&buf[..n]).is_err() {
                    self.tmp = None; // mirror disk trouble: stop caching, keep serving
                    let _ = fs::remove_file(&self.tmp_path);
                } else {
                    self.written += n as u64;
                }
            }
        }
        // Only publish once the whole body is in. On a short read (upstream EOF
        // before Content-Length) leave it unpublished, so a truncated download is
        // never renamed into the cache and served as if complete.
        let complete = self.tmp.is_some()
            && match self.expected {
                Some(e) => self.written >= e,
                None => n == 0,
            };
        if complete && !self.done {
            if let Some(mut f) = self.tmp.take() {
                let ok = f.flush().is_ok() && f.sync_all().is_ok();
                drop(f);
                if ok && fs::rename(&self.tmp_path, &self.final_path).is_ok() {
                    self.done = true;
                } else {
                    let _ = fs::remove_file(&self.tmp_path);
                }
            }
        }
        Ok(n)
    }
}

impl Drop for TeeReader {
    fn drop(&mut self) {
        if !self.done {
            let _ = fs::remove_file(&self.tmp_path);
        }
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

fn key_lock(locks: &KeyLocks, key: &str) -> Arc<Mutex<()>> {
    locks.lock().unwrap().entry(key.to_string()).or_insert_with(|| Arc::new(Mutex::new(()))).clone()
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
