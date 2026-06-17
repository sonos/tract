//! The bench-suite orchestrator (`tract bench-suite`). Reads a TOML manifest of
//! benches and, for each one, spawns a fresh `tract` child in `--emit-jsonl` mode
//! so every measurement gets the cold process the memory readings need. Each
//! child's stdout is pure JSONL (logs go to stderr): the orchestrator parses every
//! line, treats anything that is not a metric object as a hard failure, prefixes
//! the metric names and writes them to the metrics file. This replaces the shell
//! bundle (model fetch, governor pinning, scraping) with one cross-built binary.

use crate::bench_common::higher_better;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use tract_hir::internal::*;

#[derive(Deserialize)]
struct Manifest {
    /// HTTPS prefix the model files are fetched from (public-read bucket).
    base_url: String,
    #[serde(default, rename = "bench")]
    benches: Vec<Bench>,
}

#[derive(Deserialize)]
struct Bench {
    kind: Kind,
    name: String,
    /// Series label when `backends` is empty (a single plain CPU run).
    #[serde(default)]
    variant: Option<String>,
    /// Model path relative to the cache dir (and to `base_url` when fetched).
    model: String,
    /// Archive to fetch and unpack that yields `model`, for models shipped inside
    /// a tarball. When unset, `model` is fetched as a plain file.
    #[serde(default)]
    archive: Option<String>,
    /// Extra tract arguments (input facts, `--pulse`, loader flags, ...). Either a TOML
    /// array (`["-i", "264,40"]`) or a plain string that is split on whitespace
    /// (`"-i 264,40 --pulse 24"`), so a working argument line can be pasted as-is.
    #[serde(default, deserialize_with = "de_args")]
    args: Vec<String>,
    /// Backends to sweep. Empty: one plain CPU run labelled `variant`. Non-empty:
    /// one run per available backend, each labelled by the backend name.
    #[serde(default)]
    backends: Vec<Backend>,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Net,
    Llm,
}

impl Kind {
    fn label(&self) -> &'static str {
        match self {
            Kind::Net => "net",
            Kind::Llm => "llm",
        }
    }
    fn subcommand(&self) -> &'static str {
        match self {
            Kind::Net => "bench",
            Kind::Llm => "llm-bench",
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Cpu,
    Metal,
    Cuda,
}

impl Backend {
    fn label(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
        }
    }

    fn available(&self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::Metal => cfg!(target_os = "macos"),
            Backend::Cuda => Command::new("nvidia-smi")
                .arg("-L")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false),
        }
    }

    /// `(global flags, subcommand flags)` for this backend.
    fn flags(&self) -> (&'static [&'static str], &'static [&'static str]) {
        match self {
            Backend::Cpu => (&["--timeout", "180"], &[]),
            Backend::Metal => (&["--metal", "--timeout", "60"], &["--warmup-loops", "1"]),
            Backend::Cuda => (&["--cuda", "--timeout", "60"], &["--warmup-loops", "1"]),
        }
    }
}

#[derive(Deserialize)]
struct MetricLine {
    metric: String,
    value: f64,
}

/// Accept `args` as either a string (split on whitespace) or an array of strings.
fn de_args<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Args {
        Line(String),
        List(Vec<String>),
    }
    Ok(match Args::deserialize(d)? {
        Args::Line(s) => s.split_whitespace().map(String::from).collect(),
        Args::List(v) => v,
    })
}

pub fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    let manifest_path =
        matches.get_one::<String>("manifest").map(String::as_str).unwrap_or("benches.toml");
    let manifest: Manifest = toml::from_str(
        &std::fs::read_to_string(manifest_path)
            .with_context(|| format!("reading manifest {manifest_path}"))?,
    )
    .with_context(|| format!("parsing manifest {manifest_path}"))?;

    let cache_dir = PathBuf::from(
        matches
            .get_one::<String>("cache-dir")
            .cloned()
            .or_else(|| std::env::var("CACHEDIR").ok())
            .unwrap_or_else(|| {
                format!(
                    "{}/.cache/tract-ci-minion-models",
                    std::env::var("HOME").unwrap_or_default()
                )
            }),
    );
    std::fs::create_dir_all(&cache_dir)?;

    let output = matches.get_one::<String>("output").map(String::as_str).unwrap_or("metrics");
    let no_fetch = matches.get_flag("no-fetch");
    let filter = matches.get_one::<String>("filter").map(String::as_str);

    let expectations = expectations(matches)?;
    let retry_max: usize =
        matches.get_one::<String>("retry-max").map(|s| s.parse()).transpose()?.unwrap_or(2);

    let exe = std::env::current_exe()?;
    set_governor_max();
    let _gpu_clock = GpuClock::pin(); // reset on drop, when the suite finishes

    let start = Instant::now();
    let mut metrics: Vec<(String, f64)> = vec![];

    for bench in &manifest.benches {
        if filter.is_some_and(|f| !bench.name.contains(f)) {
            continue;
        }

        let runs: Vec<(Option<Backend>, String)> = if bench.backends.is_empty() {
            let variant = bench
                .variant
                .clone()
                .with_context(|| format!("{}: no variant and no backends", bench.name))?;
            vec![(None, variant)]
        } else {
            bench
                .backends
                .iter()
                .filter(|b| b.available())
                .map(|b| (Some(*b), b.label().to_string()))
                .collect()
        };
        if runs.is_empty() {
            continue;
        }

        if !no_fetch {
            if let Err(e) = fetch(&manifest.base_url, &cache_dir, bench) {
                eprintln!("  !! {}: fetch failed: {e:#}", bench.name);
                continue;
            }
        }

        for (backend, variant) in runs {
            println!("  {} {}", bench.name, variant);
            let run = || run_one(&exe, &cache_dir, bench, backend, &variant);
            match bench_run(run, &expectations, retry_max) {
                Ok(m) => metrics.extend(m),
                Err(e) => eprintln!("  !! {} {} failed: {e:#}", bench.name, variant),
            }
        }
    }

    metrics.push(("bundle.bench_runtime".to_string(), start.elapsed().as_secs() as f64));
    write_metrics(output, &metrics)?;
    println!("wrote {} metrics to {output}", metrics.len());
    Ok(())
}

/// Spawn one child `tract` for a single (bench, backend) and collect its metrics.
/// The child runs in `--emit-jsonl` mode, so its stdout must be pure JSONL; any
/// line that is not a metric object, a non-zero exit, or an empty result is an
/// error for this run (the caller logs it and moves on).
fn run_one(
    exe: &Path,
    cache_dir: &Path,
    bench: &Bench,
    backend: Option<Backend>,
    variant: &str,
) -> TractResult<Vec<(String, f64)>> {
    let (global_flags, sub_flags) = backend.map(|b| b.flags()).unwrap_or((&[], &[]));

    let mut cmd = Command::new(exe);
    cmd.arg(cache_dir.join(&bench.model));
    cmd.args(&bench.args);
    cmd.args(global_flags);
    cmd.args(["--readings", "--readings-heartbeat", "1000", "--emit-jsonl"]);
    if bench.kind == Kind::Llm {
        cmd.arg("--llm");
    }
    cmd.args(["-O", bench.kind.subcommand()]);
    if bench.kind == Kind::Net {
        cmd.arg("--allow-random-input");
    }
    cmd.args(sub_flags);
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());

    let output = cmd.output().context("spawning child tract")?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut metrics = vec![];
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let parsed: MetricLine = serde_json::from_str(line)
            .with_context(|| format!("child wrote non-JSON on stdout: {line:?}"))?;
        let key = format!("{}.{}.{}.{}", bench.kind.label(), bench.name, parsed.metric, variant)
            .replace('-', "_");
        metrics.push((key, parsed.value));
    }
    ensure!(output.status.success(), "child exited with {}", output.status);
    ensure!(!metrics.is_empty(), "child produced no metrics");
    Ok(metrics)
}

/// Run a single bench, then with expectations re-run it (a fresh child each time)
/// while it stays out of threshold — same bar as the report's reds — keeping the
/// per-metric best, up to `retry_max` re-runs. Without expectations: single shot.
fn bench_run(
    run: impl Fn() -> TractResult<Vec<(String, f64)>>,
    expectations: &HashMap<String, (f64, f64)>,
    retry_max: usize,
) -> TractResult<Vec<(String, f64)>> {
    let mut best: BTreeMap<String, f64> = run()?.into_iter().collect();
    if !expectations.is_empty() {
        let mut tries = 0;
        while tries < retry_max && out_of_threshold(&best, expectations) {
            tries += 1;
            println!("    retry {tries} (off expectation)");
            match run() {
                Ok(cand) => merge_best(&mut best, cand),
                Err(e) => eprintln!("    retry {tries} failed: {e:#}"),
            }
        }
    }
    Ok(best.into_iter().collect())
}

/// True if some metric moved worse-than-expected (lower for throughput, higher
/// otherwise) by at least its threshold.
fn out_of_threshold(
    metrics: &BTreeMap<String, f64>,
    expectations: &HashMap<String, (f64, f64)>,
) -> bool {
    metrics.iter().any(|(name, &v)| {
        expectations.get(name).is_some_and(|&(expected, thr)| {
            if expected <= 0.0 {
                return false;
            }
            let pct = (v - expected) / expected * 100.0;
            let worse = if higher_better(name) { -pct } else { pct };
            worse >= thr
        })
    })
}

/// Fold `cand` into `best`, keeping the better value per metric (max for throughput,
/// min otherwise).
fn merge_best(best: &mut BTreeMap<String, f64>, cand: Vec<(String, f64)>) {
    for (k, v) in cand {
        best.entry(k.clone())
            .and_modify(|bv| {
                if if higher_better(&k) { v > *bv } else { v < *bv } {
                    *bv = v;
                }
            })
            .or_insert(v);
    }
}

/// Resolve the expectations that drive retry: an explicit `--expectations` file, or
/// computed inline from bench-data history on the bench host (`--bench-data` plus
/// `--thresholds`/`--triple`/`--device`). Neither given: empty (single shot).
fn expectations(matches: &clap::ArgMatches) -> TractResult<HashMap<String, (f64, f64)>> {
    let get = |k| matches.get_one::<String>(k).map(String::as_str);
    if let Some(path) = get("expectations") {
        return load_expectations(path);
    }
    let Some(bench_data) = get("bench-data") else { return Ok(HashMap::new()) };
    let thresholds = get("thresholds").context("--thresholds is required with --bench-data")?;
    let triple = get("triple").context("--triple is required with --bench-data")?;
    let device = get("device").context("--device is required with --bench-data")?;
    let window: usize = get("window").map(str::parse).transpose()?.unwrap_or(10);
    let rows = crate::bench_expectations::compute(bench_data, thresholds, triple, device, window)?;
    println!("expectations: {} gated metrics from {bench_data}/{triple}/{device}", rows.len());
    Ok(rows.into_iter().map(|(m, e, t)| (m, (e, t))).collect())
}

/// Parse a `metric expected threshold` expectations file (keys underscored, as in
/// bench-data). Missing path or unreadable file yields an empty map (single-shot).
fn load_expectations(path: &str) -> TractResult<HashMap<String, (f64, f64)>> {
    let mut out = HashMap::new();
    let Ok(content) = std::fs::read_to_string(path) else { return Ok(out) };
    for line in content.lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if let [name, expected, threshold] = p[..] {
            if let (Ok(e), Ok(t)) = (expected.parse(), threshold.parse()) {
                out.insert(name.replace('-', "_"), (e, t));
            }
        }
    }
    Ok(out)
}

fn write_metrics(path: &str, metrics: &[(String, f64)]) -> TractResult<()> {
    let mut file = std::fs::File::create(path)?;
    for (k, v) in metrics {
        writeln!(file, "{k} {v}")?;
    }
    Ok(())
}

/// Fetch `bench`'s model into the cache dir if missing, unpacking the archive
/// first when the model ships inside one.
fn fetch(base_url: &str, cache_dir: &Path, bench: &Bench) -> TractResult<()> {
    if cache_dir.join(&bench.model).exists() {
        return Ok(());
    }
    if let Some(archive) = &bench.archive {
        let archive_path = cache_dir.join(archive);
        download(base_url, archive, &archive_path)?;
        let file = std::fs::File::open(&archive_path)?;
        tar::Archive::new(flate2::read::GzDecoder::new(file))
            .unpack(cache_dir)
            .with_context(|| format!("unpacking {}", archive_path.display()))?;
    } else {
        download(base_url, &bench.model, &cache_dir.join(&bench.model))?;
    }
    Ok(())
}

fn download(base_url: &str, name: &str, dest: &Path) -> TractResult<()> {
    let url = format!("{}/{}", base_url.trim_end_matches('/'), name);
    println!("  fetching {url}");
    let mut resp = crate::params::http_client()?.get(&url).send()?;
    ensure!(resp.status().is_success(), "GET {url} -> {}", resp.status());
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp)?;
    std::io::copy(&mut resp, &mut file)?;
    file.sync_all()?;
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

/// Pins the GPU graphics clock for the run and resets it on drop. The cuda runner
/// free-boosts from idle, adding session-to-session variance to evaltime that the
/// in-run retry can't damp. Best-effort: needs privilege, no-op without nvidia-smi.
/// `BENCH_GPU_CLOCK` (MHz) overrides the default (the top supported clock).
struct GpuClock {
    locked: bool,
}

impl GpuClock {
    fn pin() -> GpuClock {
        let query = || {
            let out = Command::new("nvidia-smi")
                .args(["--query-supported-clocks=graphics", "--format=csv,noheader,nounits"])
                .output()
                .ok()?;
            out.status.success().then(|| {
                String::from_utf8_lossy(&out.stdout).lines().next().map(|l| l.trim().to_string())
            })?
        };
        let clock = std::env::var("BENCH_GPU_CLOCK").ok().filter(|c| !c.is_empty()).or_else(query);
        let Some(clock) = clock.filter(|c| !c.is_empty()) else {
            return GpuClock { locked: false };
        };
        let nvidia = |args: &[String]| {
            Command::new("nvidia-smi")
                .args(args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        };
        let _ = nvidia(&["-pm".into(), "1".into()]);
        let locked =
            nvidia(&[format!("--lock-gpu-clocks={clock}")]).map(|s| s.success()).unwrap_or(false);
        if !locked {
            eprintln!("Warning: could not lock GPU clocks (need privilege?)");
        }
        GpuClock { locked }
    }
}

impl Drop for GpuClock {
    fn drop(&mut self) {
        if self.locked {
            let _ = Command::new("nvidia-smi")
                .arg("--reset-gpu-clocks")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

/// On Linux with a `userspace` governor, pin cpu0 to its top available frequency
/// (best effort; needs privilege). Mirrors the shell bundle's determinism step.
fn set_governor_max() {
    if !cfg!(target_os = "linux") {
        return;
    }
    let base = "/sys/devices/system/cpu/cpu0/cpufreq";
    let governor = std::fs::read_to_string(format!("{base}/scaling_governor"));
    if governor.map(|g| g.trim() == "userspace").unwrap_or(false) {
        if let Ok(freqs) = std::fs::read_to_string(format!("{base}/scaling_available_frequencies"))
        {
            if let Some(max) = freqs.split_whitespace().filter_map(|f| f.parse::<u64>().ok()).max()
            {
                let _ = std::fs::write(format!("{base}/scaling_setspeed"), max.to_string());
            }
        }
    }
}
