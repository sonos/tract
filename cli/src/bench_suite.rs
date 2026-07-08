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

/// One (bench, backend/variant) run kept whole — its prefixed metrics plus the
/// coordinates needed to re-run it in the second pass.
struct RunResult {
    bench_idx: usize,
    backend: Option<Backend>,
    variant: String,
    metrics: Vec<(String, f64)>,
}

tract_core::declare_knob!(
    TRACT_BENCH_BASE_URL,
    Option<String>,
    None,
    "Override the manifest's model base URL (e.g. a private mirror), resolved out of band."
);

pub fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    let manifest_path =
        matches.get_one::<String>("manifest").map(String::as_str).unwrap_or("benches.toml");
    let manifest: Manifest = toml::from_str(
        &std::fs::read_to_string(manifest_path)
            .with_context(|| format!("reading manifest {manifest_path}"))?,
    )
    .with_context(|| format!("parsing manifest {manifest_path}"))?;

    let no_cache = matches.get_flag("no-cache");
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
    if !no_cache {
        std::fs::create_dir_all(&cache_dir)?;
    }

    let output = matches.get_one::<String>("output").map(String::as_str).unwrap_or("metrics");
    // A --base-url / --cpu-governor flag sets its knob's override (highest priority). The flag is how
    // a remote (dinghy) run feeds these in: passed as `${VAR}`, expanded from the device config on the
    // dispatching host, so a private mirror URL never reaches the command line or the run log.
    for (flag, knob) in
        [("base-url", &TRACT_BENCH_BASE_URL), ("cpu-governor", &TRACT_BENCH_CPU_GOVERNOR)]
    {
        if let Some(v) = matches.get_one::<String>(flag).filter(|s| !s.is_empty()) {
            knob.set(Some(v.clone()));
        }
    }
    let base_url = TRACT_BENCH_BASE_URL
        .get()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| manifest.base_url.clone());
    let model_source = if no_cache {
        ModelSource::Url(base_url.clone())
    } else {
        ModelSource::Cache(cache_dir.clone())
    };
    let no_fetch = matches.get_flag("no-fetch");
    let skip_cpu = matches.get_flag("skip-cpu");
    let skip_backends = matches.get_flag("skip-backends");
    let filter = matches.get_one::<String>("filter").map(String::as_str);

    let expectations = expectations(matches)?;
    let retry_max: usize =
        matches.get_one::<String>("retry-max").map(|s| s.parse()).transpose()?.unwrap_or(2);
    let second_pass_max: usize =
        matches.get_one::<String>("second-pass-max").map(|s| s.parse()).transpose()?.unwrap_or(2);
    let samples: usize =
        matches.get_one::<String>("samples").map(|s| s.parse()).transpose()?.unwrap_or(0);

    let exe = std::env::current_exe()?;
    set_governor();
    let _gpu_clock = GpuClock::pin(); // reset on drop, when the suite finishes

    let start = Instant::now();
    let mut results: Vec<RunResult> = vec![];

    for (bench_idx, bench) in manifest.benches.iter().enumerate() {
        if filter.is_some_and(|f| !bench.name.contains(f)) {
            continue;
        }
        if skip_backends && !bench.backends.is_empty() {
            continue;
        }

        let runs: Vec<(Option<Backend>, String)> = if bench.backends.is_empty() {
            // Backend-less benches are a plain CPU run; --skip-cpu drops them entirely.
            if skip_cpu {
                vec![]
            } else {
                let variant = bench
                    .variant
                    .clone()
                    .with_context(|| format!("{}: no variant and no backends", bench.name))?;
                vec![(None, variant)]
            }
        } else {
            bench
                .backends
                .iter()
                .filter(|b| b.available())
                .filter(|b| !(skip_cpu && matches!(**b, Backend::Cpu)))
                .map(|b| (Some(*b), b.label().to_string()))
                .collect()
        };
        if runs.is_empty() {
            continue;
        }

        if !no_cache && !no_fetch {
            if let Err(e) = fetch(&base_url, &cache_dir, bench) {
                eprintln!("  !! {}: fetch failed: {e:#}", bench.name);
                continue;
            }
        }

        for (backend, variant) in runs {
            eprintln!("  {} {}", bench.name, variant);
            let run = || run_one(&exe, &model_source, bench, backend, &variant);
            let outcome = if samples > 1 {
                bench_median(run, samples)
            } else {
                bench_run(run, &expectations, retry_max)
            };
            match outcome {
                Ok(m) => results.push(RunResult { bench_idx, backend, variant, metrics: m }),
                Err(e) => eprintln!("  !! {} {} failed: {e:#}", bench.name, variant),
            }
        }
    }

    // Reference mode (median) is not a PR comparison, so it has no reds to chase.
    if samples <= 1 {
        second_pass(
            &exe,
            &model_source,
            &manifest,
            &expectations,
            retry_max,
            second_pass_max,
            &mut results,
        );
    }

    let mut metrics: Vec<(String, f64)> = results.into_iter().flat_map(|r| r.metrics).collect();
    metrics.push(("bundle.bench_runtime".to_string(), start.elapsed().as_secs() as f64));
    write_metrics(output, &metrics)?;
    eprintln!("wrote {} metrics to {output}", metrics.len());
    Ok(())
}

/// Where a child gets its model: a file under the local cache dir, or a URL under
/// `base_url` streamed straight into tract (no disk), for read-only targets.
enum ModelSource {
    Cache(PathBuf),
    Url(String),
}

impl ModelSource {
    fn model_arg(&self, model: &str) -> String {
        match self {
            ModelSource::Cache(dir) => dir.join(model).to_string_lossy().into_owned(),
            ModelSource::Url(base) => format!("{}/{}", base.trim_end_matches('/'), model),
        }
    }
}

/// Spawn one child `tract` for a single (bench, backend) and collect its metrics.
/// The child runs in `--emit-jsonl` mode, so its stdout must be pure JSONL; any
/// line that is not a metric object, a non-zero exit, or an empty result is an
/// error for this run (the caller logs it and moves on).
fn run_one(
    exe: &Path,
    source: &ModelSource,
    bench: &Bench,
    backend: Option<Backend>,
    variant: &str,
) -> TractResult<Vec<(String, f64)>> {
    let (global_flags, sub_flags) = backend.map(|b| b.flags()).unwrap_or((&[], &[]));

    let mut cmd = Command::new(exe);
    cmd.arg(source.model_arg(&bench.model));
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
            eprintln!("    retry {tries} (off expectation)");
            match run() {
                Ok(cand) => merge_best(&mut best, cand),
                Err(e) => eprintln!("    retry {tries} failed: {e:#}"),
            }
        }
    }
    Ok(best.into_iter().collect())
}

/// After the whole suite has run, re-run the benches still over threshold. A brief
/// (dozens of seconds) disruption traps a bench *and* its adjacent inline retries in
/// the same window; re-running now, minutes removed from any mid-suite glitch, lets a
/// transient clear while keeping the per-metric best. Bounded by `second_pass_max`:
/// more reds than that is a real regression or a suite-long problem, not a transient,
/// so they are left standing. No-op without expectations.
fn second_pass(
    exe: &Path,
    source: &ModelSource,
    manifest: &Manifest,
    expectations: &HashMap<String, (f64, f64)>,
    retry_max: usize,
    second_pass_max: usize,
    results: &mut [RunResult],
) {
    if expectations.is_empty() {
        return;
    }
    let red: Vec<usize> = results
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            let m: BTreeMap<String, f64> = r.metrics.iter().cloned().collect();
            out_of_threshold(&m, expectations)
        })
        .map(|(i, _)| i)
        .collect();
    if red.is_empty() {
        return;
    }
    if red.len() > second_pass_max {
        eprintln!(
            "second pass skipped: {} reds (> {second_pass_max}) is not a transient disruption",
            red.len()
        );
        return;
    }
    for i in red {
        let (bench_idx, backend, variant) =
            (results[i].bench_idx, results[i].backend, results[i].variant.clone());
        let bench = &manifest.benches[bench_idx];
        eprintln!("  second pass: {} {}", bench.name, variant);
        let run = || run_one(exe, source, bench, backend, &variant);
        match bench_run(run, expectations, retry_max) {
            Ok(cand) => {
                let mut best: BTreeMap<String, f64> =
                    std::mem::take(&mut results[i].metrics).into_iter().collect();
                merge_best(&mut best, cand);
                results[i].metrics = best.into_iter().collect();
            }
            Err(e) => eprintln!("  !! second pass {} {} failed: {e:#}", bench.name, variant),
        }
    }
}

/// Reference statistics: run the bench `samples` times (a fresh child each) and record
/// the per-metric median. A median drops a lone clock-glitch outlier (spuriously fast /
/// sub-floor time) that `bench_run`'s keep-best would instead latch onto — so the stored
/// nightly reference can't be poisoned by a single bad draw. Used for the reference run;
/// PR runs stay on `bench_run` (retry-until-good-enough).
fn bench_median(
    run: impl Fn() -> TractResult<Vec<(String, f64)>>,
    samples: usize,
) -> TractResult<Vec<(String, f64)>> {
    let mut acc: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut ok = 0;
    for i in 0..samples {
        match run() {
            Ok(m) => {
                ok += 1;
                for (k, v) in m {
                    acc.entry(k).or_default().push(v);
                }
            }
            Err(e) => eprintln!("    sample {} failed: {e:#}", i + 1),
        }
    }
    ensure!(ok > 0, "all {samples} samples failed");
    Ok(acc.into_iter().map(|(k, v)| (k, median(&v))).collect())
}

/// Median of a non-empty slice (mean of the two middle values for an even count).
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
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
    let rows = crate::bench_expectations::compute(bench_data, thresholds, triple, device)?;
    eprintln!("expectations: {} gated metrics from {bench_data}/{triple}/{device}", rows.len());
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

/// Write metrics as `key value` lines to `path`, or as one JSON object per line to stdout
/// when `path` is `-`. The stdout form lets a run on a remote target return its metrics over
/// the captured stdout stream (stderr carries all progress), with no file to pull back.
fn write_metrics(path: &str, metrics: &[(String, f64)]) -> TractResult<()> {
    if path == "-" {
        let mut out = std::io::stdout().lock();
        for (k, v) in metrics {
            writeln!(out, "{}", serde_json::json!({ "metric": k, "value": v }))?;
        }
        return Ok(out.flush()?);
    }
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
    // Log the relative name only: the base URL may be a private mirror we keep out of logs.
    eprintln!("  fetching {name}");
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

tract_core::declare_knob!(
    TRACT_BENCH_GPU_CLOCK,
    Option<String>,
    None,
    "GPU graphics clock (MHz) to lock before benching; unset locks the top supported clock."
);

/// Pins the GPU graphics clock for the run and resets it on drop. The cuda runner
/// free-boosts from idle, adding session-to-session variance to evaltime that the
/// in-run retry can't damp. Best-effort: needs privilege, no-op without nvidia-smi.
/// `TRACT_BENCH_GPU_CLOCK` (MHz) overrides the default (the top supported clock).
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
        let clock = TRACT_BENCH_GPU_CLOCK.get().filter(|c| !c.is_empty()).or_else(query);
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

tract_core::declare_knob!(
    TRACT_BENCH_CPU_GOVERNOR,
    Option<String>,
    None,
    "CPU scaling governor to pin on every CPU before benching (e.g. `performance`); unset leaves the host untouched."
);

/// Opt-in CPU determinism pin, the governor twin of [`GpuClock`]. When `TRACT_BENCH_CPU_GOVERNOR`
/// names a governor (e.g. `performance`), set it on every CPU (best effort; needs privilege); for
/// `userspace`, also pin each CPU to its top available frequency. Unset/empty or non-Linux: no
/// change, so a runner that does not request it (the hosted boxes) is left untouched. A bench
/// target requests it out of band — via the dinghy device's `remote_shell_vars` — never in repo.
fn set_governor() {
    if !cfg!(target_os = "linux") {
        return;
    }
    let Some(governor) = TRACT_BENCH_CPU_GOVERNOR.get().filter(|g| !g.is_empty()) else {
        return;
    };
    let Ok(cpus) = std::fs::read_dir("/sys/devices/system/cpu") else { return };
    for cpu in cpus.flatten() {
        let base = cpu.path().join("cpufreq");
        if !base.join("scaling_governor").exists() {
            continue;
        }
        let _ = std::fs::write(base.join("scaling_governor"), &governor);
        if governor == "userspace" {
            if let Ok(freqs) = std::fs::read_to_string(base.join("scaling_available_frequencies")) {
                if let Some(max) =
                    freqs.split_whitespace().filter_map(|f| f.parse::<u64>().ok()).max()
                {
                    let _ = std::fs::write(base.join("scaling_setspeed"), max.to_string());
                }
            }
        }
    }
}
