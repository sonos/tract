//! Shared bench-comparison math: the one place that answers "what move makes this
//! metric a PR red?". Used by `bench-expectations` (so the suite retries exactly the
//! would-be reds) and, later, the report. Port of `.travis/bench_common.py`.

use serde::Deserialize;
use tract_hir::internal::*;

/// Noise guards for the PR-vs-main comparison (`.travis/bench-thresholds.toml`).
#[derive(Deserialize)]
pub struct Thresholds {
    #[serde(default = "default_k")]
    pub k: f64,
    #[serde(default)]
    pub min_load_seconds: f64,
    #[serde(default)]
    pub needs_history: Vec<String>,
    #[serde(default)]
    pub ignore: Vec<String>,
    /// Per-class percent floors, in priority order (first substring match wins).
    pub floors: toml::Table,
}

fn default_k() -> f64 {
    3.0
}

impl Thresholds {
    pub fn load(path: &str) -> TractResult<Thresholds> {
        toml::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?)
            .with_context(|| format!("parsing {path}"))
    }
}

/// LLM throughput metrics (`.ppN.` / `.tgN.`) are higher-is-better; everything else
/// is lower-is-better.
pub fn higher_better(metric: &str) -> bool {
    metric.split('.').any(|seg| {
        (seg.starts_with("pp") || seg.starts_with("tg"))
            && seg.len() > 2
            && seg[2..].bytes().all(|c| c.is_ascii_digit())
    })
}

/// The inference-speed signal (evaltime / prefill / decode), shown first in the report.
pub fn is_speed(metric: &str) -> bool {
    metric.split('.').any(|seg| seg == "evaltime") || higher_better(metric)
}

/// Read a `metric value` file in file order, canonicalizing '-' -> '_' (the old minion
/// pushed underscored names to graphite, and bench-data follows that). File order is
/// kept so equal-Δ ties in the report sort the same way the values were emitted.
pub fn read_metrics(path: &str) -> Vec<(String, f64)> {
    let mut out = vec![];
    let Ok(content) = std::fs::read_to_string(path) else { return out };
    for line in content.lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() >= 2 {
            if let Ok(v) = p[1].parse::<f64>() {
                out.push((p[0].replace('-', "_"), v));
            }
        }
    }
    out
}

fn floor_for(metric: &str, floors: &toml::Table) -> f64 {
    for (cls, t) in floors {
        if cls != "default" && metric.contains(cls.as_str()) {
            return as_f64(t);
        }
    }
    floors.get("default").map(as_f64).unwrap_or(5.0)
}

fn as_f64(v: &toml::Value) -> f64 {
    v.as_float().or_else(|| v.as_integer().map(|i| i as f64)).unwrap_or(0.0)
}

/// p90 of the recent day-to-day |Δ%| of a metric series — its intrinsic run-to-run
/// dispersion. p90 (not max) so a stray real-change spike doesn't inflate it; `None`
/// when there isn't enough history to judge.
pub fn series_noise(arr: &[Option<f64>], window: usize, min_pairs: usize) -> Option<f64> {
    let mut d = vec![];
    let mut prev: Option<f64> = None;
    for &v in &arr[arr.len().saturating_sub(window)..] {
        match v {
            None => prev = None,
            Some(v) => {
                if let Some(p) = prev {
                    if p != 0.0 {
                        d.push((v - p).abs() / p.abs() * 100.0);
                    }
                }
                prev = Some(v);
            }
        }
    }
    if d.len() < min_pairs {
        return None;
    }
    d.sort_by(|a, b| a.total_cmp(b));
    Some(d[((0.9 * d.len() as f64) as usize).min(d.len() - 1)])
}

/// The |Δ%| that makes this metric a PR red, or `None` if it is never gated:
/// operational/ignored, a sub-resolution load, or a noisy class lacking the history
/// to estimate its noise. Otherwise `max(class floor, k * series noise)`.
pub fn red_threshold(
    metric: &str,
    cfg: &Thresholds,
    noise: Option<f64>,
    value: Option<f64>,
) -> Option<f64> {
    if cfg.ignore.iter().any(|c| metric.contains(c.as_str())) {
        return None;
    }
    if metric.contains("time_to") && value.is_some_and(|v| v < cfg.min_load_seconds) {
        return None;
    }
    if noise.is_none() && cfg.needs_history.iter().any(|c| metric.contains(c.as_str())) {
        return None;
    }
    let floor = floor_for(metric, &cfg.floors);
    Some(noise.map_or(floor, |n| floor.max(cfg.k * n)))
}

/// The baseline a metric is compared against: the median of its recent non-null
/// points. Used by BOTH the report (to compute the red) and bench-expectations (the
/// value shipped to the retry), so the retry and the report's red judge against the
/// same number. Median (not the latest single value) so a noisy last nightly doesn't
/// shift the reference. `None` if there's no data.
pub fn reference_value(arr: &[Option<f64>], window: usize) -> Option<f64> {
    let vals: Vec<f64> =
        arr[arr.len().saturating_sub(window)..].iter().filter_map(|&v| v).collect();
    (!vals.is_empty()).then(|| median(&vals))
}

/// Median of a non-empty slice (mean of the two middle points for even length),
/// matching Python's `statistics.median`.
pub fn median(vals: &[f64]) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Thresholds {
        toml::from_str(
            r#"
k = 3.0
min_load_seconds = 0.03
needs_history = ["time_to", "rsz"]
ignore = ["bench_runtime"]
[floors]
default = 5
evaltime = 5
active_at = 2
time_to = 5
rsz = 5
pp = 3
tg = 3
"#,
        )
        .unwrap()
    }

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn reference_value_is_windowed_median_skipping_nulls() {
        let arr = [Some(1.0), None, Some(2.0), Some(100.0)];
        assert_eq!(reference_value(&arr, 3), Some(51.0)); // median of {2,100} over last 3 (one null)
        assert_eq!(reference_value(&[None, None], 10), None);
    }

    #[test]
    fn direction_and_speed_signals() {
        assert!(higher_better("llm.m.pp512.cpu"));
        assert!(higher_better("llm.m.tg128.cpu"));
        assert!(!higher_better("net.m.evaltime.pass"));
        assert!(is_speed("net.m.evaltime.pass"));
        assert!(is_speed("llm.m.pp512.cpu"));
        assert!(!is_speed("net.m.rsz_at_model_ready.pass"));
    }

    #[test]
    fn red_threshold_gating() {
        let c = cfg();
        // operational metric: never gated
        assert_eq!(red_threshold("bundle.bench_runtime", &c, Some(1.0), Some(1.0)), None);
        // sub-resolution load: not gated
        assert_eq!(
            red_threshold("net.m.time_to_before_optimize.v", &c, Some(1.0), Some(0.01)),
            None
        );
        // needs_history class with no measured noise: not gated
        assert_eq!(red_threshold("net.m.rsz_at_model_ready.v", &c, None, Some(1e8)), None);
        // clean class without history still gates at its floor
        assert_eq!(red_threshold("net.m.evaltime.v", &c, None, Some(0.05)), Some(5.0));
        assert_eq!(red_threshold("net.m.active_at_model_ready.v", &c, None, Some(2e7)), Some(2.0));
        // with noise: max(floor, k * noise)
        assert_eq!(red_threshold("net.m.evaltime.v", &c, Some(4.0), Some(0.05)), Some(12.0));
    }
}
