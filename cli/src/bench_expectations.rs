//! `tract bench-expectations`: emit per-metric `metric expected threshold` lines for
//! one (triple, device), so the suite retries exactly the benches whose value would
//! show as a PR red. Port of `.travis/bench-expectations.py`.
//!
//! `expected` is the recent median of the non-null points; `threshold` is the |Δ%|
//! that would make the metric a red (`bench_common::red_threshold`). Never-gated
//! metrics are omitted (not retried). A device with no history yields an empty file
//! (retry disabled there, single-shot).

use crate::bench_common::{Thresholds, red_threshold, reference_value, series_noise};
use serde::Deserialize;
use std::collections::BTreeMap;
use tract_hir::internal::*;

#[derive(Deserialize)]
struct BenchData {
    #[serde(default)]
    metrics: BTreeMap<String, Vec<Option<f64>>>,
}

/// Compute the gated `(metric, expected, threshold)` rows for one (triple, device)
/// from bench-data history. Shared by the `bench-expectations` subcommand (which
/// writes them to a file) and the orchestrator (which computes them inline on the
/// bench host). A device with no history yields no rows (retry disabled there).
pub fn compute(
    bench_data: &str,
    thresholds: &str,
    triple: &str,
    device: &str,
    window: usize,
) -> TractResult<Vec<(String, f64, f64)>> {
    let cfg = Thresholds::load(thresholds)?;
    let path = format!("{bench_data}/{triple}/{device}.json");
    let mut rows = vec![];
    if std::path::Path::new(&path).exists() {
        let data: BenchData = serde_json::from_str(&std::fs::read_to_string(&path)?)
            .with_context(|| format!("parsing {path}"))?;
        for (metric, arr) in &data.metrics {
            let Some(expected) = reference_value(arr, window) else { continue };
            if let Some(thr) = red_threshold(metric, &cfg, series_noise(arr, 40, 8), Some(expected))
            {
                rows.push((metric.clone(), expected, thr));
            }
        }
    }
    Ok(rows)
}

pub fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    let get = |k| matches.get_one::<String>(k).map(String::as_str);
    let bench_data = get("bench-data").context("--bench-data is required")?;
    let thresholds = get("thresholds").context("--thresholds is required")?;
    let triple = get("triple").context("--triple is required")?;
    let device = get("device").context("--device is required")?;
    let out = get("out").context("--out is required")?;
    let window: usize = get("window").map(str::parse).transpose()?.unwrap_or(10);

    let rows = compute(bench_data, thresholds, triple, device, window)?;
    let body: String = rows.iter().map(|(m, e, t)| format!("{m} {e} {t}\n")).collect();
    std::fs::write(out, body)?;
    println!("expectations: {} gated metrics -> {out}", rows.len());
    Ok(())
}
