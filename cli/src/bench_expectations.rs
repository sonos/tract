//! `tract bench-expectations`: emit per-metric `metric expected threshold` lines for
//! one (triple, device), so the suite retries exactly the benches whose value would
//! show as a PR red. Port of `.travis/bench-expectations.py`.
//!
//! `expected` is the recent median of the non-null points; `threshold` is the |Δ%|
//! that would make the metric a red (`bench_common::red_threshold`). Never-gated
//! metrics are omitted (not retried). A device with no history yields an empty file
//! (retry disabled there, single-shot).

use crate::bench_common::{Thresholds, median, red_threshold, series_noise};
use serde::Deserialize;
use std::collections::BTreeMap;
use tract_hir::internal::*;

#[derive(Deserialize)]
struct BenchData {
    #[serde(default)]
    metrics: BTreeMap<String, Vec<Option<f64>>>,
}

pub fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    let get = |k| matches.get_one::<String>(k).map(String::as_str);
    let bench_data = get("bench-data").context("--bench-data is required")?;
    let thresholds = get("thresholds").context("--thresholds is required")?;
    let triple = get("triple").context("--triple is required")?;
    let device = get("device").context("--device is required")?;
    let out = get("out").context("--out is required")?;
    let window: usize = get("window").map(str::parse).transpose()?.unwrap_or(10);

    let cfg = Thresholds::load(thresholds)?;
    let path = format!("{bench_data}/{triple}/{device}.json");

    let mut lines = vec![];
    if std::path::Path::new(&path).exists() {
        let data: BenchData = serde_json::from_str(&std::fs::read_to_string(&path)?)
            .with_context(|| format!("parsing {path}"))?;
        for (metric, arr) in &data.metrics {
            let vals: Vec<f64> =
                arr[arr.len().saturating_sub(window)..].iter().filter_map(|&v| v).collect();
            if vals.is_empty() {
                continue;
            }
            let expected = median(&vals);
            if let Some(thr) = red_threshold(metric, &cfg, series_noise(arr, 40, 8), Some(expected))
            {
                lines.push(format!("{metric} {expected} {thr}"));
            }
        }
    }

    std::fs::write(out, lines.iter().map(|l| format!("{l}\n")).collect::<String>())?;
    println!("expectations: {} gated metrics -> {out}", lines.len());
    Ok(())
}
