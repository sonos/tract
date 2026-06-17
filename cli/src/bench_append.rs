//! `tract bench-append`: append one nightly run to the columnar bench-data branch.
//! Port of `.travis/bench-append.py`.
//!
//! File `<out>/<triple>/<device>.json` = {"start_day":"YYYY-MM-DD","metrics":{"<m>":[v0,v1,...]}}.
//! Column i of every array is start_day + i days; null = no run that day. Values are
//! rounded to 4 significant figures and written one per line (JSON with no indent) so a
//! daily append is an insertion into each array and the git diff stays small.

use crate::bench_common::read_metrics;
use serde::Deserialize;
use std::collections::BTreeMap;
use time::{Date, Month, OffsetDateTime};
use tract_hir::internal::*;

#[derive(Deserialize)]
struct BenchData {
    start_day: String,
    #[serde(default)]
    metrics: BTreeMap<String, Vec<Option<f64>>>,
}

fn parse_date(s: &str) -> TractResult<Date> {
    let p: Vec<&str> = s.split('-').collect();
    ensure!(p.len() == 3, "bad date {s:?}");
    Ok(Date::from_calendar_date(
        p[0].parse()?,
        Month::try_from(p[1].parse::<u8>()?)?,
        p[2].parse()?,
    )?)
}

/// Round to 4 significant figures, half-to-even (matching Python's `round`). Going
/// through `{:e}` formatting both rounds correctly and avoids the scaling error of
/// multiply/divide by a power of ten.
fn sig(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    format!("{x:.3e}").parse().unwrap_or(x)
}

/// JSON value as bench-data writes it: `null`, an integer with no decimal point, or a
/// shortest-round-trip float (Rust's `f64` Display matches Python's repr for these).
fn fmt_value(v: Option<f64>) -> String {
    v.map_or_else(|| "null".to_string(), |x| x.to_string())
}

pub fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    let get = |k| matches.get_one::<String>(k).map(String::as_str);
    let metrics_path = get("metrics").context("--metrics is required")?;
    let out = get("out").context("--out is required")?;
    let triple = get("triple").context("--triple is required")?;
    let device = get("device").context("--device is required")?;
    let day = match get("day") {
        Some(s) => parse_date(s)?,
        None => OffsetDateTime::now_utc().date(),
    };

    let today: BTreeMap<String, f64> =
        read_metrics(metrics_path).into_iter().map(|(k, v)| (k, sig(v))).collect();
    ensure!(!today.is_empty(), "no metrics in {metrics_path}");

    let path = format!("{out}/{triple}/{device}.json");
    let (start, mut arrays) = if std::path::Path::new(&path).exists() {
        let d: BenchData = serde_json::from_str(&std::fs::read_to_string(&path)?)
            .with_context(|| format!("parsing {path}"))?;
        (parse_date(&d.start_day)?, d.metrics)
    } else {
        (day, BTreeMap::new())
    };

    let idx = (day - start).whole_days();
    ensure!(idx >= 0, "day {day} precedes start_day {start}");
    let n = idx as usize + 1;

    let keys: std::collections::BTreeSet<&String> = arrays.keys().chain(today.keys()).collect();
    for k in keys.into_iter().cloned().collect::<Vec<_>>() {
        let a = arrays.entry(k.clone()).or_default();
        if a.len() < n {
            a.resize(n, None); // null-pad skipped days / a new metric's prefix
        }
        a[idx as usize] = today.get(&k).copied(); // value today, else null
    }

    let mut s = format!("{{\n\"start_day\": \"{start}\",\n\"metrics\": {{");
    for (i, (name, arr)) in arrays.iter().enumerate() {
        s += &format!("\n\"{name}\": [");
        for (j, v) in arr.iter().enumerate() {
            s += &format!("\n{}", fmt_value(*v));
            if j + 1 < arr.len() {
                s += ",";
            }
        }
        s += "\n]";
        if i + 1 < arrays.len() {
            s += ",";
        }
    }
    s += "\n}\n}\n";

    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, s)?;
    println!("appended {device} {day}: {} metrics, span now {n} days", today.len());
    Ok(())
}
