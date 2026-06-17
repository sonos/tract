//! `tract bench-report`: render the PR-vs-main bench comparison. Fan-in step that
//! consumes every device's result (a dir per device under `--results`, each with
//! `meta.json` = {device, triple} and a `metrics` file) plus the bench-data checkout
//! (the nightly-main reference), and emits the PR-comment markdown (`--out`, movers
//! only, worst first) and, if `$GITHUB_STEP_SUMMARY` is set, the full per-device
//! table. Port of `.travis/bench-report.py`; the markdown layout lives in the
//! `bench-comment.md.j2` / `bench-summary.md.j2` templates so it can be tuned without
//! a rebuild. Single-shot vs the reference; |Δ| must reach the adaptive threshold
//! (`bench_common::red_threshold`) to count as a mover. Direction-aware.

use crate::bench_common::{
    Thresholds, higher_better, is_speed, read_metrics, red_threshold, reference_value, series_noise,
};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use time::{Date, Duration, Month, OffsetDateTime};
use tract_hir::internal::*;

#[derive(Deserialize)]
struct Meta {
    device: String,
    triple: String,
}

#[derive(Deserialize)]
struct RefData {
    start_day: String,
    #[serde(default)]
    metrics: BTreeMap<String, Vec<Option<f64>>>,
}

/// A single PR-vs-reference comparison for one metric on one device.
struct Row {
    device: String,
    metric: String,
    refv: f64,
    prv: f64,
    delta: f64,
    worse: bool,
    mover: bool,
}

/// A table row as the templates consume it (all formatting done here).
#[derive(Serialize)]
struct Cell {
    icon: String,
    flag: String,
    delta: String,
    cell: String,
    device: String,
    ref_fmt: String,
    pr_fmt: String,
}

#[derive(Serialize)]
struct DeviceSummary {
    device: String,
    count: usize,
    rows: Vec<Cell>,
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

/// Per-metric latest value, per-metric noise, and the date of the most recent point.
type Reference = (BTreeMap<String, f64>, BTreeMap<String, Option<f64>>, Option<Date>);

/// Latest non-null value + recent noise per metric from bench-data/<triple>/<device>.json,
/// plus the date of the most recent reference point.
fn reference(bench_data: &str, triple: &str, device: &str) -> TractResult<Reference> {
    let path = format!("{bench_data}/{triple}/{device}.json");
    if !std::path::Path::new(&path).exists() {
        return Ok((BTreeMap::new(), BTreeMap::new(), None));
    }
    let d: RefData = serde_json::from_str(&std::fs::read_to_string(&path)?)
        .with_context(|| format!("parsing {path}"))?;
    let start = parse_date(&d.start_day)?;
    let (mut vals, mut noise) = (BTreeMap::new(), BTreeMap::new());
    let mut last_idx: i64 = -1;
    for (m, arr) in &d.metrics {
        noise.insert(m.clone(), series_noise(arr, 40, 8));
        if let Some(v) = reference_value(arr, 10) {
            vals.insert(m.clone(), v); // median baseline, == what bench-expectations ships
        }
        if let Some(i) = arr.iter().rposition(Option::is_some) {
            last_idx = last_idx.max(i as i64); // latest non-null day, for ref-date display only
        }
    }
    let ref_day = (last_idx >= 0).then(|| start + Duration::days(last_idx));
    Ok((vals, noise, ref_day))
}

// metric type-token -> (human label, unit); first substring match wins.
const TYPE_INFO: &[(&str, &str, &str)] = &[
    ("evaltime", "evaltime", "s"),
    ("time_to_model_ready", "load+optimize", "s"),
    ("time_to_before_optimize", "load", "s"),
    ("rsz_at_model_ready", "RSS @ ready", "mem"),
    ("rsz_at_before_optimize", "RSS @ load", "mem"),
    ("active_at_model_ready", "heap @ ready", "mem"),
    ("active_at_before_optimize", "heap @ load", "mem"),
    ("pp512", "prefill", "tok"),
    ("tg128", "decode", "tok"),
    ("bench_runtime", "bench wall", "s"),
    ("binary_size", "binary size", "mem"),
];

/// (model, label, variant, unit) for a metric key `kind.model.type.variant`.
fn describe(metric: &str) -> (String, String, String, &'static str) {
    let (mut label, mut unit) = (None, "raw");
    for (key, lbl, u) in TYPE_INFO {
        if metric.contains(key) {
            label = Some(*lbl);
            unit = u;
            break;
        }
    }
    let p: Vec<&str> = metric.split('.').collect();
    let nth = |i: usize| p.get(i).copied().unwrap_or(metric);
    if p[0] == "net" || p[0] == "llm" {
        let label = label.map(str::to_string).unwrap_or_else(|| nth(2).to_string());
        let variant = if p.len() >= 4 { p[p.len() - 1] } else { "" };
        (nth(1).to_string(), label, variant.to_string(), unit)
    } else {
        let label = label.map(str::to_string).unwrap_or_else(|| nth(1).to_string());
        (p[0].to_string(), label, String::new(), unit)
    }
}

/// Format like C `printf %g` with `p` significant figures (trailing zeros stripped).
fn fmt_g(x: f64, p: usize) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let p = p.max(1) as i32;
    let e = x.abs().log10().floor() as i32;
    if e < -4 || e >= p {
        return format!("{:.*e}", (p - 1).max(0) as usize, x);
    }
    let s = format!("{:.*}", (p - 1 - e).max(0) as usize, x);
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

fn fmt_val(v: f64, unit: &str) -> String {
    match unit {
        "s" if v < 1.0 => format!("{} ms", fmt_g(v * 1000.0, 3)),
        "s" => format!("{} s", fmt_g(v, 3)),
        "mem" if v >= 1e9 => format!("{} GB", fmt_g(v / 1e9, 3)),
        "mem" if v >= 1e6 => format!("{} MB", fmt_g(v / 1e6, 3)),
        "mem" => format!("{} kB", fmt_g(v / 1e3, 3)),
        "tok" => format!("{} tok/s", fmt_g(v, 4)),
        _ => fmt_g(v, 4),
    }
}

/// Two-line table cell: model on top, `label · variant` below in small text.
fn cell_md(metric: &str) -> String {
    let (model, label, variant, _) = describe(metric);
    let sub = if variant.is_empty() { label } else { format!("{label} · {variant}") };
    format!("{model}<br><sub>{sub}</sub>")
}

/// Normalize rendered markdown so the template can be edited without pixel-perfect
/// whitespace control: drop leading blank lines, collapse blank-line runs to a single
/// blank, and end with exactly one newline.
fn tidy(s: &str) -> String {
    let mut out = String::new();
    let mut pending_blank = false;
    for line in s.lines() {
        if line.trim().is_empty() {
            pending_blank = !out.is_empty();
        } else {
            if pending_blank {
                out.push('\n');
            }
            pending_blank = false;
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn to_cell(r: &Row) -> Cell {
    let unit = describe(&r.metric).3;
    Cell {
        icon: if r.worse { "🔴" } else { "🟢" }.to_string(),
        flag: if r.mover && r.worse {
            "🔴"
        } else if r.mover {
            "🟢"
        } else {
            ""
        }
        .to_string(),
        delta: format!("{:+.1}%", r.delta),
        cell: cell_md(&r.metric),
        device: r.device.clone(),
        ref_fmt: fmt_val(r.refv, unit),
        pr_fmt: fmt_val(r.prv, unit),
    }
}

pub fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    let get = |k| matches.get_one::<String>(k).map(String::as_str);
    let results = get("results").context("--results is required")?;
    let bench_data = get("bench-data").context("--bench-data is required")?;
    let pr_sha = get("pr-sha").context("--pr-sha is required")?;
    let out = get("out").context("--out is required")?;
    let templates = get("templates").unwrap_or(".travis");
    let cfg = Thresholds::load(get("thresholds").context("--thresholds is required")?)?;
    let today = match get("today") {
        Some(s) => parse_date(s)?,
        None => OffsetDateTime::now_utc().date(),
    };
    let run_url = std::env::var("RUN_URL").unwrap_or_else(|_| "#".to_string());

    let mut result_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(results)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.json").exists())
        .collect();
    result_dirs.sort();

    let mut rows = vec![];
    let mut devices = vec![];
    let mut ref_days = vec![];
    for dir in &result_dirs {
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
        devices.push(meta.device.clone());
        let pr = read_metrics(dir.join("metrics").to_str().unwrap_or_default());
        let (refv, noise, ref_day) = reference(bench_data, &meta.triple, &meta.device)?;
        if let Some(d) = ref_day {
            ref_days.push(d);
        }
        for (metric, prv) in &pr {
            let (metric, prv) = (metric, *prv);
            let Some(&rv) = refv.get(metric) else { continue };
            if rv == 0.0 {
                continue;
            }
            let delta = (prv - rv) / rv * 100.0;
            let worse = if higher_better(metric) { delta < 0.0 } else { delta > 0.0 };
            let thr = red_threshold(metric, &cfg, noise.get(metric).copied().flatten(), Some(rv));
            let mover = thr.is_some_and(|t| delta.abs() >= t);
            rows.push(Row {
                device: meta.device.clone(),
                metric: metric.clone(),
                refv: rv,
                prv,
                delta,
                worse,
                mover,
            });
        }
    }

    // No comparable metrics (no device results, or no reference) -> write nothing, so a
    // cancelled/empty run can't overwrite a real comment with a vacuous "no regressions".
    if rows.is_empty() {
        println!("no comparable metrics; not writing a comment");
        return Ok(());
    }

    let movers: Vec<&Row> = rows.iter().filter(|r| r.mover).collect();
    let by_delta = |a: &&Row, b: &&Row| b.delta.abs().total_cmp(&a.delta.abs());

    let mut regr: Vec<&Row> = movers.iter().copied().filter(|r| r.worse).collect();
    regr.sort_by(by_delta);
    let mut impr: Vec<&Row> = movers.iter().copied().filter(|r| !r.worse).collect();
    impr.sort_by(|a, b| is_speed(&b.metric).cmp(&is_speed(&a.metric)).then_with(|| by_delta(a, b)));

    let speed_regr: Vec<Cell> =
        regr.iter().filter(|r| is_speed(&r.metric)).map(|r| to_cell(r)).collect();
    let other_regr: Vec<Cell> =
        regr.iter().filter(|r| !is_speed(&r.metric)).map(|r| to_cell(r)).collect();
    let impr_cells: Vec<Cell> = impr.iter().map(|r| to_cell(r)).collect();

    let head = if !speed_regr.is_empty() {
        let mut h = format!("🔴 **Bench vs main — {} speed regression(s)**", speed_regr.len());
        if !other_regr.is_empty() {
            h += &format!(" · {} load/memory", other_regr.len());
        }
        h
    } else if !other_regr.is_empty() {
        format!(
            "🟡 **Bench vs main — no speed regressions** · {} load/memory mover(s)",
            other_regr.len()
        )
    } else {
        "✅ **Bench vs main — no regressions**".to_string()
    };

    let latest = ref_days.iter().max().copied();
    let ref_day = latest.map_or_else(|| "n/a".to_string(), |d| d.to_string());
    let age = latest.map_or_else(|| "?".to_string(), |d| (today - d).whole_days().to_string());
    let mut uniq = devices.clone();
    uniq.sort();
    uniq.dedup();
    let devs = uniq.iter().map(|d| format!("`{d}`")).collect::<Vec<_>>().join(", ");

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    let comment = std::fs::read_to_string(format!("{templates}/bench-comment.md.j2"))?;
    env.add_template("comment", &comment)?;
    let rendered = env.get_template("comment")?.render(context! {
        head, ref_day, age, pr_sha9 => &pr_sha[..pr_sha.len().min(9)], devs,
        n_metrics => rows.len(), speed_regr, other_regr, impr => impr_cells, run_url,
    })?;
    std::fs::write(out, tidy(&rendered))?;

    if let Ok(summary_path) = std::env::var("GITHUB_STEP_SUMMARY") {
        let mut dev_summaries = vec![];
        for dev in &uniq {
            let mut drows: Vec<&Row> = rows.iter().filter(|r| &r.device == dev).collect();
            drows.sort_by(by_delta);
            dev_summaries.push(DeviceSummary {
                device: dev.clone(),
                count: drows.len(),
                rows: drows.iter().map(|r| to_cell(r)).collect(),
            });
        }
        let summary = std::fs::read_to_string(format!("{templates}/bench-summary.md.j2"))?;
        env.add_template("summary", &summary)?;
        let rendered = env.get_template("summary")?.render(context! {
            ref_day, pr_sha9 => &pr_sha[..pr_sha.len().min(9)], devices => dev_summaries,
        })?;
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(summary_path)?
            .write_all(tidy(&rendered).as_bytes())?;
    }

    println!("regressions={} improvements={} metrics={}", regr.len(), impr.len(), rows.len());
    Ok(())
}
