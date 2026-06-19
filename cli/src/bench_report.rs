//! `tract bench-report`: render the PR-vs-main bench comparison. Fan-in step that
//! consumes every device's result (a dir per device under `--results`, each with
//! `meta.json` = {device, triple} and a `metrics` file) plus the bench-data checkout
//! (the nightly-main reference), and emits the PR-comment markdown (`--out`, movers
//! only, worst first) and, if `$GITHUB_STEP_SUMMARY` is set, the full per-device
//! table. Port of `.travis/bench-report.py`; the markdown layout lives in the
//! `bench-comment.md.j2` / `bench-report.md.j2` templates so it can be tuned without
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

/// A named sub-table within a device's full listing (Speed / Load / Memory / …).
#[derive(Serialize)]
struct Group {
    title: String,
    rows: Vec<Cell>,
}

#[derive(Serialize)]
struct DeviceReport {
    device: String,
    regr: usize,
    impr: usize,
    stable: usize,
    total: usize,
    groups: Vec<Group>,
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

/// Collapse a metric's variant to the column axis: cpu vs gpu. The variant is
/// overloaded — for `backends` benches it is the backend (`cpu`/`cuda`/`metal`),
/// for backend-less benches it is a config label (`pass`, `pulse8`, `2600ms`, …)
/// that always ran on CPU. Only cuda/metal/gpu count as gpu; everything else cpu.
fn backend_norm(variant: &str) -> &str {
    if variant.contains("cuda") || variant.contains("metal") || variant.contains("gpu") {
        "gpu"
    } else {
        "cpu"
    }
}

/// Worst glyph over a set of rows for one lamp: regression dominates, then
/// improvement, then stable; `–` when no row maps to the cell at all.
fn worst_glyph<'a>(rows: impl Iterator<Item = &'a Row>) -> &'static str {
    let (mut seen, mut impr) = (false, false);
    for r in rows {
        seen = true;
        if r.mover && r.worse {
            return "🔴";
        }
        impr |= r.mover;
    }
    if !seen {
        "–"
    } else if impr {
        "🟢"
    } else {
        "⚪"
    }
}

fn star_type(metric: &str) -> Option<&'static str> {
    if metric.contains("evaltime") {
        Some("evaltime")
    } else if metric.contains("pp512") {
        Some("pp")
    } else if metric.contains("tg128") {
        Some("tg")
    } else {
        None
    }
}

/// At-a-glance markdown: rows = models, columns = `device·{cpu,gpu}`. Each cell
/// holds two "lamps" showing the bottleneck axis — nets: batch·pulse (non-pulse
/// vs the worst of the pulse variants), llms: prefill·decode. A lamp is the worst
/// glyph over every variant feeding it; `–` when that side has no run. Empty if
/// nothing maps.
fn star_matrix_md(rows: &[Row]) -> String {
    struct P<'a> {
        row: &'a Row,
        dev: &'a str,
        model: String,
        is_llm: bool,
        backend: &'a str,
        ttype: &'static str,
        pulse: bool,
    }
    let parsed: Vec<P> = rows
        .iter()
        .filter_map(|r| {
            let kind = r.metric.split('.').next().unwrap_or("");
            let ttype = star_type(&r.metric)?;
            if kind != "net" && kind != "llm" {
                return None;
            }
            let variant = r.metric.rsplit('.').next().unwrap_or("");
            Some(P {
                row: r,
                dev: &r.device,
                model: describe(&r.metric).0,
                is_llm: kind == "llm",
                backend: backend_norm(variant),
                ttype,
                pulse: variant.contains("pulse"),
            })
        })
        .collect();
    if parsed.is_empty() {
        return String::new();
    }
    let mut cols: Vec<(&str, &str)> = parsed.iter().map(|p| (p.dev, p.backend)).collect();
    cols.sort_unstable();
    cols.dedup();
    let mut models: Vec<(bool, &str)> =
        parsed.iter().map(|p| (p.is_llm, p.model.as_str())).collect();
    models.sort_unstable();
    models.dedup();
    let lamp = |d: &str, b: &str, model: &str, pred: &dyn Fn(&P) -> bool| {
        worst_glyph(
            parsed
                .iter()
                .filter(|p| p.dev == d && p.backend == b && p.model == model && pred(p))
                .map(|p| p.row),
        )
    };

    let mut s = String::from("### Star metric — model × device\n\n| model |");
    for (d, b) in &cols {
        s.push_str(&format!(" {d}·{b} |"));
    }
    s.push_str("\n|---|");
    for _ in &cols {
        s.push_str(":--:|");
    }
    s.push('\n');
    for (is_llm, model) in &models {
        s.push_str(&format!("| {model} |"));
        for (d, b) in &cols {
            let (l, r) = if *is_llm {
                (lamp(d, b, model, &|p| p.ttype == "pp"), lamp(d, b, model, &|p| p.ttype == "tg"))
            } else {
                (lamp(d, b, model, &|p| !p.pulse), lamp(d, b, model, &|p| p.pulse))
            };
            s.push_str(&format!(" {l}{r} |"));
        }
        s.push('\n');
    }
    s.push_str(
        "\n_🟢 better · 🔴 worse · ⚪ within noise · – n/a · \
         net cell = batch·pulse · LLM cell = prefill·decode · lamp = worst of its variants_\n",
    );
    s
}

/// Coarse bucket for the full-report sub-tables, in display order.
const GROUP_ORDER: &[&str] = &["Speed", "Load", "Memory", "Other"];

fn group_of(metric: &str) -> &'static str {
    if is_speed(metric) {
        "Speed"
    } else if describe(metric).3 == "mem" {
        "Memory"
    } else if describe(metric).3 == "s" {
        "Load"
    } else {
        "Other"
    }
}

/// One [`DeviceReport`] per device: regr/impr/stable counts plus rows bucketed
/// into [`GROUP_ORDER`], each group movers-first then by largest move.
fn build_device_reports(rows: &[Row], devices: &[String]) -> Vec<DeviceReport> {
    devices
        .iter()
        .map(|dev| {
            let drows: Vec<&Row> = rows.iter().filter(|r| &r.device == dev).collect();
            let regr = drows.iter().filter(|r| r.mover && r.worse).count();
            let impr = drows.iter().filter(|r| r.mover && !r.worse).count();
            let mut groups = vec![];
            for title in GROUP_ORDER {
                let mut g: Vec<&Row> =
                    drows.iter().copied().filter(|r| group_of(&r.metric) == *title).collect();
                if g.is_empty() {
                    continue;
                }
                g.sort_by(|a, b| {
                    b.mover.cmp(&a.mover).then(b.delta.abs().total_cmp(&a.delta.abs()))
                });
                groups.push(Group {
                    title: (*title).to_string(),
                    rows: g.iter().map(|r| to_cell(r)).collect(),
                });
            }
            DeviceReport {
                device: dev.clone(),
                regr,
                impr,
                stable: drows.len() - regr - impr,
                total: drows.len(),
                groups,
            }
        })
        .collect()
}

fn render_report(
    templates: &str,
    ref_day: &str,
    pr_sha9: &str,
    star_matrix: &str,
    devices: &[DeviceReport],
) -> TractResult<String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    let tpl = std::fs::read_to_string(format!("{templates}/bench-report.md.j2"))?;
    env.add_template("report", &tpl)?;
    let rendered =
        env.get_template("report")?.render(context! { ref_day, pr_sha9, star_matrix, devices })?;
    Ok(tidy(&rendered))
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

    if let Ok(report_path) = std::env::var("GITHUB_STEP_SUMMARY") {
        let dev_reports = build_device_reports(&rows, &uniq);
        let rendered = render_report(
            templates,
            &ref_day,
            &pr_sha[..pr_sha.len().min(9)],
            &star_matrix_md(&rows),
            &dev_reports,
        )?;
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(report_path)?
            .write_all(rendered.as_bytes())?;
    }

    println!("regressions={} improvements={} metrics={}", regr.len(), impr.len(), rows.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(device: &str, metric: &str, refv: f64, prv: f64, worse: bool, mover: bool) -> Row {
        Row {
            device: device.into(),
            metric: metric.into(),
            refv,
            prv,
            delta: (prv - refv) / refv * 100.0,
            worse,
            mover,
        }
    }

    #[test]
    fn report_groups_counts_and_legend() {
        let rows = vec![
            row("gpu", "net.bert.evaltime", 100.0, 108.0, true, true),
            row("gpu", "llm.qwen.tg128.q40", 50.0, 47.0, true, true),
            row("gpu", "llm.qwen.pp512.q40", 800.0, 815.0, false, false),
            row("gpu", "net.bert.rsz_at_model_ready", 1.0e9, 1.02e9, true, false),
            row("gpu", "net.bert.time_to_model_ready", 2.0, 1.5, false, true),
        ];
        let devices = vec!["gpu".to_string()];
        let ds = build_device_reports(&rows, &devices);
        assert_eq!(ds[0].regr, 2);
        assert_eq!(ds[0].impr, 1);
        assert_eq!(ds[0].total, 5);
        assert_eq!(ds[0].stable, 2);

        let md =
            render_report("../.travis", "2026-06-19", "abcdef123", &star_matrix_md(&rows), &ds)
                .unwrap();
        assert!(md.contains("### Speed"), "missing Speed group:\n{md}");
        assert!(md.contains("### Load"), "missing Load group:\n{md}");
        assert!(md.contains("### Memory"), "missing Memory group:\n{md}");
        assert!(md.contains("| device |"), "missing overview table");
        assert!(md.contains("lower is better"), "missing legend");
    }

    #[test]
    fn star_matrix_lamps() {
        let rows = vec![
            row("x64", "net.en_tdnn_8M.evaltime.2600ms", 100.0, 90.0, false, true),
            row("x64", "net.en_tdnn_8M.evaltime.pulse_240ms", 50.0, 56.0, true, true),
            row("x64", "net.trunet.evaltime.pulse1_f16", 10.0, 10.05, false, false),
            row("x64", "llm.qwen.pp512.cuda", 800.0, 790.0, true, true),
            row("x64", "llm.qwen.tg128.cuda", 50.0, 55.0, false, true),
        ];
        let md = star_matrix_md(&rows);
        assert!(md.contains("x64·cpu"), "missing cpu column:\n{md}");
        assert!(md.contains("x64·gpu"), "missing gpu column:\n{md}");
        // net: batch·pulse; cpu has both variants, gpu has none.
        assert!(md.contains("| en_tdnn_8M | 🟢🔴 | –– |"), "en_tdnn lamps:\n{md}");
        // pulse-only net: batch lamp absent.
        assert!(md.contains("| trunet | –⚪ | –– |"), "trunet lamps:\n{md}");
        // llm: prefill·decode on gpu, nothing on cpu.
        assert!(md.contains("| qwen | –– | 🔴🟢 |"), "qwen lamps:\n{md}");
    }
}
