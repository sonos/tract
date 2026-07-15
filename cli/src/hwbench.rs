use std::io::IsTerminal;

use clap::{Args, FromArgMatches};
use nu_ansi_term::Color::*;
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_libcli::terminal::si_prefix;
use tract_linalg::hwbench::bandwidth::{l1_bandwidth_seq, main_memory_bandwith_seq};
use tract_linalg::hwbench::runner::run_bench;
use tract_linalg::mmm::{AsInputValue, FusedSpec, ImplementationQuality};

#[derive(serde::Serialize)]
struct Bandwidth {
    threads: usize,
    bytes_per_s: f64,
}

#[derive(serde::Serialize)]
struct KernelResult {
    kernel: String,
    packing: usize,
    layout: String,
    flop_per_s: f64,
    picked: bool,
}

/// One benched shape: every candidate kernel with its measured throughput,
/// sorted fastest-first. `picked` flags the one the live dispatcher selects.
#[derive(serde::Serialize)]
struct ShapeResult {
    m: usize,
    k: usize,
    n: usize,
    dt: String,
    /// Whether `--assert` gates on this shape. Diagnostic shapes (known-hard for
    /// every arch's picker) are benched and reported but not gated.
    gated: bool,
    kernels: Vec<KernelResult>,
}

impl ShapeResult {
    fn title(&self) -> String {
        format!("{}x{}x{}x{}", self.m, self.k, self.n, self.dt)
    }
    fn best(&self) -> Option<&KernelResult> {
        self.kernels.first()
    }
    fn picked(&self) -> Option<&KernelResult> {
        self.kernels.iter().find(|k| k.picked)
    }
    /// Picked kernel throughput as a fraction of the fastest measured kernel
    /// (1.0 = the dispatcher picked the best). `None` if nothing was picked.
    fn pick_ratio(&self) -> Option<f64> {
        let best = self.best()?.flop_per_s;
        let picked = self.picked()?.flop_per_s;
        Some(if best > 0.0 { picked / best } else { 1.0 })
    }
}

#[derive(serde::Serialize, Default)]
struct Report {
    #[serde(skip_serializing_if = "Option::is_none")]
    cache: Option<Vec<Bandwidth>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    main_memory: Option<Vec<Bandwidth>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    matmul: Vec<ShapeResult>,
}

#[derive(Args, Debug)]
pub(crate) struct HwbenchParams {
    /// Matmul shapes as M,K,N[,dt] (dt f32|f16, default both); repeatable. Omit for the default battery.
    #[arg(value_name = "SHAPE")]
    shapes: Vec<String>,
    /// Skip the cache (L1) bandwidth probe
    #[arg(long)]
    no_cache: bool,
    /// Skip the main-memory bandwidth probe
    #[arg(long)]
    no_memory: bool,
    /// Skip the matmul kernel benchmarks
    #[arg(long)]
    no_matmul: bool,
    /// Emit results as JSON on stdout
    #[arg(long)]
    json: bool,
    /// Exit nonzero if a picked kernel lags the fastest by more than --tolerance
    #[arg(long)]
    assert: bool,
    /// Tolerance percent a pick may lag the fastest kernel under --assert
    #[arg(long, default_value_t = 5.0)]
    tolerance: f64,
}

pub(crate) fn command() -> clap::Command {
    HwbenchParams::augment_args(
        clap::Command::new("hwbench").about("Report hardware metrics and matmul kernel throughput"),
    )
}

fn parse_dt(s: &str) -> TractResult<DatumType> {
    match s.to_lowercase().as_str() {
        "f32" => Ok(f32::datum_type()),
        "f16" => Ok(f16::datum_type()),
        _ => bail!("unknown dt {s:?} in shape (want f32 or f16)"),
    }
}

/// Expand `M,K,N` (both f32 and f16) or `M,K,N,dt` into concrete bench requests.
fn parse_shape(spec: &str) -> TractResult<Vec<(DatumType, usize, usize, usize, bool)>> {
    let f = spec.split(',').collect_vec();
    let dts = match f.len() {
        3 => vec![f32::datum_type(), f16::datum_type()],
        4 => vec![parse_dt(f[3])?],
        _ => bail!("shape must be M,K,N or M,K,N,dt (got {spec:?})"),
    };
    let m = f[0].parse()?;
    let k = f[1].parse()?;
    let n = f[2].parse()?;
    // Explicit shapes are always gated — the user asked for them.
    Ok(dts.into_iter().map(|dt| (dt, m, k, n, true)).collect())
}

/// Curated battery run when no explicit shape is given: square, matvec,
/// im2col-conv (large-N / small-K, where the picker most often mis-selects),
/// and the M-padding cases the picker was built to handle. Both f32 and f16.
fn default_battery() -> Vec<(DatumType, usize, usize, usize, bool)> {
    // (m, k, n, gate). gate=false = diagnostic: still benched and reported, but not
    // gated by --assert. Reserved for shapes every arch's picker mis-selects and a
    // static scale/quality model can't nail (tiny-K conv crossover, M-padding). Kept
    // as a tracked known-issue list; flip to true as the picker learns them.
    let mut shapes = vec![
        (512, 512, 512, false), // square (a55 mispick, diagnostic)
        (512, 512, 120, true),  // square, N divisible by every nr (fair cross-kernel throughput)
        (256, 256, 256, true),  // mid square
        (512, 512, 1, false), // matvec — n=1 mmv path, ~1 Gf/s and noisy on slow boards (diagnostic)
        (2048, 2048, 1, false), // wide matvec / LLM decode — n=1 mmv, noisy (diagnostic)
        (32, 27, 22201, false), // inceptionv3 first conv: tiny K=27, kernel crossover (diagnostic)
        (192, 288, 1225, true), // inceptionv3 mid conv im2col
        (64, 64, 64, true),   // small
        (20, 256, 2, false),  // M-padding case, mr overshoot (diagnostic)
        (50, 256, 4, false),  // M-padding case (diagnostic)
    ];
    // armv7 (a7/a9): slow in-order cores and a 32-bit space; keep only light problems.
    if cfg!(target_arch = "arm") {
        shapes.retain(|&(m, k, n, _)| m * k * n <= 5_000_000);
    }
    shapes
        .into_iter()
        .flat_map(|(m, k, n, g)| [(f32::datum_type(), m, k, n, g), (f16::datum_type(), m, k, n, g)])
        .collect()
}

pub(crate) fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    let params = HwbenchParams::from_arg_matches(matches)?;

    if !params.json {
        print_host();
    }

    let mut report = Report::default();
    if !params.no_cache || !params.no_memory {
        let mut threads = (1..=num_cpus::get()).collect_vec();
        for extra in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0] {
            let value = (num_cpus::get() * (extra * 4.) as usize) / 4;
            if !threads.contains(&value) {
                threads.push(value);
            }
        }
        if !params.no_cache {
            let bw = threads
                .iter()
                .map(|&t| Bandwidth { threads: t, bytes_per_s: l1_bandwidth_seq(t) })
                .collect_vec();
            if params.json {
                report.cache = Some(bw);
            } else {
                print_bandwidth("# Cache", "L1", &bw);
            }
        }
        if !params.no_memory {
            let bw = threads
                .iter()
                .map(|&t| Bandwidth { threads: t, bytes_per_s: main_memory_bandwith_seq(t) })
                .collect_vec();
            if params.json {
                report.main_memory = Some(bw);
            } else {
                print_bandwidth("# Main memory", "L∞", &bw);
            }
        }
    }

    if !params.no_matmul {
        let requests = if params.shapes.is_empty() {
            default_battery()
        } else {
            params.shapes.iter().map(|s| parse_shape(s)).flatten_ok().collect::<TractResult<_>>()?
        };
        for (dt, m, k, n, gated) in requests {
            let shape = bench_shape(dt, m, k, n, gated)?;
            if !params.json {
                print_shape(&shape);
            }
            report.matmul.push(shape);
        }
    }

    if params.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    if params.assert {
        assert_picks(&report.matmul, params.tolerance)?;
    }

    Ok(())
}

fn assert_picks(shapes: &[ShapeResult], tolerance: f64) -> TractResult<()> {
    let floor = 1.0 - tolerance / 100.0;
    let mut failures = 0;
    for shape in shapes {
        // Diagnostic shapes are reported but not gated (known-hard for every picker).
        if !shape.gated {
            continue;
        }
        // No viable kernel for this dtype on this target (e.g. armv7 f16 has only the
        // emulated Dreadful kernel, which the bench skips) — nothing to assert.
        if shape.kernels.is_empty() {
            continue;
        }
        match (shape.pick_ratio(), shape.picked(), shape.best()) {
            (Some(ratio), Some(picked), Some(best)) if ratio < floor => {
                failures += 1;
                eprintln!(
                    "PICK REGRESSION {}: picked {} at {:.1}% of best {} ({} vs {})",
                    shape.title(),
                    picked.kernel,
                    ratio * 100.0,
                    best.kernel,
                    si_prefix(picked.flop_per_s, "flop/s"),
                    si_prefix(best.flop_per_s, "flop/s"),
                );
            }
            (None, _, Some(best)) => {
                failures += 1;
                eprintln!(
                    "PICK REGRESSION {}: dispatcher chose an emulated/skipped kernel over {}",
                    shape.title(),
                    best.kernel,
                );
            }
            _ => {}
        }
    }
    if failures > 0 {
        bail!("{failures} kernel pick(s) lag the fastest by more than {tolerance}%");
    }
    Ok(())
}

fn print_host() {
    println!("# Cores");
    println!("cpus: {}", num_cpus::get());
    println!("physical cpus: {}", num_cpus::get_physical());
    println!();

    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        println!("# Excerpt from /proc/cpuinfo");
        for line in cpuinfo.lines() {
            if line.is_empty() {
                break;
            }
            if ["model name", "cache size", "bogomips", "BogoMIPS", "Features", "CPU", "flags"]
                .iter()
                .any(|needle| line.starts_with(needle))
            {
                println!(" * {line}");
            }
        }
        println!();

        if let Some(flags) = cpuinfo
            .lines()
            .find(|line| line.starts_with("flags") || line.starts_with("Features"))
            .and_then(|l| l.split_once(":"))
            .map(|pair| pair.1)
        {
            print!("# Relevant CPU flags/features: ");
            for flag in flags.split_whitespace() {
                if ["fpu", "sse", "avx", "f16", "fma", "fp", "asimd", "neon", "vfp"]
                    .iter()
                    .any(|needle| flag.starts_with(needle))
                {
                    print!("{flag} ")
                };
            }
            println!("\n");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!(
            "# Aarch64 subfamily detected by tract-linalg: {:?}\n",
            tract_linalg::arm64::Kind::choose()
        );
    }
}

fn print_bandwidth(header: &str, label: &str, bw: &[Bandwidth]) {
    println!("{header}");
    for b in bw {
        println!(
            "{:2}-thread {label} : {} — {}",
            b.threads,
            si_prefix(b.bytes_per_s, "B/s"),
            si_prefix(b.bytes_per_s / b.threads as f64, "B/s/thread"),
        );
    }
    println!();
}

fn print_shape(shape: &ShapeResult) {
    let tag = if shape.gated { "" } else { "  [diagnostic — not gated]" };
    println!("# Matmul {}{tag}\n", shape.title());
    for k in &shape.kernels {
        print!("{:>35} {:30}", format!("{} ({})", k.kernel, k.packing), k.layout);
        let color = if k.flop_per_s.log10() > 9.0 {
            Green
        } else if k.flop_per_s.log10() > 6.0 {
            Yellow
        } else {
            LightRed
        };
        println!(
            " {} {}",
            color.paint(si_prefix(k.flop_per_s, "flop/s")),
            if k.picked { "<--" } else { "" }
        );
    }
    println!();
}

/// Measured throughput (flop/s) of every candidate kernel at one shape, for the
/// cost-model dataset gatherer. Reuses the same timing path as the pick-gate so a
/// fitted model and `hwbench --assert` agree.
pub(crate) fn kernel_times(
    dt: DatumType,
    m: usize,
    k: usize,
    n: usize,
) -> TractResult<Vec<(String, f64)>> {
    Ok(bench_shape(dt, m, k, n, false)?
        .kernels
        .into_iter()
        .map(|k| (k.kernel, k.flop_per_s))
        .collect())
}

fn bench_shape(
    dt: DatumType,
    m: usize,
    k: usize,
    n: usize,
    gated: bool,
) -> TractResult<ShapeResult> {
    let a = Tensor::zero_dt(dt, &[m, k])?;
    let b = Tensor::zero_dt(dt, &[k, n])?;
    let mut c = Tensor::zero_dt(dt, &[m, n])?;
    let pick = planned_pick(dt, m, k, n)?;
    let mmms = tract_linalg::ops().mmm_impls();
    let mut kernels: Vec<KernelResult> = unsafe {
        mmms.iter()
            // Skip fallback kernels: Dreadful emulates the datatype op-by-op
            // (f16->f32->f16), Generic is unvectorised scalar. Both are last-resort,
            // 100-500x off a real kernel, and only ever the pick when no real kernel
            // exists for this dtype on this target (e.g. armv7 f16) — where the pick
            // among equally-useless fallbacks is noise, not a regression to gate on.
            .filter(|mmm| {
                !matches!(
                    mmm.quality(),
                    ImplementationQuality::Dreadful | ImplementationQuality::Generic
                )
            })
            .flat_map(|mmm| {
                mmm.packings().iter().enumerate().map(move |(pix, (pa, pb))| (mmm, pix, pa, pb))
            })
            .filter(|(_mmm, _pix, pa, pb)| {
                pa.precursor().as_dt() == Some(dt) && pb.precursor().as_dt() == Some(dt)
            })
            .map(|(mmm, pix, pa, pb)| {
                if std::io::stderr().is_terminal() {
                    eprint!("Benching {} ({pix}) at {m}x{k}x{n}x{dt:?}", mmm.name());
                }
                let a = pa.prepare_one(&a, 1, 0).unwrap();
                let b = pb.prepare_one(&b, 0, 1).unwrap();
                let pc = mmm.c_view(Some(0), Some(1)).wrap(&c.view_mut());
                let time = run_bench(|loops| {
                    let mut scratch = mmm.allocate_scratch_space();
                    for _ in 0..loops {
                        mmm.run_with_scratch_space(
                            m,
                            n,
                            scratch.as_mut(),
                            &[
                                FusedSpec::AddMatMul {
                                    a: AsInputValue::Borrowed(&*a),
                                    b: AsInputValue::Borrowed(&*b),
                                    packing: pix,
                                },
                                FusedSpec::Store(pc),
                            ],
                        )
                        .unwrap();
                    }
                });
                if std::io::stderr().is_terminal() {
                    eprint!("\x1B[2K\r"); // clear current line + CR
                }
                KernelResult {
                    kernel: mmm.name().to_string(),
                    packing: pix,
                    layout: format!("{pa} • {pb}"),
                    flop_per_s: (m * k * n) as f64 / time,
                    picked: false,
                }
            })
            .collect()
    };
    for kernel in &mut kernels {
        kernel.picked = pick
            .as_ref()
            .is_some_and(|(name, packing)| kernel.kernel == *name && kernel.packing == *packing);
    }
    kernels.sort_by(|x, y| y.flop_per_s.total_cmp(&x.flop_per_s));
    Ok(ShapeResult { m, k, n, dt: format!("{dt:?}"), gated, kernels })
}

/// The (kernel name, packing index) the real planner picks for this shape,
/// by building the matmul it would see (const weight × runtime activation) and
/// optimizing it — the same path a model takes, not the `ops().mmm()` shortcut.
fn planned_pick(
    dt: DatumType,
    m: usize,
    k: usize,
    n: usize,
) -> TractResult<Option<(String, usize)>> {
    use tract_core::ops::einsum::EinSum;
    use tract_core::ops::matmul::optimized::{OptMatMul, ProtoFusedSpec};

    let mut model = TypedModel::default();
    let a = model.add_const("a", Tensor::zero_dt(dt, &[m, k])?)?;
    let b = model.add_source("b", dt.fact([k, n]))?;
    let mm = model.wire_node("mm", EinSum::new("mk,kn->mn".parse()?, dt), &[a, b])?;
    model.select_output_outlets(&mm)?;
    let model = model.into_optimized()?;

    let Some(node) = model.nodes.iter().find(|n| n.op_is::<OptMatMul>()) else {
        return Ok(None);
    };
    let op = node.op_as::<OptMatMul>().unwrap();
    // Concrete n: the vector kernel (mode 0) runs for n==1, the matrix kernel otherwise.
    let mode = if n == 1 { 0 } else { op.mmm.len() - 1 };
    let kernel = op.mmm[mode].name().to_string();
    let packing = op.micro_ops.iter().find_map(|micro_op| match micro_op {
        ProtoFusedSpec::AddMatMul { packings, .. } => Some(packings[mode].0),
        _ => None,
    });
    Ok(packing.map(|packing| (kernel, packing)))
}
