//! `tract cost-model` — gather a matmul-kernel timing dataset on the running CPU and
//! fit an analytic `LinearCostModel` from it. `gather` runs on the target (e.g. via
//! cargo-dinghy); `fit` is a pure host-side transform. Regenerating a CPU's cost model
//! after a kernel change is: gather on the board, fit, drop the file in, wire it.

use clap::{Args, FromArgMatches, Subcommand};
use std::collections::HashMap;
use std::io::Write;
use tract_core::internal::*;

#[derive(Subcommand)]
enum Cmd {
    /// Time every f32 matmul kernel over a shape sweep and write a dataset.
    /// Run on the target CPU (e.g. via cargo-dinghy).
    Gather(Gather),
    /// Fit a LinearCostModel from a gathered dataset and emit a Rust source file.
    Fit(Fit),
    /// One-shot on the target CPU (run from the tract source tree): detect the
    /// platform, gather a class-appropriate dataset (seed shapes + random sweep),
    /// fit, validate against the currently-installed picker, and write the new
    /// model to a side file with a ready-to-use `mv` — never overwrites in place.
    Regen(Regen),
}

#[derive(Args)]
struct Gather {
    /// M values: range 'lo-hi' or list 'a,b,c'
    #[arg(long, default_value = "1-256")]
    m: String,
    /// K values: range 'lo-hi' or list 'a,b,c'
    #[arg(long, default_value = "1-256")]
    k: String,
    /// N values: range 'lo-hi' or list 'a,b,c'
    #[arg(long, default_value = "1-256")]
    n: String,
    /// Skip sampled shapes whose m*k*n exceeds this
    #[arg(long, default_value_t = 4_000_000)]
    mkn: usize,
    /// Number of shapes to sample
    #[arg(long, short, default_value_t = 128)]
    size: usize,
    /// PRNG seed (the sweep is deterministic in the seed, so datasets are reproducible)
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Dataset output path ('-' for stdout)
    out: String,
}

#[derive(Args)]
struct Fit {
    /// Dataset produced by `gather`
    dataset: String,
    /// Output .rs path
    out: String,
}

pub fn command() -> clap::Command {
    Cmd::augment_subcommands(
        clap::Command::new("cost-model")
            .about("Gather a matmul-kernel timing dataset and fit a LinearCostModel")
            .subcommand_required(true),
    )
}

pub fn handle(matches: &clap::ArgMatches) -> TractResult<()> {
    match Cmd::from_arg_matches(matches)? {
        Cmd::Gather(g) => gather(g),
        Cmd::Fit(f) => fit(f),
        Cmd::Regen(r) => regen(r),
    }
}

#[derive(Args)]
struct Regen {
    /// PRNG seed for the random sweep (deterministic in the seed)
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Override the random-sweep sample count for the detected class
    #[arg(long)]
    size: Option<usize>,
    /// Force a platform id instead of auto-detecting (e.g. intel_avx512, cortex_a7, apple_m4)
    #[arg(long)]
    platform: Option<String>,
    /// Directory for the generated side files (default: system temp dir)
    #[arg(long)]
    out_dir: Option<String>,
}

// --- shape sampling: deterministic so a dataset is reproducible from (args, seed) ---

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_add(0x9e3779b97f4a7c15))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

enum Sampling {
    Range(usize, usize),
    Fixed(Vec<usize>),
}
impl Sampling {
    fn parse(s: &str) -> TractResult<Sampling> {
        Ok(if let Some((a, b)) = s.split_once('-') {
            Sampling::Range(a.parse()?, b.parse()?)
        } else {
            Sampling::Fixed(s.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?)
        })
    }
    fn sample(&self, rng: &mut Lcg) -> usize {
        match self {
            Sampling::Range(a, b) => a + (rng.next() as usize) % (b - a + 1),
            Sampling::Fixed(v) => v[(rng.next() as usize) % v.len()],
        }
    }
}

fn gather(g: Gather) -> TractResult<()> {
    let (ms, ks, ns) = (Sampling::parse(&g.m)?, Sampling::parse(&g.k)?, Sampling::parse(&g.n)?);
    let mut rng = Lcg::new(g.seed);
    let mut w: Box<dyn Write> = if g.out == "-" {
        Box::new(std::io::stdout())
    } else {
        Box::new(std::fs::File::create(&g.out)?)
    };
    let dt = f32::datum_type();
    let (mut done, mut tries) = (0usize, 0usize);
    while done < g.size && tries < g.size.saturating_mul(1000).max(1000) {
        tries += 1;
        let (m, k, n) = (ms.sample(&mut rng), ks.sample(&mut rng), ns.sample(&mut rng));
        if m == 0 || k == 0 || n == 0 || m * k * n > g.mkn {
            continue;
        }
        for (kernel, flop_per_s) in crate::hwbench::kernel_times(dt, m, k, n)? {
            let (mr, nr) = geom(&kernel);
            let dur = (m * k * n) as f64 / flop_per_s;
            writeln!(w, "{kernel} {mr} {nr} {m} {k} {n} {dur}")?;
        }
        done += 1;
        eprintln!("  {done}/{}  {m}x{k}x{n}", g.size);
    }
    Ok(())
}

// --- fit: least-squares per-kernel time model, emit a LinearCostModel source file ---

fn geom(name: &str) -> (usize, usize) {
    name.split('_')
        .find_map(|p| {
            let (a, b) = p.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or_else(|| panic!("no NxM tile geometry in kernel name {name}"))
}
fn padded_work(m: usize, k: usize, n: usize, mr: usize, nr: usize) -> f64 {
    (m.div_ceil(mr) * mr * n.div_ceil(nr) * nr * k) as f64
}
fn n_tiles(m: usize, n: usize, mr: usize, nr: usize) -> f64 {
    (m.div_ceil(mr) * n.div_ceil(nr)) as f64
}

/// Solve a small symmetric system `a·x = b` (n <= 3) by Gaussian elimination.
#[allow(clippy::needless_range_loop)] // indexed elimination reads a[col] while writing a[r]
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv =
            (col..n).max_by(|&r, &s| a[r][col].abs().partial_cmp(&a[s][col].abs()).unwrap())?;
        a.swap(col, piv);
        b.swap(col, piv);
        if a[col][col].abs() < 1e-40 {
            return None;
        }
        for r in 0..n {
            if r != col {
                let f = a[r][col] / a[col][col];
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some((0..n).map(|i| b[i] / a[i][i]).collect())
}

/// Non-negative least squares for `time = c0·padded_work + c1·n_tiles + c2`. Times are
/// sums of non-negative costs, so the coefficients must be >= 0; unconstrained LS on noisy
/// data can produce a negative coefficient that predicts a near-zero time for some shape and
/// picks a slow kernel. Exact for 3 features: the optimum lies on a face of the non-negative
/// orthant, so fit each of the 8 active-subsets unconstrained and keep the feasible (all >=0)
/// one with the smallest residual.
fn fit_nnls(rows: &[([f64; 3], f64)]) -> [f64; 3] {
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for (x, y) in rows {
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += x[i] * x[j];
            }
            atb[i] += x[i] * y;
        }
    }
    let mut best = ([0.0f64; 3], f64::MAX);
    for mask in 1u8..8 {
        let idx: Vec<usize> = (0..3).filter(|i| mask & (1 << i) != 0).collect();
        let sub_a: Vec<Vec<f64>> =
            idx.iter().map(|&i| idx.iter().map(|&j| ata[i][j]).collect()).collect();
        let sub_b: Vec<f64> = idx.iter().map(|&i| atb[i]).collect();
        let Some(sol) = solve(sub_a, sub_b) else { continue };
        if sol.iter().any(|&v| v < 0.0) {
            continue;
        }
        let mut c = [0.0f64; 3];
        for (k, &i) in idx.iter().enumerate() {
            c[i] = sol[k];
        }
        let mut resid = -2.0 * (0..3).map(|i| c[i] * atb[i]).sum::<f64>();
        for i in 0..3 {
            for j in 0..3 {
                resid += c[i] * ata[i][j] * c[j];
            }
        }
        if resid < best.1 {
            best = (c, resid);
        }
    }
    best.0
}

fn fit(f: Fit) -> TractResult<()> {
    let txt = std::fs::read_to_string(&f.dataset)?;
    let mut rows: Vec<(String, [f64; 3], f64)> = vec![];
    let p = |s: &str| -> usize { s.parse().unwrap() };
    for line in txt.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        // `kernel m k n dur` (5) or `kernel mr nr m k n dur` (7)
        let (mr, nr, m, k, n, dur) = match cols.len() {
            5 => {
                let (mr, nr) = geom(cols[0]);
                (mr, nr, p(cols[1]), p(cols[2]), p(cols[3]), cols[4].parse()?)
            }
            7 => (p(cols[1]), p(cols[2]), p(cols[3]), p(cols[4]), p(cols[5]), cols[6].parse()?),
            _ => continue,
        };
        rows.push((
            cols[0].to_string(),
            [padded_work(m, k, n, mr, nr), n_tiles(m, n, mr, nr), 1.0],
            dur,
        ));
    }
    ensure!(!rows.is_empty(), "no usable rows in {}", f.dataset);
    let mut kernels: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    kernels.sort();
    kernels.dedup();
    let default_kernel = kernels
        .iter()
        .max_by_key(|name| {
            let (mr, nr) = geom(name);
            mr * nr
        })
        .unwrap()
        .clone();

    let mut src = String::new();
    src.push_str("use crate::frame::mmm::LinearCostModel;\n\n");
    src.push_str("pub fn linear_model() -> LinearCostModel<'static> {\n");
    src.push_str("    LinearCostModel {\n");
    src.push_str(&format!("        default_kernel: {default_kernel:?},\n"));
    src.push_str("        kernels: &[\n");
    for kern in &kernels {
        src.push_str(&format!("            {kern:?},\n"));
    }
    src.push_str("        ],\n        coeffs: &[\n");
    for kern in &kernels {
        let per: Vec<([f64; 3], f64)> =
            rows.iter().filter(|r| &r.0 == kern).map(|r| (r.1, r.2)).collect();
        let c = fit_nnls(&per);
        src.push_str(&format!(
            "            [{:e}, {:e}, {:e}],\n",
            c[0] as f32, c[1] as f32, c[2] as f32
        ));
    }
    src.push_str("        ],\n    }\n}\n");
    std::fs::write(&f.out, src)?;
    eprintln!("wrote {} ({} kernels, default {default_kernel})", f.out, kernels.len());
    Ok(())
}

// --- regen: detect the running platform, gather a class-appropriate dataset, fit,
//     validate against the installed picker, and emit a side file + `mv` (no in-place write) ---

/// Device class: sets the shape regime a platform's models must cover.
#[derive(Clone, Copy, PartialEq)]
enum Class {
    Small32,
    Small64,
    Big64,
}

impl Class {
    fn tag(self) -> &'static str {
        match self {
            Class::Small32 => "small32",
            Class::Small64 => "small64",
            Class::Big64 => "big64",
        }
    }
    /// (m_hi, k_hi, n_hi, mkn_cap, default random-sweep size)
    fn sweep(self) -> (usize, usize, usize, usize, usize) {
        match self {
            Class::Small32 => (192, 192, 192, 4_000_000, 160),
            Class::Small64 => (256, 256, 256, 8_000_000, 180),
            Class::Big64 => (4096, 4096, 512, 268_435_456, 160),
        }
    }
}

struct Platform {
    id: String,
    cpu: String,
    class: Class,
    rs_rel: String,
    txt_rel: String,
}

fn platform_from_id(id: &str, cpu: String) -> TractResult<Platform> {
    let (dir, class) = if id.starts_with("cortex_a7") || id.starts_with("cortex_a9") {
        ("linalg/src/arm32", Class::Small32)
    } else if id.starts_with("cortex_a") || id.starts_with("neoverse") {
        ("linalg/src/arm64", Class::Small64)
    } else if id.starts_with("apple_") {
        ("linalg/src/arm64", Class::Big64)
    } else if id.starts_with("intel_") || id.starts_with("amd_") {
        ("linalg/src/x86_64_fma", Class::Big64)
    } else {
        bail!("unknown platform id '{id}'");
    };
    Ok(Platform {
        rs_rel: format!("{dir}/{id}_linear.rs"),
        txt_rel: format!("{dir}/{id}.txt"),
        id: id.to_string(),
        cpu,
        class,
    })
}

fn cpuinfo_field(name: &str) -> Option<String> {
    std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
    })
}

fn shell(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(target_arch = "x86_64")]
fn auto_platform() -> Option<Platform> {
    let vendor = match std::env::var("TRACT_X86_KIND").ok().as_deref() {
        Some("intel") => "intel",
        Some("amd") => "amd",
        Some(_) => return None,
        None => {
            let id = std::arch::x86_64::__cpuid(0);
            let mut s = [0u8; 12];
            s[0..4].copy_from_slice(&id.ebx.to_le_bytes());
            s[4..8].copy_from_slice(&id.edx.to_le_bytes());
            s[8..12].copy_from_slice(&id.ecx.to_le_bytes());
            match &s {
                b"GenuineIntel" => "intel",
                b"AuthenticAMD" => "amd",
                _ => return None,
            }
        }
    };
    let tier = if is_x86_feature_detected!("avx512f") { "avx512" } else { "fma" };
    let cpu = cpuinfo_field("model name").unwrap_or_else(|| "unknown x86_64".to_string());
    platform_from_id(&format!("{vendor}_{tier}"), cpu).ok()
}

#[cfg(target_arch = "aarch64")]
fn auto_platform() -> Option<Platform> {
    if cfg!(target_os = "macos") {
        let brand = shell("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
        let id = if brand.contains("M1") {
            "apple_m1"
        } else if brand.contains("M2") {
            "apple_m2"
        } else if brand.contains("M3") {
            "apple_m3"
        } else if brand.contains("M4") {
            "apple_m4"
        } else {
            return None;
        };
        return platform_from_id(id, brand).ok();
    }
    let part = cpuinfo_field("CPU part")?;
    let id = match part.as_str() {
        "0xd03" => "cortex_a53",
        "0xd05" => "cortex_a55",
        _ => return None,
    };
    let cpu = cpuinfo_field("model name").unwrap_or_else(|| part.clone());
    platform_from_id(id, cpu).ok()
}

#[cfg(target_arch = "arm")]
fn auto_platform() -> Option<Platform> {
    let part = cpuinfo_field("CPU part")?;
    let id = match part.as_str() {
        "0xc07" => "cortex_a7",
        "0xc09" => "cortex_a9",
        _ => return None,
    };
    let cpu = cpuinfo_field("model name").unwrap_or_else(|| part.clone());
    platform_from_id(id, cpu).ok()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
fn auto_platform() -> Option<Platform> {
    None
}

fn load_seed_shapes(class: Class) -> Vec<(usize, usize, usize)> {
    let path = format!("linalg/cost-model-seeds/{}.txt", class.tag());
    let Ok(s) = std::fs::read_to_string(&path) else { return vec![] };
    s.lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect()
}

/// Log-uniform draw in `[lo, hi]` — densely samples small values, where kernel
/// selection is hardest and real workloads concentrate.
fn log_uniform(rng: &mut Lcg, lo: usize, hi: usize) -> usize {
    if hi <= lo {
        return lo.max(1);
    }
    let u = (rng.next() as f64) / ((1u64 << 31) as f64);
    let (l, h) = ((lo.max(1) as f64).ln(), (hi as f64).ln());
    ((l + u * (h - l)).exp().round() as usize).clamp(lo.max(1), hi)
}

fn render_rs(
    default_kernel: &str,
    kernels: &[String],
    coeffs: &[[f64; 3]],
    header: &str,
) -> String {
    let mut src = String::new();
    src.push_str(header);
    src.push_str("use crate::frame::mmm::LinearCostModel;\n\n");
    src.push_str("pub fn linear_model() -> LinearCostModel<'static> {\n");
    src.push_str("    LinearCostModel {\n");
    src.push_str(&format!("        default_kernel: {default_kernel:?},\n"));
    src.push_str("        kernels: &[\n");
    for kn in kernels {
        src.push_str(&format!("            {kn:?},\n"));
    }
    src.push_str("        ],\n        coeffs: &[\n");
    for c in coeffs {
        src.push_str(&format!(
            "            [{:e}, {:e}, {:e}],\n",
            c[0] as f32, c[1] as f32, c[2] as f32
        ));
    }
    src.push_str("        ],\n    }\n}\n");
    src
}

fn provenance(plat: &Platform, cur: f64, new: f64, n: usize) -> String {
    let user = std::env::var("USER")
        .ok()
        .or_else(|| std::env::var("USERNAME").ok())
        .or_else(|| shell("id", &["-un"]))
        .unwrap_or_else(|| "unknown".into());
    let host = shell("hostname", &[]).unwrap_or_else(|| "unknown".into());
    let date = shell("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_else(|| "unknown".into());
    let git = shell("git", &["rev-parse", "--short", "HEAD"])
        .map(|h| format!(" (git {h})"))
        .unwrap_or_default();
    format!(
        "// Generated by `tract cost-model regen` — do not hand-edit.\n\
         // platform: {}   cpu: {}\n\
         // user: {user}   host: {host}   date: {date}\n\
         // tract: {}{git}\n\
         // validation over {n} shapes: current picker regret {cur:.4}x -> linear {new:.4}x\n",
        plat.id,
        plat.cpu,
        env!("CARGO_PKG_VERSION"),
    )
}

fn regen(r: Regen) -> TractResult<()> {
    let plat = match &r.platform {
        Some(id) => platform_from_id(id, "(forced)".to_string())?,
        None => auto_platform().ok_or_else(|| {
            format_err!("could not detect a cost-model platform for this CPU; pass --platform")
        })?,
    };
    let class = plat.class;
    let (m_hi, k_hi, n_hi, mkn_cap, def_size) = class.sweep();
    let size = r.size.unwrap_or(def_size);
    eprintln!("platform {} (class {}), cpu: {}", plat.id, class.tag(), plat.cpu);

    // shape set = seed shapes (model findings) + deterministic random sweep, n >= 2 (n=1 is mmv)
    let seeds = load_seed_shapes(class);
    eprintln!("{} seed shape(s) from linalg/cost-model-seeds/{}.txt", seeds.len(), class.tag());
    let mut shapes: Vec<(usize, usize, usize)> =
        seeds.into_iter().filter(|&(_, _, n)| n >= 2).collect();
    let mut rng = Lcg::new(r.seed);
    let (mut got, mut tries) = (0usize, 0usize);
    while got < size && tries < size * 1000 + 1000 {
        tries += 1;
        let (m, k, n) = (
            log_uniform(&mut rng, 1, m_hi),
            log_uniform(&mut rng, 1, k_hi),
            log_uniform(&mut rng, 2, n_hi),
        );
        if n < 2 || m * k * n > mkn_cap {
            continue;
        }
        shapes.push((m, k, n));
        got += 1;
    }

    // gather: time every f32 mmm kernel at each shape
    let dt = f32::datum_type();
    let mut rows: Vec<(String, usize, usize, usize, usize, usize, f64)> = vec![];
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        for (kernel, flop_per_s) in crate::hwbench::kernel_times(dt, m, k, n)? {
            let (mr, nr) = geom(&kernel);
            rows.push((kernel, mr, nr, m, k, n, (m * k * n) as f64 / flop_per_s));
        }
        eprintln!("  {}/{}  {m}x{k}x{n}", i + 1, shapes.len());
    }
    ensure!(!rows.is_empty(), "no timing rows gathered");

    // fit: per-kernel NNLS on [padded_work, n_tiles, 1]
    let mut kernels: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    kernels.sort();
    kernels.dedup();
    let coeffs: Vec<[f64; 3]> = kernels
        .iter()
        .map(|kn| {
            let per: Vec<([f64; 3], f64)> = rows
                .iter()
                .filter(|r| &r.0 == kn)
                .map(|r| {
                    ([padded_work(r.3, r.4, r.5, r.1, r.2), n_tiles(r.3, r.5, r.1, r.2), 1.0], r.6)
                })
                .collect();
            fit_nnls(&per)
        })
        .collect();
    let default_kernel = kernels.iter().max_by_key(|n| geom(n).0 * geom(n).1).unwrap().clone();

    // validate: new linear vs the installed picker, over the gathered shapes
    let idx: HashMap<&str, usize> =
        kernels.iter().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
    let mut by_shape: HashMap<(usize, usize, usize), HashMap<String, f64>> = HashMap::new();
    for (kn, _, _, m, k, n, dur) in &rows {
        by_shape.entry((*m, *k, *n)).or_default().insert(kn.clone(), *dur);
    }
    let predict = |kn: &str, m: usize, k: usize, n: usize| -> f64 {
        let (mr, nr) = geom(kn);
        let c = &coeffs[idx[kn]];
        c[0] * padded_work(m, k, n, mr, nr) + c[1] * n_tiles(m, n, mr, nr) + c[2]
    };
    let (mut new_sum, mut new_orc, mut cur_sum, mut cur_orc) = (0.0, 0.0, 0.0, 0.0);
    let (mut cur_skip, mut n_shapes, mut worst_new, mut worst_cur) =
        (0usize, 0usize, 1.0f64, 1.0f64);
    for (&(m, k, n), km) in &by_shape {
        let oracle = km.values().cloned().fold(f64::MAX, f64::min);
        if oracle == f64::MAX {
            continue;
        }
        n_shapes += 1;
        let newk = km
            .keys()
            .min_by(|a, b| predict(a, m, k, n).partial_cmp(&predict(b, m, k, n)).unwrap())
            .unwrap();
        new_sum += km[newk];
        new_orc += oracle;
        worst_new = worst_new.max(km[newk] / oracle);
        match tract_linalg::ops().mmm(dt, Some(m), Some(k), Some(n)) {
            Some(cur) if km.contains_key(cur.name()) => {
                let cd = km[cur.name()];
                cur_sum += cd;
                cur_orc += oracle;
                worst_cur = worst_cur.max(cd / oracle);
            }
            _ => cur_skip += 1,
        }
    }
    let cur_regret = if cur_orc > 0.0 { cur_sum / cur_orc } else { f64::NAN };
    let new_regret = if new_orc > 0.0 { new_sum / new_orc } else { f64::NAN };
    eprintln!("\nvalidation over {n_shapes} shapes ({} kernels):", kernels.len());
    eprintln!(
        "  current picker : regret {cur_regret:.4}x  worst {worst_cur:.2}x{}",
        if cur_skip > 0 {
            format!("  ({cur_skip} shape(s) skipped: picked an untimed kernel)")
        } else {
            String::new()
        }
    );
    eprintln!("  new linear     : regret {new_regret:.4}x  worst {worst_new:.2}x");

    // emit side files + ready-to-use mv (never overwrite in place)
    let header = provenance(&plat, cur_regret, new_regret, n_shapes);
    let src = render_rs(&default_kernel, &kernels, &coeffs, &header);
    let mut ds = String::new();
    for (kn, mr, nr, m, k, n, dur) in &rows {
        ds.push_str(&format!("{kn} {mr} {nr} {m} {k} {n} {dur}\n"));
    }
    let out_dir = r.out_dir.map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&out_dir)?;
    let rs_out = out_dir.join(format!("{}_linear.rs", plat.id));
    let txt_out = out_dir.join(format!("{}.txt", plat.id));
    std::fs::write(&rs_out, src)?;
    std::fs::write(&txt_out, ds)?;
    println!("\n# regenerated {} model. review, then install with:", plat.id);
    println!("mv {} {}", rs_out.display(), plat.rs_rel);
    println!("mv {} {}", txt_out.display(), plat.txt_rel);
    Ok(())
}
