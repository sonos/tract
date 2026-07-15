//! `tract cost-model` — gather a matmul-kernel timing dataset on the running CPU and
//! fit an analytic `LinearCostModel` from it. `gather` runs on the target (e.g. via
//! cargo-dinghy); `fit` is a pure host-side transform. Regenerating a CPU's cost model
//! after a kernel change is: gather on the board, fit, drop the file in, wire it.

use clap::{Args, FromArgMatches, Subcommand};
use std::io::Write;
use tract_core::internal::*;

#[derive(Subcommand)]
enum Cmd {
    /// Time every f32 matmul kernel over a shape sweep and write a dataset.
    /// Run on the target CPU (e.g. via cargo-dinghy).
    Gather(Gather),
    /// Fit a LinearCostModel from a gathered dataset and emit a Rust source file.
    Fit(Fit),
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
    }
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

/// Least squares for `y = b0*x0 + b1*x1 + b2` via 3x3 normal equations.
#[allow(clippy::needless_range_loop)] // indexed Gaussian elimination reads a[col] while writing a[r]
fn fit3(rows: &[([f64; 3], f64)]) -> [f64; 3] {
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
    let mut a = [[0.0; 4]; 3];
    for i in 0..3 {
        a[i][..3].copy_from_slice(&ata[i]);
        a[i][3] = atb[i];
    }
    for col in 0..3 {
        let piv = (col..3)
            .max_by(|&r, &s| a[r][col].abs().partial_cmp(&a[s][col].abs()).unwrap())
            .unwrap();
        a.swap(col, piv);
        let d = a[col][col];
        for j in col..4 {
            a[col][j] /= d;
        }
        for r in 0..3 {
            if r != col {
                let f = a[r][col];
                for j in col..4 {
                    a[r][j] -= f * a[col][j];
                }
            }
        }
    }
    [a[0][3], a[1][3], a[2][3]]
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
        let c = fit3(&per);
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
