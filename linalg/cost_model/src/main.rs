use pbr::ProgressBar;
use tract_data::internal::*;
use tract_linalg::{frame::MatMatMul, mmm::FusedSpec};

use rand::prelude::*;
use std::io::Write;
use std::time::{Duration, Instant};
use tract_itertools::Itertools;

pub fn ruin_cache() {
    let _a = (0..1_000_000).collect::<Vec<i32>>();
}

fn order_f<F: tract_num_traits::Float>(&a: &F, &b: &F) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Greater
    }
}

pub struct Bencher {
    bench_time_target: Duration,
    chunk_time_target: Duration,
    chunks_min_count: usize,
    chunks_max_count: usize,
}

impl Bencher {
    fn black_box<T>(dummy: T) -> T {
        unsafe {
            let ret = std::ptr::read_volatile(&dummy);
            std::mem::forget(dummy);
            ret
        }
    }

    pub fn run_bench<T, I, P: FnMut() -> I, F: FnMut(I) -> T>(&self, mut prep: P, mut f: F) -> f64 {
        let i = prep();
        let i2 = prep();
        Self::black_box(f(i));
        let start = Instant::now();
        Self::black_box(f(i2));
        let once = start.elapsed();
        //   dbg!(once);
        let evaled = if once < Duration::from_millis(1) {
            let is = (0..1000).map(|_| prep()).collect_vec();
            let start = Instant::now();
            for i in is {
                Self::black_box(f(i));
            }
            start.elapsed().as_secs_f64() / 1000.
        } else {
            once.as_secs_f64()
        };
        //    let warmup = (0.2 / evaled) as usize;
        //    let iters = 5.0 / evaled as f64;
        // chunk just need to be big enough be measurable
        //    dbg!(evaled);
        let chunk = ((self.chunk_time_target.as_secs_f64() / evaled) as usize).max(1);
        // chunks is the number of measure. make it 1000 at least, 10000 at most
        let chunks = ((self.bench_time_target.as_secs_f64() / (evaled * chunk as f64)) as usize)
            .max(self.chunks_min_count)
            .min(self.chunks_max_count);
        // let chunks = 10;
        //dbg!(chunk, chunks);
        let mut measures = vec![0.0; chunks];
        /*
        for _ in 0..warmup {
        black_box(f());
        }
        */
        for i in 0..chunks {
            let is = (0..chunk).map(|_| prep());
            let start = Instant::now();
            for i in is {
                Self::black_box(f(i));
            }
            let time = start.elapsed().as_secs_f64();
            measures[i] = time / chunk as f64;
        }
        measures.sort_by(order_f);
        let q1 = measures[chunks / 4];
        /*
           let q3 = measures[chunks - chunks / 4];
           let iq = q3 - q1;
        //    measures.retain(|&x| x >= q1 && x <= q3);
        let epsilon = iq * 2. / (q3 + q1);
        eprintln!("evaled: {} chunk:{} chunks: {} epsilon: {:.3e}", evaled, chunk, chunks, epsilon);
        */
        /*
        let mut hist = vec![0; 101];
        for m in &measures {
        let bucket = (m - measures[0]) / (measures[measures.len() - 1] - measures[0]);
        hist[(100. * bucket) as usize] += 1;
        }
        eprintln!("{hist:?}");
        eprintln!("q1: {}", measures[measures.len() / 4]);
        */
        /*
        eprintln!("avg: {}", );
        measures[chunks / 4] //[..chunks / 2].iter().copied().sum::<f64>() / (chunks / 2) as f64
        */
        q1
    }
}

fn measure_add_mat_mul(
    bencher: &Bencher,
    mm: &dyn MatMatMul,
    m: usize,
    k: usize,
    n: usize,
) -> f64 {
    let dt = mm.internal_type();
    unsafe {
        let time = bencher.run_bench(
            || {
                let a =
                    Tensor::zero_aligned_dt(dt, &[mm.a_pack().len(k, m)], mm.a_pack().alignment())
                        .unwrap();
                let b =
                    Tensor::zero_aligned_dt(dt, &[mm.b_pack().len(k, n)], mm.b_pack().alignment())
                        .unwrap();
                let c = Tensor::zero_dt(dt, &[m, n]).unwrap();
                let pa = mm.a_packed(dt.size_of(), k).wrap(&a.view());
                let pb = mm.b_packed(dt.size_of(), k).wrap(&b.view()).unwrap();
                let pc = mm.c_view(0, 1).wrap(&c.view());
                let scratch = mm.allocate_scratch_space();
                (scratch, a, b, c, pa, pb, pc)
            },
            |(mut scratch, _, _, _, pa, pb, pc)| {
                mm.run_with_scratch_space(
                    m,
                    n,
                    scratch.as_mut(),
                    &[FusedSpec::AddMatMul { a: pa, b: pb.clone(), k }, FusedSpec::Store(pc)],
                )
                .unwrap();
            },
        );
        time
    }
}

#[derive(Clone)]
struct Sample {
    kernel: String,
    mr: usize,
    nr: usize,
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Clone)]
struct Dataset(Vec<(Sample, f64)>);

impl Dataset {
    pub fn smart_sample(mmm: &[&dyn MatMatMul]) -> Vec<Sample> {
        let mut inputs = vec![];
        for mm in mmm {
            let ms = [1, 2, 4, 32, 128]
                .iter()
                .map(|m| m * mm.mr())
                .flat_map(|m| [m - 1, m, m + 1])
                .collect_vec();
            let ns = [1, 2, 4, 32, 128]
                .iter()
                .map(|m| m * mm.nr())
                .flat_map(|m| [m - 1, m, m + 1])
                .collect_vec();
            let ks = [32, 128, 1024];
            for m in ms {
                for &n in &ns {
                    for k in ks {
                        inputs.push(Sample {
                            kernel: mm.kernel_name().to_string(),
                            mr: mm.mr(),
                            nr: mm.nr(),
                            m,
                            k,
                            n,
                        });
                    }
                }
            }
        }
        inputs
    }

    pub fn allkernels_random_sample(
        mmm: &[&dyn MatMatMul],
        size: usize,
        max_m: usize,
        max_k: usize,
        max_n: usize,
        max_mkn: usize,
    ) -> Vec<Sample> {
        let mut inputs = vec![];
        let mut rng = thread_rng();
        while inputs.len() < size * mmm.len() {
            let m = rng.gen_range(1..max_m);
            let k = rng.gen_range(0..max_k);
            let n = rng.gen_range(1..max_n);
            if max_mkn >= m * k * n {
                for mm in mmm {
                    inputs.push(Sample {
                        kernel: mm.kernel_name().to_string(),
                        mr: mm.mr(),
                        nr: mm.nr(),
                        m,
                        k,
                        n,
                    });
                }
            }
        }
        inputs
    }

    pub fn make_dataset(
        bencher: &Bencher,
        mut inputs: Vec<Sample>,
        mmm: &[&dyn MatMatMul],
    ) -> Dataset {
        //        let ruin_cache_time = bencher.run_bench(|| ruin_cache());
        let mut rng = thread_rng();
        inputs.shuffle(&mut rng);
        let mut progress_bar = ProgressBar::new(inputs.len() as _);
        let mut samples = vec![];
        for s in inputs {
            let mm = mmm.iter().find(|mm| mm.kernel_name() == s.kernel).unwrap();
            let y = measure_add_mat_mul(&bencher, *mm, s.m, s.k, s.n);
            samples.push((s.clone(), y));
            progress_bar.inc();
        }
        progress_bar.finish();
        Dataset(samples)
    }

    pub fn save(&self, filename: &str) {
        let mut f = std::fs::File::create(filename).unwrap();
        for (s, y) in &self.0 {
            writeln!(&mut f, "{} {} {} {} {} {} {}", s.kernel, s.mr, s.nr, s.m, s.k, s.n, y)
                .unwrap();
        }
    }

    #[allow(dead_code)]
    pub fn load(filename: &str) -> Dataset {
        let samples = std::fs::read_to_string(filename)
            .unwrap()
            .lines()
            .map(|l| {
                let (kernel, mr, nr, m, k, n, y) = scan_fmt::scan_fmt!(
                    l,
                    "{} {} {} {} {} {} {}",
                    String,
                    usize,
                    usize,
                    usize,
                    usize,
                    usize,
                    f64
                )
                .unwrap();
                (Sample { kernel, mr, nr, m, k, n }, y)
            })
            .collect();
        Dataset(samples)
    }
}

fn display_comparison(m: usize, k: usize, n: usize, alts: &[(&str, f64)], choice: Option<&str>) {
    alts.iter().sorted_by(|a, b| order_f(&a.1, &b.1)).enumerate().for_each(|(ix, (s, t))| {
        let line = format!(
            "{:30} truth: {:9.03} us / {:9.03} GFLops",
            s,
            t * 1e6,
            (m * k * n) as f64 / t / 1e9,
        );
        if Some(*s) == choice {
            if ix == 0 {
                println!("{}", ansi_term::Color::Green.bold().paint(line));
            } else {
                println!("{}", ansi_term::Color::Red.bold().paint(line));
            }
        } else {
            println!("{}", line);
        }
    });
}

fn main() {
    use clap::*;

    let parser = App::new("tract-linalg-cost-model")
        .arg(
            Arg::new("bench-time-target")
                .long("bench-time-target")
                .default_value("0.1")
                .help("Target time for chunk sizing"),
        )
        .arg(
            Arg::new("chunk-time-target")
                .long("chunk-time-target")
                .default_value("0.01")
                .help("Target time for chunk sizing"),
        )
        .arg(
            Arg::new("chunks-min-count")
                .long("chunks-min-count")
                .default_value("100")
                .help("Minimum number of chunks"),
        )
        .arg(
            Arg::new("chunks-max-count")
                .long("chunks-max-count")
                .default_value("10000")
                .help("Minimum number of chunks"),
        )
        .subcommand(App::new("list-models"))
        .subcommand(
            App::new("time")
                .arg(Arg::new("mm").long("mm").help("Filter kernels").takes_value(true))
                .arg(Arg::new("m"))
                .arg(Arg::new("k"))
                .arg(Arg::new("n")),
        )
        .subcommand(
            App::new("ds")
                .arg(Arg::new("mm").long("mm").help("Filter kernels").takes_value(true))
                .arg(
                    Arg::new("m")
                        .short('m')
                        .help("Max m value")
                        .takes_value(true)
                        .default_value("512"),
                )
                .arg(
                    Arg::new("k")
                        .short('k')
                        .help("Max k value")
                        .takes_value(true)
                        .default_value("512"),
                )
                .arg(
                    Arg::new("n")
                        .short('n')
                        .help("Max n value")
                        .takes_value(true)
                        .default_value("512"),
                )
                .arg(
                    Arg::new("mkn")
                        .long("mkn")
                        .help("Max m*k*n value")
                        .takes_value(true)
                        .default_value("4194304"),
                )
                .arg(
                    Arg::new("size")
                        .short('s')
                        .long("size")
                        .help("Sample size (total)")
                        .takes_value(true)
                        .default_value("128"),
                )
                .arg(
                    Arg::new("strat")
                        .long("strat")
                        .help("Strategy for sampling")
                        .takes_value(true)
                        .possible_values(["smart", "random"])
                        .default_value("random"),
                )
                .arg(Arg::new("name").required(true)),
        );

    let matches = parser.get_matches();

    let bencher = Bencher {
        bench_time_target: Duration::from_secs_f64(
            matches.value_of_t("bench-time-target").unwrap(),
        ),
        chunk_time_target: Duration::from_secs_f64(
            matches.value_of_t("chunk-time-target").unwrap(),
        ),
        chunks_min_count: matches.value_of_t("chunks-min-count").unwrap(),
        chunks_max_count: matches.value_of_t("chunks-max-count").unwrap(),
    };

    let impls = tract_linalg::ops().mmm_f32_impls().iter().collect_vec();
    let mmms: Vec<&dyn MatMatMul> = impls.iter().map(|p| &***p).collect_vec();
    match matches.subcommand() {
        Some(("list-models", _sub)) => {
            for mmm in mmms {
                println!("{}", mmm.kernel_name());
            }
        }
        Some(("ds", sub)) => {
            let mut mmms = mmms.clone();
            if let Some(mm) = sub.value_of("mm") {
                mmms.retain(|m| m.kernel_name().contains(mm));
            }
            let inputs = match sub.value_of("strat").unwrap() {
                "smart" => Dataset::smart_sample(&*mmms),
                "random" => Dataset::allkernels_random_sample(
                    &*mmms,
                    sub.value_of_t("size").unwrap(),
                    sub.value_of_t("m").unwrap(),
                    sub.value_of_t("k").unwrap(),
                    sub.value_of_t("n").unwrap(),
                    sub.value_of_t("mkn").unwrap(),
                ),
                _ => unreachable!(),
            };
            Dataset::make_dataset(&bencher, inputs, &mmms).save(sub.value_of("name").unwrap());
        }
        Some(("time", sub)) => {
            let mut mmms = impls.clone();
            if let Some(mm) = sub.value_of("mm") {
                mmms.retain(|m| m.kernel_name().contains(mm));
            }
            let m: usize = sub.value_of("m").unwrap().parse().unwrap();
            let k: usize = sub.value_of("k").unwrap().parse().unwrap();
            let n: usize = sub.value_of("n").unwrap().parse().unwrap();
            let mut alts = vec![];
            for mm in &mmms {
                let y = measure_add_mat_mul(&bencher, &***mm, m, k, n);
                alts.push((mm.kernel_name(), y));
            }
            display_comparison(m, k, n, &*alts, None);
        }
        _ => panic!(),
    };
}
