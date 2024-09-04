use pbr::ProgressBar;
use tract_data::internal::*;
use tract_linalg::{frame::MatMatMul, mmm::FusedSpec};

use rand::prelude::*;
use std::io::Write;
use std::ops::Range;
use std::str::FromStr;
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
    probe: Option<readings_probe::Probe>,
}

impl Bencher {
    fn black_box<T>(dummy: T) -> T {
        unsafe {
            let ret = std::ptr::read_volatile(&dummy);
            std::mem::forget(dummy);
            ret
        }
    }

    pub fn run_bench<T, I, P: FnMut() -> Vec<I>, F: FnMut(&mut I) -> T>(
        &self,
        mut prep: P,
        mut f: F,
    ) -> f64 {
        let mut inputs = prep();
        let islen = inputs.len();
        Self::black_box(f(&mut inputs[0]));
        let start = Instant::now();
        Self::black_box(f(&mut inputs[1.min(islen - 1)]));
        let once = start.elapsed();
        //   dbg!(once);
        let evaled = if once < Duration::from_millis(1) {
            let start = Instant::now();
            for i in 0..1000 {
                Self::black_box(f(&mut inputs[i % islen]));
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
        let mut input = 0;
        for i in 0..chunks {
            let start = Instant::now();
            for _ in 0..chunk {
                Self::black_box(f(&mut inputs[input]));
                input += 1;
                if input == inputs.len() {
                    input = 0
                }
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

fn measure_add_mat_mul(bencher: &Bencher, mm: &dyn MatMatMul, m: usize, k: usize, n: usize) -> f64 {
    let dt = mm.internal_type();
    if let Some(probe) = &bencher.probe {
        probe.log_event(&format!("start_{},{},{}", m, k, n)).unwrap();
    }
    let a = Tensor::zero_dt(dt, &[m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[k, n]).unwrap();
    unsafe {
        let time = bencher.run_bench(
            || {
                let pb_size = 4 * (m * k + m * n + k * n);
                let inputs = (10_000_000 / pb_size).max(1);
                (0..inputs)
                    .map(|_| {
                        let (packed_a, packed_b) = mm.packings()[0];
                        let pa = packed_a.prepare_tensor(&a, 1, 0).unwrap();
                        let pb = packed_b.prepare_tensor(&b, 0, 1).unwrap();
                        let c = Tensor::zero_dt(dt, &[m, n]).unwrap();
                        let pc = mm.c_view(0, 1).wrap(&c.view());
                        let scratch = mm.allocate_scratch_space();
                        (scratch, c, pa, pb, pc)
                    })
                    .collect()
            },
            #[allow(unused_mut)] // not sure why the warning pops
            |(scratch, _c, pa, pb, mut pc)| {
                mm.run_with_scratch_space(
                    m,
                    n,
                    scratch.as_mut(),
                    &[
                        FusedSpec::AddMatMul { a: &**pa, b: &**pb, packing: 0 },
                        FusedSpec::Store(pc),
                    ],
                )
                .unwrap();
            },
        );
        time
    }
}

#[derive(Clone, Debug)]
enum SamplingStrategy {
    Random(Range<usize>),
    Fixed(Vec<usize>),
}

impl SamplingStrategy {
    fn sample(&self) -> Vec<usize> {
        use SamplingStrategy::*;
        let mut rng = thread_rng();
        match self {
            Random(range) => vec![rng.gen_range(range.clone())],
            Fixed(v) => v.clone(),
        }
    }
}

impl FromStr for SamplingStrategy {
    type Err = TractError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.contains("-") {
            let (min, max) = s.split_once("-").unwrap();
            Ok(SamplingStrategy::Random(
                min.parse::<usize>().unwrap()..max.parse::<usize>().unwrap() + 1,
            ))
        } else {
            Ok(SamplingStrategy::Fixed(s.split(",").map(|s| s.parse::<usize>().unwrap()).collect()))
        }
    }
}

#[derive(Clone, Debug)]
struct Sample {
    kernel: String,
    mr: usize,
    nr: usize,
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Clone, Debug)]
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
        m: SamplingStrategy,
        k: SamplingStrategy,
        n: SamplingStrategy,
        max_mkn: usize,
    ) -> Vec<Sample> {
        let mut inputs = vec![];
        for _ in 0..size {
            let ms = m.sample();
            let ks = k.sample();
            let ns = n.sample();
            for m in ms {
                for &k in &ks {
                    for &n in &ns {
                        for mm in mmm {
                            if max_mkn < m * k * n {
                                continue;
                            }
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
}

fn display_comparison(
    m: usize,
    k: usize,
    n: usize,
    alts: &[(impl AsRef<str>, f64)],
    choice: Option<&str>,
) {
    alts.iter().sorted_by(|a, b| order_f(&a.1, &b.1)).enumerate().for_each(|(ix, (s, t))| {
        let s = s.as_ref();
        let line = format!(
            "{:30} truth: {:9.03} us / {:9.03} GFLops",
            s,
            t * 1e6,
            (m * k * n) as f64 / t / 1e9,
        );
        if Some(s) == choice {
            if ix == 0 {
                println!("{}", nu_ansi_term::Color::Green.bold().paint(line));
            } else {
                println!("{}", nu_ansi_term::Color::Red.bold().paint(line));
            }
        } else {
            println!("{}", line);
        }
    });
}

fn main() {
    use clap::*;

    let probe = if let Ok(file) = std::fs::File::create("readings.out") {
        let mut probe = readings_probe::Probe::new(file).unwrap();
        probe.spawn_heartbeat(std::time::Duration::from_millis(1000)).unwrap();
        Some(probe)
    } else {
        None
    };

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
                        .help("m values: 1-512 or 1,16,32")
                        .takes_value(true)
                        .default_value("1-512"),
                )
                .arg(
                    Arg::new("k")
                        .short('k')
                        .help("k values: 1-512 or 1,16,32")
                        .takes_value(true)
                        .default_value("1-512"),
                )
                .arg(
                    Arg::new("n")
                        .short('n')
                        .help("m values: 1-512 or 1,16,32")
                        .takes_value(true)
                        .default_value("1-512"),
                )
                .arg(
                    Arg::new("mkn")
                        .long("mkn")
                        .help("Max m*k*n value")
                        .takes_value(true)
                        .default_value("9999999999"),
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
        probe,
    };

    let impls = tract_linalg::ops().mmm_impls().iter().collect_vec();
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
