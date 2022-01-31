use linregress::fit_low_level_regression_model;
use pbr::ProgressBar;
use randomforest::criterion::Mse;
use randomforest::table::TableBuilder;
use randomforest::RandomForestRegressorOptions;
use tract_data::internal::*;
use tract_linalg::{frame::MatMatMul, mmm::FusedSpec};

use rand::prelude::*;
use std::io::Write;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use tract_itertools::Itertools;

use tract_linalg::frame::mmm::cost_model::CostModel;

pub fn ruin_cache() {
    let _a = (0..1_000_000).collect::<Vec<i32>>();
}

fn order_f64(&a: &f64, &b: &f64) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Greater
    }
}

fn order_f32(&a: &f32, &b: &f32) -> std::cmp::Ordering {
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

    pub fn run_bench<T, F: FnMut() -> T>(&self, mut f: F) -> f64 {
        Self::black_box(f());
        let start = Instant::now();
        Self::black_box(f());
        let once = start.elapsed();
        //   dbg!(once);
        let evaled = if once < Duration::from_millis(1) {
            let start = Instant::now();
            for _ in 0..1000 {
                Self::black_box(f());
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
            let start = Instant::now();
            for _ in 0..chunk {
                Self::black_box(f());
            }
            let time = start.elapsed().as_secs_f64();
            measures[i] = time / chunk as f64;
        }
        measures.sort_by(order_f64);
        let q1 = measures[chunks / 4];
        let q3 = measures[chunks - chunks / 4];
        let iq = q3 - q1;
        //    measures.retain(|&x| x >= q1 && x <= q3);
        let epsilon = iq * 2. / (q3 + q1);
        eprintln!("evaled: {} chunk:{} chunks: {} epsilon: {:.3e}", evaled, chunk, chunks, epsilon);
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
    ruin_cache_time: f64,
    mm: &dyn MatMatMul,
    m: usize,
    k: usize,
    n: usize,
) -> f64 {
    let dt = mm.internal_type();
    let pa = Tensor::zero_aligned_dt(dt, &[mm.a_pack().len(k, m)], mm.a_pack().alignment()).unwrap();
    let pb = Tensor::zero_aligned_dt(dt, &[mm.b_pack().len(k, n)], mm.b_pack().alignment()).unwrap();
    let pc = Tensor::zero_dt(dt, &[m, n]).unwrap();
    unsafe {
        let pa = mm.a_packed(dt.size_of(), k).wrap(&pa.view());
        let pb = mm.b_packed(dt.size_of(), k).wrap(&pb.view()).unwrap();
        let pc = mm.c_view(0, 1).wrap(&pc.view());
        let mut scratch = mm.allocate_scratch_space();
        let time = bencher.run_bench(|| {
            ruin_cache();
            mm.run_with_scratch_space(
                m,
                n,
                scratch.as_mut(),
                &[FusedSpec::AddMatMul { a: pa, b: pb.clone(), k }, FusedSpec::Store(pc)],
            )
            .unwrap();
        });
        time - ruin_cache_time
    }
}

struct Dataset(Vec<(String, usize, usize, usize, f64)>);

impl Dataset {
    /*
    pub fn gen_inputs(
    wanted: usize,
    mm: &dyn MatMatMul,
    m: SamplingStrat,
    k: SamplingStrat,
    n: SamplingStrat,
    ) -> Vec<(String, usize, usize, usize)> {
    let mut samples = vec![];
    for _ in 0..wanted {
    let m = m.sample(mm.mr());
    let k = k.sample(mm.mr() + mm.nr());
    let n = n.sample(mm.nr());
    samples.push((mm.kernel_name().to_string(), m, k, n));
    }
    samples
    }
    */

    pub fn make_dataset(bencher: &Bencher, mmm: &[impl AsRef<dyn MatMatMul>]) -> Dataset {
        let ruin_cache_time = bencher.run_bench(|| ruin_cache());
        let mut rng = thread_rng();
        let mut inputs = vec![];
        for mm in mmm {
            let mm = mm.as_ref();
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
                        inputs.push((mm.kernel_name().to_string(), m, k, n));
                    }
                }
            }
        }
        inputs.shuffle(&mut rng);
        let mut progress_bar = ProgressBar::new(inputs.len() as _);
        let mut samples = vec![];
        for (s, m, k, n) in inputs {
            let mm = mmm.iter().find(|mm| mm.as_ref().kernel_name() == s).unwrap().as_ref();
            let y = measure_add_mat_mul(&bencher, ruin_cache_time, mm, m, k, n);
            samples.push((s, m, k, n, y));
            progress_bar.inc();
        }
        progress_bar.finish();
        Dataset(samples)
    }

    pub fn save(&self, filename: &str) {
        let mut f = std::fs::File::create(filename).unwrap();
        for (s, m, k, n, y) in &self.0 {
            writeln!(&mut f, "{} {} {} {} {}", s, m, k, n, y).unwrap();
        }
    }

    pub fn load(filename: &str) -> Dataset {
        let d = std::fs::read_to_string(filename)
            .unwrap()
            .lines()
            .map(|l| {
                scan_fmt::scan_fmt!(l, "{} {} {} {} {}", String, usize, usize, usize, f64).unwrap()
            })
            .collect();
        Dataset(d)
    }
}

fn train(ds: &Dataset, mm: impl AsRef<dyn MatMatMul>) -> CostModel {
    let mm = mm.as_ref();
    dbg!(&mm.kernel_name());
    let mut data = vec![];
    let mut count = 0;
    for (s, m, k, n, y) in &ds.0 {
        if mm.kernel_name() == s {
            data.push(*y);
            data.push(1.0);
            let rows = m.divceil(mm.mr());
            let cols = n.divceil(mm.nr());
            data.push((rows * cols * k) as f64);
            count += 1;
        }
    }
    let model = fit_low_level_regression_model(&*data, count, data.len() / count).unwrap();
    dbg!(&mm.kernel_name());
    dbg!(&model.rsquared);
    let mut model = model.parameters;
    let intercept = model.remove(0);
    let alpha = model.remove(0);

    let mut residuals = 0.;
    let mut table_builder = TableBuilder::new();
    for (s, m, k, n, y) in &ds.0 {
        if mm.kernel_name() == s {
            let rows = m.divceil(mm.mr());
            let cols = n.divceil(mm.nr());
            let mkn = rows * cols * k;
            let features: Vec<f64> =
                CostModel::features(mm.mr(), mm.nr(), *m, *k, *n).into_iter().collect();
            let residual = *y - mkn as f64 * alpha - intercept;
            residuals += (residual / y).powi(2);
            table_builder.add_row(&features, residual).unwrap();
            let rows = m.divceil(mm.mr());
            let cols = n.divceil(mm.nr());
            data.push((rows * cols * k) as f64);
            count += 1;
        }
    }
    let table = table_builder.build().unwrap();
    let forest = RandomForestRegressorOptions::new()
        .seed(0)
        .trees(NonZeroUsize::new(3).unwrap())
        .fit(Mse, table);
    let mut v = vec![];
    forest.serialize(&mut v).unwrap();

    let model = CostModel {
        mr: mm.mr(),
        nr: mm.nr(),
        intercept: intercept as f32,
        alpha: alpha as f32,
        forest,
    };

    let mut sqr = 0.;
    for (s, m, k, n, y) in &ds.0 {
        if mm.kernel_name() == s {
            sqr += ((model.predict(*m, *k, *n) - *y as f32) / *y as f32).powi(2);
        }
    }

    dbg!(v.len());
    dbg!(residuals / count as f64);
    dbg!(sqr / count as f32);
    model
}

fn compare(model: &CostModel, m: usize, k: usize, n: usize, t: f32) {
    let prediction = model.predict(m, k, n);
    let ratio_for_color = ((prediction - t).abs() / t * 50.).min(1.);
    let color = colorous::RED_YELLOW_GREEN.eval_continuous((1.0 - ratio_for_color) as _);
    let color = ansi_term::Color::RGB(color.r, color.g, color.b);
    let line = format!(
        "{:4} {:4} {:4}  pred: {:9.03} us truth: {:9.03} us {:5.2}%",
        m,
        k,
        n,
        prediction * 1e6,
        t * 1e6,
        (prediction - t) / t * 100.
    );
    println!("{}", color.bold().paint(line));
}

fn eval(model: &CostModel, name: &str, ds: &Dataset) {
    for (_, m, k, n, y) in
        ds.0.iter()
            .filter(|p| p.0 == name)
            .sorted_by_key(|p| (p.1, p.2, p.3, (p.4 * 1e12) as usize))
    {
        compare(model, *m, *k, *n, *y as f32);
    }
}

fn train_and_dump(impls: &[impl AsRef<dyn MatMatMul>], ds: &Dataset, writer: &mut dyn Write) {
    writeln!(writer, "use crate::frame::mmm::cost_model::CostModel;").unwrap();
    writeln!(writer, "pub fn models() -> Vec<(&'static str, CostModel)> {{").unwrap();
    writeln!(writer, "vec!(").unwrap();
    for mm in impls {
        let model = train(&ds, mm);
        let mm = mm.as_ref();
        writeln!(
            writer,
            "(\"{}\", CostModel {{ mr: {}, nr: {},",
            mm.kernel_name(),
            mm.mr(),
            mm.nr()
        )
        .unwrap();
        writeln!(writer, "alpha: {},", model.alpha).unwrap();
        writeln!(writer, "intercept: {},", model.intercept).unwrap();
        writeln!(writer, "}}),").unwrap();
    }
    writeln!(writer, ")}}").unwrap();
}

fn display_comaparison(m: usize, k: usize, n: usize, alts: &[(&str, f64, f64)]) {
    let best_choice = alts.iter().min_by(|a, b| order_f64(&a.2, &b.2)).unwrap();
    alts.iter().sorted_by(|a, b| order_f64(&a.1, &b.1)).enumerate().for_each(
                |(ix, (s, t, p))| {
                    let line = format!(
                        "{:30} pred: {:9.03} us / {:9.03} GFlops ; truth: {:9.03} us / {:9.03} GFLops ; diff: {:5.2}%",
                        s,
                        p * 1e6,
                        (m * k * n) as f64 / p / 1e9,
                        t * 1e6,
                        (m * k * n) as f64 / t / 1e9,
                        (p - t) / t * 100.,
                        );
                    if &best_choice.0 == s {
                        if ix == 0 {
                            println!("{}", ansi_term::Color::Green.bold().paint(line));
                        } else {
                            println!("{}", ansi_term::Color::Red.bold().paint(line));
                        }
                    } else {
                        println!("{}", line);
                    }
                },
                );
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
            App::new("e2e")
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .takes_value(true)
                        .help("Filename to write models to (in rust form)"),
                )
                .arg(Arg::new("ds").long("dataset").takes_value(true).help("Dataset to read")),
        )
        .subcommand(
            App::new("time")
                .arg(Arg::new("mm").long("mm").help("Filter kernels").takes_value(true))
                .arg(Arg::new("m"))
                .arg(Arg::new("k"))
                .arg(Arg::new("n")),
        )
        .subcommand(
            App::new("train-eval")
                .arg(
                    Arg::new("no-truth")
                        .long("no-truth")
                        .takes_value(false)
                        .help("Do not measure ground truth."),
                )
                .arg(Arg::new("train").required(true))
                .arg(Arg::new("m"))
                .arg(Arg::new("k"))
                .arg(Arg::new("n")),
        )
        .subcommand(
            App::new("ds")
                .arg(Arg::new("mm").long("mm").help("Filter kernels").takes_value(true))
                .arg(Arg::new("name").required(true)),
        )
        .subcommand(
            App::new("train")
                .arg(Arg::new("mm").long("mm").help("Filter kernels").takes_value(true))
                .arg(Arg::new("train").required(true))
                .arg(Arg::new("eval")),
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

    let impls = tract_linalg::ops().mmm_f32_impls().iter().map(|p| p.0.clone()).collect_vec();
    match matches.subcommand() {
        Some(("list-models", _sub)) => {
            for mmm in impls {
                println!("{}", mmm.kernel_name());
            }
        }
        Some(("ds", sub)) => {
            let mut mmms = impls.clone();
            if let Some(mm) = sub.value_of("mm") {
                mmms.retain(|m| m.kernel_name().contains(mm));
            }
            Dataset::make_dataset(&bencher, &mmms).save(sub.value_of("name").unwrap());
        }
        Some(("e2e", sub)) => {
            let ds = if let Some(ds) = sub.value_of("ds") {
                Dataset::load(ds)
            } else {
                Dataset::make_dataset(&bencher, &impls)
            };
            let mut writer: Box<dyn std::io::Write> = if let Some(filename) = sub.value_of("output")
            {
                Box::new(std::fs::File::create(filename).unwrap())
            } else {
                Box::new(std::io::stdout())
            };
            train_and_dump(&*impls, &ds, &mut writer);
        }
        Some(("train", sub)) => {
            let ds = Dataset::load(sub.value_of("train").unwrap());
            let mut mmms = impls.clone();
            if let Some(mm) = sub.value_of("mm") {
                mmms.retain(|m| m.kernel_name().contains(mm));
            }
            let models = ds.0.iter().map(|p| &p.0).unique();
            for mm in models {
                if let Some(mm) = impls.iter().find(|p| p.kernel_name() == mm) {
                    let model = train(&ds, &mm);
                    if let Some(ds2) = sub.value_of("eval") {
                        let ds2 = Dataset::load(ds2);
                        eval(&model, mm.kernel_name(), &ds2);
                    }
                }
            }
        }
        Some(("time", sub)) => {
            let ruin_cache_time = bencher.run_bench(|| ruin_cache());
            let mut mmms = impls.clone();
            if let Some(mm) = sub.value_of("mm") {
                mmms.retain(|m| m.kernel_name().contains(mm));
            }
            let m: usize = sub.value_of("m").unwrap().parse().unwrap();
            let k: usize = sub.value_of("k").unwrap().parse().unwrap();
            let n: usize = sub.value_of("n").unwrap().parse().unwrap();
            let mut alts = vec![];
            for (mm, model) in &mmms {
                let y = measure_add_mat_mul(&bencher, ruin_cache_time, &**mm, m, k, n);
                let predicted = model.as_ref().map(|model| model.predict(m, k, n));
                alts.push((mm.kernel_name(), y, predicted.unwrap_or(0.)));
            }
            display_comaparison(m, k, n, &alts);
        }
        Some(("train-eval", sub)) => {
            let ruin_cache_time = bencher.run_bench(|| ruin_cache());
            let ds = Dataset::load(sub.value_of("train").unwrap());
            let mut mmms = impls.clone();
            if let Some(mm) = sub.value_of("mm") {
                mmms.retain(|m| m.kernel_name().contains(mm));
            }
            let models = ds.0.iter().map(|p| &p.0).unique();
            let m: usize = sub.value_of("m").unwrap().parse().unwrap();
            let k: usize = sub.value_of("k").unwrap().parse().unwrap();
            let n: usize = sub.value_of("n").unwrap().parse().unwrap();
            let do_truth = !sub.is_present("no-truth");
            let mut alts = vec![];
            for mm in models {
                let mm = impls.iter().find(|p| p.kernel_name() == mm).unwrap();
                let model = train(&ds, &mm);
                let p = model.predict(m, k, n);
                let y = if do_truth {
                    measure_add_mat_mul(&bencher, ruin_cache_time, &**mm, m, k, n) as f32
                } else {
                    p
                };
                alts.push((mm.kernel_name(), y, p));
            }
            display_comaparison(m, k, n, &alts);
        }
        _ => panic!(),
    };
}
