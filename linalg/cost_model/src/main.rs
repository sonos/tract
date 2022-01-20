use linregress::fit_low_level_regression_model;
use pbr::ProgressBar;
use tract_data::internal::*;
use tract_linalg::{
    frame::MatMatMul,
    generic,
    mmm::{self, FusedSpec},
};

use rand::{prelude::SliceRandom, Rng};
use tract_itertools::Itertools;
use DatumType::F32;

use tract_linalg::frame::mmm::cost_model::Model;

#[path = "../../benches/nano.rs"]
mod nano;
use nano::*;

lazy_static::lazy_static! {
    static ref RUIN_CACHE:f64 = run_bench(|| ruin_cache());
}

pub fn ruin_cache() {
    let _a = (0..1000000).collect::<Vec<i32>>();
}

fn measure_add_mat_mul(mm: &dyn MatMatMul, dt: DatumType, m: usize, k: usize, n: usize) -> f64 {
    let pa = Tensor::zero_aligned_dt(dt, &[mm.a_pack(k).len(m)], mm.a_pack(k).alignment()).unwrap();
    let pb = Tensor::zero_aligned_dt(dt, &[mm.b_pack(k).len(n)], mm.b_pack(k).alignment()).unwrap();
    let pc = Tensor::zero_dt(dt, &[m, n]).unwrap();
    unsafe {
        let pa = mm.a_packed(dt.size_of(), k).wrap(&pa.view());
        let pb = mm.b_packed(dt.size_of(), k).wrap(&pb.view());
        let pc = mm.c_view_with_axis(0, 1).wrap(&pc.view());
        let mut scratch = mm.allocate_scratch_space();
        let time = run_bench(|| {
            ruin_cache();
            mm.run_with_scratch_space(
                m,
                n,
                scratch.as_mut(),
                &[FusedSpec::AddMatMul { a: pa, b: pb.clone(), k }, FusedSpec::Store(pc)],
            )
            .unwrap();
        });
        time - *RUIN_CACHE
    }
}

struct Dataset(Vec<(String, usize, usize, usize, f64)>);

impl Dataset {
    pub fn make_dataset(mmm: &[impl AsRef<dyn MatMatMul>], dt: DatumType) -> Dataset {
        let mut rng = rand::thread_rng();
        let mut data = vec![];
        for mm in mmm {
            let mm = mm.as_ref();
            let mut samples = vec![];
            for _ in 0..30 {
                let m: usize = rng.gen_range(1..=2 * mm.mr());
                let k: usize = rng.gen_range(1..=2 * (mm.mr() + mm.nr()));
                let n: usize = mm.nr() * rng.gen_range(1..=2);
                samples.push((m, k, n));
            }
            for _ in 0..30 {
                let m: usize = mm.mr() * rng.gen_range(1..=2);
                let k: usize = rng.gen_range(1..=2 * (mm.mr() + mm.nr()));
                let n: usize = rng.gen_range(1..=2 * mm.nr());
                samples.push((m, k, n));
            }
            for _ in 0..30 {
                let m: usize = rng.gen_range(1..=2 * mm.mr());
                let k: usize = rng.gen_range(1..=2 * (mm.mr() + mm.nr()));
                let n: usize = rng.gen_range(1..=2 * mm.nr());
                samples.push((m, k, n));
            }
            samples.shuffle(&mut rng);
            let mut progress_bar = ProgressBar::new(samples.len() as _);
            println!("Sampling: `{}'", mm.kernel_name());
            for ix in 0..samples.len() {
                let (m, k, n) = samples[ix];
                let y = measure_add_mat_mul(mm, dt, m, k, n);
                data.push((mm.kernel_name().to_string(), m, k, n, y));
                progress_bar.inc();
            }
            progress_bar.finish();
        }
        Dataset(data)
    }

    pub fn save(&self, filename: &str) {
        use std::io::Write;
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

fn train(ds: &Dataset, mm: &dyn MatMatMul) -> Model {
    let mut data = vec![];
    let mut count = 0;
    for (s, m, k, n, y) in &ds.0 {
        if mm.kernel_name() == s {
            data.push(*y);
            data.push(1.0);
            data.extend(Model::features(mm.mr(), mm.nr(), *m, *k, *n));
            count += 1;
        }
    }
    let model = fit_low_level_regression_model(&*data, count, data.len() / count).unwrap();
    dbg!(&model);
    let mut model = model.parameters;
    Model { mr: mm.mr(), nr: mm.nr(), intercept: model.remove(0), coef: model }
}

fn compare(model: &Model, m: usize, k: usize, n: usize, t: f64) {
    let prediction = model.predict(m, k, n);
    let ratio_for_color = ((prediction - t).abs() / t * 50.).min(1.);
    let color = colorous::RED_YELLOW_GREEN.eval_continuous(1.0 - ratio_for_color);
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

fn eval(model: &Model, name: &str, ds: &Dataset) {
    for (_, m, k, n, y) in
        ds.0.iter()
            .filter(|p| p.0 == name)
            .sorted_by_key(|p| (p.1, p.2, p.3, (p.4 * 1e12) as usize))
    {
        compare(model, *m, *k, *n, *y);
    }
}

fn main() {
    use clap::*;

    let parser = App::new("tract-linalg-cost-model")
        .subcommand(App::new("list-models"))
        .subcommand(App::new("ds").arg(Arg::new("name").required(true)))
        .subcommand(App::new("train").arg(Arg::new("train").required(true)).arg(Arg::new("eval")));

    let matches = parser.get_matches();

    let mm = mmm::MatMatMulImpl::<generic::GenericMmm4x4<f32, f32, f32>, f32>::new();
    //    let mm = mmm::MatMatMulImpl::<tract_linalg::arm64::MatMatMulF32x12x8, f32>::new();
    match matches.subcommand() {
        Some(("list-models", _sub)) => {
            for mmm in tract_linalg::ops().mmm_f32_impls() {
                println!("{}", mmm.kernel_name());
            }
        }
        Some(("ds", sub)) => {
            Dataset::make_dataset(&tract_linalg::ops().mmm_f32_impls(), F32)
                .save(sub.value_of("name").unwrap());
        }
        Some(("train", sub)) => {
            let ds = Dataset::load(sub.value_of("train").unwrap());
            let model = train(&ds, &mm);
            if let Some(ds2) = sub.value_of("eval") {
                let ds2 = Dataset::load(ds2);
                eval(&model, mm.kernel_name(), &ds2);
            }
        }
        _ => panic!(),
    };
}
