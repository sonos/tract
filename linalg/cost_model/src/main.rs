use linregress::{
    fit_low_level_regression_model, fit_low_level_regression_model_without_statistics,
};
use pbr::ProgressBar;
use tract_data::internal::*;
use tract_linalg::{
    frame::MatMatMul,
    generic,
    mmm::{self, FusedSpec},
};

use rand::{prelude::SliceRandom, Rng};
use std::fmt;
use tract_itertools::Itertools;
use DatumType::F32;

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

struct Dataset(Vec<(usize, usize, usize, f64)>);

impl Dataset {
    pub fn make_dataset(mm: &dyn MatMatMul, dt: DatumType) -> Dataset {
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        /*
        for _ in 0..100 {
            let m: usize = mm.mr() * rng.gen_range(1..=6);
            let k: usize = rng.gen_range(0..256);
            let n: usize = mm.nr() * rng.gen_range(1..=6);
            samples.push((m, k, n, 0.));
        }
        for _ in 0..100 {
            let m: usize = mm.mr() * rng.gen_range(1..=16);
            let k: usize = rng.gen_range(0..256);
            let n: usize = mm.nr() * rng.gen_range(1..=16);
            samples.push((m, k, n, 0.));
        }
        for x in 1..=(4 * mm.mr()).min(32) {
            samples.push((x, 64, 64, 0.));
            samples.push((64, x, 64, 0.));
            samples.push((64, 64, x, 0.));
        }
        */
        /*
        for m in 1..=(2 * mm.mr()) {
            for n in 1..=(2 * mm.nr()) {
                for k in 1..16 {
                    samples.push((m, k, n, 0.));
                }
            }
        }
        */
        /*
        let mut k = 0;
        loop {
            samples.push((64, k, 64, 0.));
            k += (k / 16).max(1);
            if k > 4096 {
                break;
            }
        }
        */
        /*
        for k in 0..32 {
            samples.push((64, k, 64, 0.));
        }
        */
        for _ in 0..100 {
            let m: usize = rng.gen_range(1..=2 * mm.mr());
            let k: usize = rng.gen_range(1..=2 * (mm.mr() + mm.nr()));
            let n: usize = mm.nr() * rng.gen_range(1..=2);
            samples.push((m, k, n, 0.));
        }
        for _ in 0..100 {
            let m: usize = mm.mr() * rng.gen_range(1..=2);
            let k: usize = rng.gen_range(1..=2 * (mm.mr() + mm.nr()));
            let n: usize = rng.gen_range(1..=2 * mm.nr());
            samples.push((m, k, n, 0.));
        }
        for _ in 0..100 {
            let m: usize = rng.gen_range(1..=2 * mm.mr());
            let k: usize = rng.gen_range(1..=2 * (mm.mr() + mm.nr()));
            let n: usize = rng.gen_range(1..=2 * mm.nr());
            samples.push((m, k, n, 0.));
        }
        /*
        for _ in 0..20 {
            let m: usize = rng.gen_range(1..128 * mm.mr());
            let k: usize = rng.gen_range(0..128);
            let n: usize = rng.gen_range(1..128 * mm.nr());
            samples.push((m, k, n, 0.));
        }
        */
        samples.shuffle(&mut rng);
        let mut progress_bar = ProgressBar::new(samples.len() as _);
        for ix in 0..samples.len() {
            let (m, k, n, _) = samples[ix];
            let y = measure_add_mat_mul(mm, dt, m, k, n);
            samples[ix].3 = y;
            progress_bar.inc();
        }
        progress_bar.finish();
        Dataset(samples)
    }

    pub fn save(&self, filename: &str) {
        use std::io::Write;
        let mut f = std::fs::File::create(filename).unwrap();
        for &(m, k, n, y) in &self.0 {
            writeln!(&mut f, "{m} {k} {n} {y}").unwrap();
        }
    }

    pub fn load(filename: &str) -> Dataset {
        let d = std::fs::read_to_string(filename)
            .unwrap()
            .lines()
            .map(|l| scan_fmt::scan_fmt!(l, "{} {} {} {}", usize, usize, usize, f64).unwrap())
            .collect();
        Dataset(d)
    }
}

struct Model {
    mr: usize,
    nr: usize,
    intercept: f64,
    coef: Vec<f64>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Model: intercept: {:.3e} coefs: {}",
            self.intercept,
            self.coef.iter().map(|x| format!("{:.3e}", x)).join(", ")
        )
    }
}

impl Model {
    pub fn features(
        mr: usize,
        nr: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> impl IntoIterator<Item = f64> + fmt::Debug {
        let rows = m.divceil(mr);
        let cols = n.divceil(nr);
        [
            k as f64,
            (k * k) as f64,
            (rows * cols) as f64,
            (rows * cols * k) as f64,
            (rows * rows * cols * cols) as f64,
            (rows * rows * cols * cols * k) as f64,
            cols as f64,
            rows as f64,
            (rows * rows) as f64,
            (cols * cols) as f64,
            // rows, cols
            //            m.divceil(mr) as f64,
            //            n.divceil(nr) as f64,
            //            // tiles
            ((m == 1) as usize) as f64,
            ((rows == 1) as usize) as f64,
            ((cols == 1) as usize) as f64,
            ((rows == 1) as usize * cols) as f64,
            ((cols == 1) as usize * rows) as f64,
            (rows * rows * cols) as f64,
            (rows * cols * cols) as f64,
            //            // partial tiles right
            //            (m.divceil(mr) * ((n % nr) != 0) as usize) as f64,
            //            (m.divceil(mr) * (n % nr) as usize) as f64,
            //            // partial tiles down
            (cols * ((m % mr) != 0) as usize) as f64,
            (rows * ((n % nr) != 0) as usize) as f64,
            (cols * (m % mr) as usize) as f64,
            (rows * (n % nr) as usize) as f64,
            (cols * rows * (m % mr) as usize) as f64,
            (cols * rows * (n % nr) as usize) as f64,
            (cols * cols * (m % mr) as usize) as f64,
            (rows * rows * (n % nr) as usize) as f64,
            (cols * rows * ((n % nr) != 0) as usize * ((m % mr) != 0) as usize) as f64,
            (((n % nr) != 0) as usize * ((m % mr) != 0) as usize) as f64,
            ((n % nr) * (m % mr)) as f64,
            (cols * cols * rows * rows * ((n % nr) != 0) as usize * ((m % mr) != 0) as usize)
                as f64,
        ]
    }

    pub fn train(ds: &Dataset, mm: &dyn MatMatMul) -> Model {
        let mut data = vec![];
        let mut count = 0;
        for &(m, k, n, y) in &ds.0 {
            data.push(y);
            data.push(1.0);
            data.extend(Self::features(mm.mr(), mm.nr(), m, k, n));
            count += 1;
        }
        let model = fit_low_level_regression_model(&*data, count, data.len() / count).unwrap();
        dbg!(&model);
        let mut model = model.parameters;
        Model { mr: mm.mr(), nr: mm.nr(), intercept: model.remove(0), coef: model }
    }

    pub fn predict(&self, m: usize, k: usize, n: usize) -> f64 {
        let feats = Self::features(self.mr, self.nr, m, k, n);
        self.intercept + self.coef.iter().zip(feats).map(|(c, x)| c * x).sum::<f64>()
    }

    fn compare(&self, m: usize, k: usize, n: usize, t: f64) {
        let prediction = self.predict(m, k, n);
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

    fn eval(&self, ds: &Dataset) {
        for &(m, k, n, y) in ds.0.iter().sorted_by_key(|p| (p.0, p.1, p.2, (p.3 * 1e12) as usize)) {
            self.compare(m, k, n, y);
        }
    }
}

fn main() {
    use clap::*;

    let parser = App::new("tract-linalg-cost-model")
        .subcommand(App::new("ds").arg(Arg::new("name").required(true)))
        .subcommand(App::new("train").arg(Arg::new("train").required(true)).arg(Arg::new("eval")));

    let matches = parser.get_matches();

    let mm = mmm::MatMatMulImpl::<generic::GenericMmm4x4<f32, f32, f32>, f32>::new();
//    let mm = mmm::MatMatMulImpl::<tract_linalg::arm64::MatMatMulF32x12x8, f32>::new();
    match matches.subcommand() {
        Some(("ds", sub)) => {
            Dataset::make_dataset(&mm, F32).save(sub.value_of("name").unwrap());
        }
        Some(("train", sub)) => {
            let ds = Dataset::load(sub.value_of("train").unwrap());
            let model = Model::train(&ds, &mm);
            if let Some(eval) = sub.value_of("eval") {
                let ds = Dataset::load(eval);
                model.eval(&ds);
            }
        }
        _ => panic!(),
    };
}
