use linregress::fit_low_level_regression_model_without_statistics;
use tract_data::internal::*;
use tract_linalg::{
    frame::MatMatMul,
    generic,
    mmm::{self, FusedSpec},
};

use rand::Rng;
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
        let mut data = vec![];
        let mut append = |m: usize, k: usize, n: usize| {
            let y = measure_add_mat_mul(mm, dt, m, k, n);
            data.push((m, k, n, y))
        };
        for _ in 0..10 {
            let m: usize = mm.mr() * rng.gen_range(1..=14);
            let k: usize = rng.gen_range(0..128);
            let n: usize = mm.nr() * rng.gen_range(1..=4);
            append(m, 64, 64);
        }
        for _ in 0..10 {
            let m: usize = mm.mr() * rng.gen_range(1..=16);
            let k: usize = rng.gen_range(0..128);
            let n: usize = mm.nr() * rng.gen_range(1..=4);
            append(m, 64, 64);
        }
        for m in 1..(4 * mm.mr()).min(32) {
            append(m, 64, 64);
        }
        for _ in 0..20 {
            let m: usize = rng.gen_range(1..16 * mm.mr());
            let k: usize = rng.gen_range(0..128);
            let n: usize = rng.gen_range(1..4 * mm.nr());
            append(m, 64, 64);
        }
        Dataset(data)
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
        _k: usize,
        n: usize,
    ) -> impl IntoIterator<Item = f64> + fmt::Debug {
        let rows = m.divceil(mr);
        let cols = 1; // n.divceil(nr);
        [
            // rows, cols
            //            m.divceil(mr) as f64,
            //            n.divceil(nr) as f64,
            //            // tiles
            (rows * cols) as f64,
            ((rows == 1) as usize) as f64,
            ((rows == 1) as usize * cols) as f64,
            (rows * rows * cols) as f64,
            //            // partial tiles right
            //            (m.divceil(mr) * ((n % nr) != 0) as usize) as f64,
            //            (m.divceil(mr) * (n % nr) as usize) as f64,
            //            // partial tiles down
            (cols * ((m % mr) != 0) as usize) as f64,
            (cols * (m % mr) as usize) as f64,
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
        let mut model =
            fit_low_level_regression_model_without_statistics(&*data, count, data.len() / count)
                .unwrap();
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
            "{:4} {:4} {:4}  pred:{:8.03} us truth:{:8.03} us {:5.2}%",
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
