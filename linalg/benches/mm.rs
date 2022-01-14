use tract_data::internal::*;
use tract_linalg::{
    frame::MatMatMul,
    generic,
    mmm::{self, FusedSpec},
};

use std::fmt;
use DatumType::F32;

mod nano;
mod utils;
use nano::*;

fn measure_add_mat_mul(mm: &dyn MatMatMul, dt: DatumType, m: usize, k: usize, n: usize) -> f64 {
    let pa = Tensor::zero_aligned_dt(dt, &[mm.a_pack(k).len(m)], mm.a_pack(k).alignment()).unwrap();
    let pb = Tensor::zero_aligned_dt(dt, &[mm.b_pack(k).len(n)], mm.b_pack(k).alignment()).unwrap();
    unsafe {
        let pa = mm.a_packed(dt.size_of(), k).wrap(&pa.view());
        let pb = mm.b_packed(dt.size_of(), k).wrap(&pb.view());
        let mut scratch = mm.allocate_scratch_space();
        let ruin_cache = run_bench(|| utils::ruin_cache());
        let time = run_bench(|| {
            utils::ruin_cache();
            mm.run_with_scratch_space(
                m,
                n,
                scratch.as_mut(),
                &[FusedSpec::AddMatMul { a: pa, b: pb.clone(), k }],
            )
            .unwrap();
        });
        time - ruin_cache
    }
}

struct Model {
    mr: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model: alpha={:.3e} beta={:.3e} gamma={:.3e}", self.alpha, self.beta, self.gamma)
    }
}

impl Model {
    pub fn new(mm: &dyn MatMatMul, dt: DatumType) -> Model {
        let at_16mr = measure_add_mat_mul(mm, dt, 16 * mm.mr(), 64, 64);
        let at_mr = measure_add_mat_mul(mm, dt, mm.mr(), 64, 64);
        let alpha = (at_16mr - at_mr) / 15.;
        let beta = at_mr - alpha;
        let at_mr_plus_one = measure_add_mat_mul(mm, dt, mm.mr() + 1, 64, 64);
        let gamma = at_mr_plus_one - at_mr;
        Model { mr: mm.mr(), alpha, beta, gamma }
    }

    pub fn predict(&self, m: usize, k: usize, n: usize) -> f64 {
        assert_eq!(k, 64);
        assert_eq!(n, 64);
        self.alpha * (m / self.mr) as f64
            + self.beta
            + if m % self.mr != 0 { self.gamma } else { 0. }
    }
}

fn compare(mmm: &dyn MatMatMul, model: &Model, dt: DatumType, m: usize, k: usize, n: usize) {
    let t = measure_add_mat_mul(mmm, dt, m, k, n);
    let prediction = model.predict(m, k, n);
    println!("{:3}   {:6.03}us   {:5.2}%", m, t * 1e6, (prediction - t) / t * 100.);
}

fn modelize(mmm: &dyn MatMatMul) {
    let model = Model::new(&*mmm, DatumType::F32);
    eprintln!("{}: {:?}", mmm, model);
    let k = 64;
    let n = 64;
    for i in 1..16 {
        compare(&*mmm, &model, F32, i * mmm.mr(), k, n);
    }
    for m in 1..4 * mmm.mr() {
        compare(&*mmm, &model, F32, m, k, n);
    }
}

fn main() {
    modelize(&mmm::MatMatMulImpl::<generic::GenericMmm4x4<f32, f32, f32>, f32>::new());
    println!();
    modelize(&mmm::MatMatMulImpl::<tract_linalg::x86_64_fma::mmm::MatMatMulF32x16x6, f32>::new());
}
