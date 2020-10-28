#[macro_use]
extern crate criterion;
use criterion::Criterion;

use tract_data::internal::*;

pub fn vec(len: usize, align: usize) -> Tensor {
    unsafe {
        let mut tensor = Tensor::uninitialized_aligned::<f32>(&[len], align).unwrap();
        tensor.clear::<f32>();
        tensor
    }
}

fn pack_a(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("pack_a_{}x{}x{}", m, k, n), move |b| {
        let mm = (tract_linalg::ops().mmm_f32)(m, k, n);
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let mut pa = vec(mm.a_pack().len(), mm.a_pack().alignment());
        b.iter(move || unsafe { mm.a_pack().pack(&mut pa.view_mut(), &a.view(), false) })
    });
}

fn pack_b(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("pack_b_{}x{}x{}", m, k, n), move |be| unsafe {
        let mm = (tract_linalg::ops().mmm_f32)(m, k, n);
        let b = Tensor::zero::<f32>(&[k, n]).unwrap();
        let mut pb = Tensor::uninitialized_aligned::<f32>(&[n * k], 4).unwrap();
        be.iter(move || {
            mm.b_pack().pack(
                pb.as_ptr_mut_unchecked::<f32>(),
                b.as_ptr_unchecked::<f32>(),
                n as _,
                1,
            )
        })
    });
}

fn mat_mul_prepacked(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("mat_mul_prepacked_{}x{}x{}", m, k, n), move |be| {
        let mm = (tract_linalg::ops().mmm_f32)(m, k, n);
        let pa = vec(mm.a_pack().len(), mm.a_pack().alignment());
        let pb = vec(mm.b_pack().len(), mm.b_pack().alignment());
        let mut c = vec![0.0; m * n];
        unsafe {
            be.iter(move || {
                mm.run(
                    pa.as_ptr_unchecked::<f32>() as _,
                    pb.as_ptr_unchecked::<f32>() as _,
                    c.as_mut_ptr() as _,
                    &[],
                )
            });
        }
    });
}

fn simple(c: &mut Criterion, m: usize, k: usize, n: usize) {
    pack_a(c, m, k, n);
    pack_b(c, m, k, n);
    mat_mul_prepacked(c, m, k, n);
}

fn s16x60x8(c: &mut Criterion) {
    simple(c, 16, 60, 8)
}

criterion_group!(benches, s16x60x8);
criterion_main!(benches);
