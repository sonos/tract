#[macro_use]
extern crate criterion;
extern crate tract_linalg;
use criterion::Criterion;

fn pack_a(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("pack_a_{}x{}x{}", m, k, n), move |b| {
        let mm = (tract_linalg::ops().smm)(m, k, n);
        let a = vec![0.0; m * k];
        let mut pa = vec![0.0; mm.packed_a_len()];
        b.iter(move || mm.pack_a(pa.as_mut_ptr(), a.as_ptr(), k as _, 1))
    });
}

fn pack_b(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("pack_b_{}x{}x{}", m, k, n), move |be| {
        let mm = (tract_linalg::ops().smm)(m, k, n);
        let b = vec![0.0; n * k];
        let mut pb = vec![0.0; mm.packed_b_len()];
        be.iter(move || mm.pack_b(pb.as_mut_ptr(), b.as_ptr(), n as _, 1))
    });
}

fn mat_mul_prepacked(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("mat_mul_prepacked_{}x{}x{}", m, k, n), move |be| {
        let mm = (tract_linalg::ops().smm)(m, k, n);
        let a = vec![0.0; mm.packed_a_len()];
        let b = vec![0.0; mm.packed_b_len()];
        let mut c = vec![0.0; m * n];
        be.iter(move || mm.mat_mul_prepacked(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n as _, 1))
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
