#[macro_use]
extern crate criterion;
extern crate tract_linalg;
use criterion::Criterion;

pub fn vec(len: usize, align: usize) -> *mut f32 {
    let layout =
        std::alloc::Layout::from_size_align(len * std::mem::size_of::<f32>(), align).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 }
}

fn pack_a(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("pack_a_{}x{}x{}", m, k, n), move |b| {
        let mm = (tract_linalg::ops().mmm_f32)(m, k, n);
        let a = vec(m * k, 4);
        let pa = vec(mm.a_pack().len(), mm.a_pack().alignment());
        b.iter(move || mm.a_pack().pack(pa, a, k as _, 1))
    });
}

fn pack_b(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("pack_b_{}x{}x{}", m, k, n), move |be| {
        let mm = (tract_linalg::ops().mmm_f32)(m, k, n);
        let b = vec(n * k, 4);
        let pb = vec(mm.b_pack().len(), mm.b_pack().alignment());
        be.iter(move || mm.b_pack().pack(pb, b, n as _, 1))
    });
}

fn mat_mul_prepacked(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_function(&format!("mat_mul_prepacked_{}x{}x{}", m, k, n), move |be| {
        let mm = (tract_linalg::ops().mmm_f32)(m, k, n);
        let pa = vec(mm.a_pack().len(), mm.a_pack().alignment());
        let pb = vec(mm.b_pack().len(), mm.b_pack().alignment());
        let mut c = vec![0.0; m * n];
        unsafe {
            be.iter(move || mm.run(pa as _, pb as _, c.as_mut_ptr() as _, &[]));
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
