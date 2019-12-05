extern crate criterion;
extern crate tract_linalg;
use criterion::*;

pub fn vec<T>(len: usize, align: usize) -> *mut T {
    let layout =
        std::alloc::Layout::from_size_align(len * std::mem::size_of::<T>(), align).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) as *mut T }
}

fn mat_mul_smmm(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    let mm = (tract_linalg::ops().smmm)(m, k, n);
    let pa = vec(mm.a_pack().len(), mm.a_pack().alignment());
    let pb = vec(mm.b_pack().len(), mm.b_pack().alignment());
    let mut c = vec![0.0; m * n];
    be.iter(move || unsafe { mm.run(pa, pb, c.as_mut_ptr(), &[]) });
}

fn mat_mul_i8(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    let mm = (tract_linalg::ops().qmmm_i8_i8)(m, k, n);
    let pa = vec(mm.as_mmm().a_pack().len(), mm.as_mmm().a_pack().alignment());
    let pb = vec(mm.as_mmm().b_pack().len(), mm.as_mmm().b_pack().alignment());
    let mut c = vec![0i8; m * n];
    be.iter(move || unsafe { mm.run(pa, pb, c.as_mut_ptr(), &[]) });
}

fn packed_packed(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group("packed_packed");
    let id = format!("{}x{}x{}", m, k, n);
    group.bench_with_input(BenchmarkId::new("f32", &id), &(m, k, n), mat_mul_smmm);
    group.bench_with_input(BenchmarkId::new("i8", &id), &(m, k, n), mat_mul_i8);
}

fn all(c: &mut Criterion) {
    packed_packed(c, 256, 200, 24); // tdnn1
    packed_packed(c, 256, 256, 8); // fastlstm1 and 2
    packed_packed(c, 1690, 256, 8); // output
}

criterion_group!(benches, all);
criterion_main!(benches);
