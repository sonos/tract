#[macro_use]
extern crate criterion;
extern crate tract_linalg;
use criterion::Criterion;

pub fn vec(len: usize, align: usize) -> *mut f32 {
    let layout =
        std::alloc::Layout::from_size_align(len * std::mem::size_of::<f32>(), align).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 }
}

fn mat_mul_smm(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    let mm = (tract_linalg::ops().smm)(m, k, n);
    let pa = vec(mm.packed_a_len(), mm.packed_a_alignment());
    let pb = vec(mm.b_pack().len(), mm.b_pack().alignment());
    let mut c = vec![0.0; m * n];
    be.iter(move || mm.mat_mul_prepacked(pa, pb, c.as_mut_ptr(), n as _, 1))
}

fn mat_mul_stile(be: &mut criterion::Bencher, &(m, k, n): &(usize, usize, usize)) {
    let mm = (tract_linalg::ops().stile)(m, k, n);
    let pa = vec(mm.a_pack().len(), mm.a_pack().alignment());
    let pb = vec(mm.b_pack().len(), mm.b_pack().alignment());
    let mut c = vec![0.0; m * n];
    be.iter(move || unsafe {
        mm.run(
            &mm.a_from_packed(pa),
            &mm.b_from_packed(pb),
            &mut mm.c_from_data_and_strides(c.as_mut_ptr(), n as _, 1),
        )
    });
}

fn mat_mul_prepacked(c: &mut Criterion, m: usize, k: usize, n: usize) {
    c.bench_functions(
        &format!("mat_mul_prepacked"),
        vec![criterion::Fun::new("smm", mat_mul_smm), criterion::Fun::new("stitle", mat_mul_stile)],
        (m, k, n),
    );
}

fn s64x288x21609(c: &mut Criterion) {
    mat_mul_prepacked(c, 64, 288, 21609)
}

criterion_group!(benches, s64x288x21609);
criterion_main!(benches);
