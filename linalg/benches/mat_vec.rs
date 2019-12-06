extern crate criterion;
extern crate tract_linalg;
use criterion::*;

pub fn vec(len: usize, align: usize) -> *mut f32 {
    let layout =
        std::alloc::Layout::from_size_align(len * std::mem::size_of::<f32>(), align).unwrap();
    unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 }
}

fn mat_vec_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_vec_mul");
    for (m, k) in [(64usize, 64usize)].iter() {
        group.throughput(Throughput::Elements((m * k) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(m, k),
            |be, (&m, &k)| {
                let mut mm = (tract_linalg::ops().smmm)(m, k, 1);
                let pa = vec(mm.a_pack().len(), mm.a_pack().alignment());
                let b = vec![0.0; k];
                let mut c = vec![0.0; m];
                unsafe { mm.b_vec_from_data(); }
                be.iter(move || unsafe { mm.run(pa, b.as_ptr(), c.as_mut_ptr(), &[]) });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, mat_vec_mul);
criterion_main!(benches);
