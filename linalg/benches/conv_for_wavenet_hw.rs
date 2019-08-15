#[macro_use]
extern crate criterion;
extern crate tract_linalg;
use criterion::Criterion;

fn conv(c: &mut Criterion, dilation: usize, pulse: usize, ci: usize, co: usize) {
    c.bench_function(&format!("conv_d{}p{}ci{}co{}", dilation, pulse, ci, co), move |be| {
        let t = pulse + 2 * dilation;
        let data_offsets: Vec<_> = (0..pulse).map(|x| x as isize).collect();
        let kernel_offsets: Vec<_> =
            (0..ci).flat_map(|ici| (0..3).map(move |x| (ici * t * 3 + x * t) as isize)).collect();
        let conv = (tract_linalg::ops().smmm)(co, kernel_offsets.len(), data_offsets.len());
        let a = tract_linalg::align::realign_vec(
            vec![0.0; conv.a_pack().len()],
            conv.a_pack().alignment(),
        );
        let input = vec![0.0; ci * t];
        let mut output = vec![0.0; co * t];
        be.iter(move || unsafe {
            conv.run(
                &conv.a_from_packed(a.as_ptr()),
                &conv.b_from_data_and_offsets(input.as_ptr(), &*kernel_offsets, &*data_offsets),
                &mut conv.c_from_data_and_strides(output.as_mut_ptr(), t as _, 1),
                &[],
            )
        });
    });
}

fn convs(c: &mut Criterion) {
    conv(c, 1, 8, 16, 64);
    conv(c, 2, 8, 16, 64);
    conv(c, 4, 8, 16, 64);
    conv(c, 8, 8, 16, 64);
}

criterion_group!(benches, convs);
criterion_main!(benches);
