#[macro_use]
extern crate criterion;
extern crate tract_linalg;
use criterion::Criterion;

fn conv(c: &mut Criterion, dilation: usize, pulse: usize, ci: usize, co: usize) {
    c.bench_function(&format!("conv_d{}p{}ci{}co{}", dilation, pulse, ci, co), move |be| {
        let t = pulse + 2 * dilation;
        let n = pulse;
        let data_offsets = (0..pulse).map(|x| x as isize).collect();
        let kernel_offsets =
            (0..ci).flat_map(|ici| (0..3).map(move |x| (ici * t * 3 + x * t) as isize)).collect();
        let conv = (tract_linalg::ops().sconv)(co, kernel_offsets, data_offsets);
        let a = tract_linalg::align::realign_vec(
            vec![0.0; conv.packed_a_len()],
            conv.packed_a_alignment(),
        );
        let input = vec![0.0; ci * t];
        let mut output = vec![0.0; co * t];
        be.iter(move || conv.conv(a.as_ptr(), input.as_ptr(), output.as_mut_ptr(), n as _, 1))
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
