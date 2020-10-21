#[macro_use]
extern crate criterion;
extern crate tract_data;
extern crate tract_linalg;
use criterion::Criterion;

use tract_data::prelude::*;

fn conv(c: &mut Criterion, dilation: usize, pulse: usize, ci: usize, co: usize) {
    c.bench_function(&format!("conv_d{}p{}ci{}co{}", dilation, pulse, ci, co), move |be| {
        let t = pulse + 2 * dilation;
        let data_offsets: Vec<_> = (0..pulse).map(|x| x as isize).collect();
        let kernel_offsets: Vec<_> =
            (0..ci).flat_map(|ici| (0..3).map(move |x| (ici * t * 3 + x * t) as isize)).collect();
        let mut conv = (tract_linalg::ops().mmm_f32)(co, kernel_offsets.len(), data_offsets.len());
        unsafe {
            conv.c_from_data_and_strides(t as _, 1);
        }
        let a = unsafe {
            Tensor::from_slice_align(&vec![0.0; conv.a_pack().len()], conv.a_pack().alignment())
                .unwrap()
        };
        let input = vec![0.0; ci * t];
        let mut output = vec![0.0; co * t];
        be.iter(move || unsafe {
            conv.run(a.as_ptr_unchecked(), input.as_ptr(), output.as_mut_ptr(), &[]);
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
