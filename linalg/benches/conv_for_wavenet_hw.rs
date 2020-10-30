#[macro_use]
extern crate criterion;
extern crate tract_data;
extern crate tract_linalg;
use criterion::Criterion;

use tract_data::internal::*;

fn conv(c: &mut Criterion, dilation: usize, pulse: usize, ci: usize, co: usize) {
    c.bench_function(&format!("conv_d{}p{}ci{}co{}", dilation, pulse, ci, co), move |be| unsafe {
        let t = pulse + 2 * dilation;
        let data_offsets: Vec<_> = (0..pulse).map(|x| x as isize).collect();
        let kernel_offsets: Vec<_> =
            (0..ci).flat_map(|ici| (0..3).map(move |x| (ici * t * 3 + x * t) as isize)).collect();
        let mut conv = (tract_linalg::ops().mmm_f32)(co, kernel_offsets.len(), data_offsets.len());
        conv.c_from_data_and_strides(t as _, 1);
        let a =
            Tensor::from_slice_align(&vec![0.0; conv.a_pack().len()], conv.a_pack().alignment())
                .unwrap();
        let input = Tensor::zero::<f32>(&[ci, t]).unwrap();
        let mut output = Tensor::zero::<f32>(&[co, t]).unwrap();
        be.iter(move || conv.run(&a.view(), &input.view(), &mut output.view_mut(), &[]).unwrap());
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
