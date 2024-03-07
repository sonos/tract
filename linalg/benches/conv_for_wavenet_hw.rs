#[macro_use]
extern crate criterion;
extern crate tract_data;
extern crate tract_linalg;
use criterion::Criterion;
use tract_linalg::frame::mmm::FusedSpec;

use tract_data::internal::*;
use DatumType::F32;

fn conv(c: &mut Criterion, dilation: usize, pulse: usize, ci: usize, co: usize) {
    c.bench_function(&format!("conv_d{dilation}p{pulse}ci{ci}co{co}"), move |be| unsafe {
        let t = pulse + 2 * dilation;
        let k = ci * 3;
        let mm =
            tract_linalg::ops().mmm(F32, F32, F32, Some(co), Some(ci * 3), Some(pulse)).unwrap();
        mm.c_from_data_and_strides(F32.size_of(), t as _, 1);
        let a = Tensor::zero_aligned::<f32>(&[mm.a_pack().len(k, co)], mm.a_pack().alignment())
            .unwrap();
        let input = Tensor::zero::<f32>(&[ci, t]).unwrap();
        let mut output = Tensor::zero::<f32>(&[co, t]).unwrap();
        be.iter(move || {
            mm.run(
                co,
                t,
                &[
                    FusedSpec::AddMatMul {
                        a: mm.a_packed(F32.size_of(), k).wrap(&a.view()),
                        b: mm.b_packed(F32.size_of(), k).wrap(&input.view()),
                        k,
                    },
                    FusedSpec::Store(mm.c_view(0, 1).wrap(&output.view_mut())),
                ],
            )
            .unwrap()
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
