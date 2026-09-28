#[macro_use]
extern crate criterion;
extern crate tract_data;

use criterion::Criterion;
use tract_data::internal::*;

fn bench_move(
    c: &mut Criterion,
    name: &str,
    dt: DatumType,
    shape: &[usize],
    from: usize,
    to: usize,
) {
    let t = Tensor::zero_dt(dt, shape).unwrap();
    c.bench_function(name, |b| {
        b.iter(|| t.clone().move_axis(from, to).unwrap());
    });
}

fn clone_only(c: &mut Criterion) {
    let t = Tensor::zero_dt(DatumType::F32, &[1, 64, 100, 96]).unwrap();
    c.bench_function("clone_only_1x64x100x96_f32", |b| {
        b.iter(|| t.clone());
    });
}

fn moves(c: &mut Criterion) {
    bench_move(c, "nchw_to_nhwc_1x64x100x96_f32", DatumType::F32, &[1, 64, 100, 96], 1, 3);
    bench_move(c, "nhwc_to_nchw_1x100x96x64_f32", DatumType::F32, &[1, 100, 96, 64], 3, 1);
    bench_move(c, "mobilenet_stem_1x32x112x112_f32", DatumType::F32, &[1, 32, 112, 112], 1, 3);
    bench_move(c, "attention_1x12x128x64_f32", DatumType::F32, &[1, 12, 128, 64], 1, 2);
    bench_move(c, "attention_inv_1x128x12x64_f32", DatumType::F32, &[1, 128, 12, 64], 2, 1);
    bench_move(c, "batch_transpose_2x64x4096_f32", DatumType::F32, &[2, 64, 4096], 1, 2);
    bench_move(c, "transpose_4096x64_f32", DatumType::F32, &[4096, 64], 0, 1);
    bench_move(c, "transpose_8192x64_f32", DatumType::F32, &[8192, 64], 0, 1);
    bench_move(c, "transpose_4000x64_f32", DatumType::F32, &[4000, 64], 0, 1);
    bench_move(c, "nchw_to_nhwc_1x64x100x96_f16", DatumType::F16, &[1, 64, 100, 96], 1, 3);
    bench_move(c, "nchw_to_nhwc_1x64x100x96_u8", DatumType::U8, &[1, 64, 100, 96], 1, 3);
}

criterion_group!(benches, clone_only, moves);
criterion_main!(benches);
