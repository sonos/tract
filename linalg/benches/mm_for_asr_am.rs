use criterion::*;
use tract_data::internal::*;
use tract_linalg::frame::mmm::MatrixStore;
use tract_linalg::frame::MatMatMul;

use DatumType::*;

fn ruin_cache() {
    let _a = (0..1000000).collect::<Vec<i32>>();
}

unsafe fn run(
    be: &mut Bencher,
    mm: &dyn MatMatMul,
    pa: &MatrixStore,
    pb: &MatrixStore,
    c: &mut MatrixStore,
    cold: bool,
) {
    be.iter_custom(move |iters| {
        let mut dur = std::time::Duration::default();
        for _ in 0..iters {
            if cold {
                ruin_cache();
            }
            let instant = std::time::Instant::now();
            mm.run(&pa, &pb, c, &[]).unwrap();
            let time = instant.elapsed();
            dur += time;
        }
        dur
    });
}

fn mat_mat(be: &mut Bencher, &(dt, m, k, n, cold): &(DatumType, usize, usize, usize, bool)) {
    let mm = tract_linalg::ops().mmm(dt, dt, dt, m, k, n).unwrap();
    let pa = Tensor::zero_aligned_dt(dt, &[mm.a_pack().len(m)], mm.a_pack().alignment()).unwrap();
    let pb = Tensor::zero_aligned_dt(dt, &[mm.b_pack().len(n)], mm.b_pack().alignment()).unwrap();
    let mut c = Tensor::zero_dt(dt, &[m, n]).unwrap();
    unsafe {
        run(
            be,
            &*mm,
            &mm.a_packed().wrap(&pa.view()),
            &mm.b_packed().wrap(&pb.view()),
            &mut mm.c_view().wrap(&mut c.view_mut()),
            cold,
        );
    }
}

fn mat_vec(be: &mut Bencher, &(dt, m, k, n, cold): &(DatumType, usize, usize, usize, bool)) {
    let mm = tract_linalg::ops().mmm(dt, dt, dt, m, k, n).unwrap();
    let pa = Tensor::zero_aligned_dt(dt, &[mm.a_pack().len(m)], mm.a_pack().alignment()).unwrap();
    let pb = Tensor::zero_dt(dt, &[k, 1]).unwrap();
    let mut c = Tensor::zero_dt(dt, &[m, n]).unwrap();
    unsafe {
        run(
            be,
            &*mm,
            &mm.a_packed().wrap(&pa.view()),
            &mm.b_vec_from_data().wrap(&pb.view()),
            &mut mm.c_view().wrap(&mut c.view_mut()),
            cold,
        );
    }
}

fn packed_packed(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group("packed_packed");
    let id = format!("{}x{}x{}", m, k, n);
    group.bench_with_input(BenchmarkId::new("f32/cold", &id), &(F32, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("f32/hot", &id), &(F32, m, k, n, false), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/cold", &id), &(I8, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/hot", &id), &(I8, m, k, n, false), mat_mat);
}

fn packed_vec(c: &mut Criterion, m: usize, k: usize, n: usize) {
    assert_eq!(n, 1);
    let mut group = c.benchmark_group("packed_vec");
    let id = format!("{}x{}x{}", m, k, n);
    group.bench_with_input(BenchmarkId::new("f32/cold", &id), &(F32, m, k, n, true), mat_vec);
    group.bench_with_input(BenchmarkId::new("f32/hot", &id), &(F32, m, k, n, false), mat_vec);
    group.bench_with_input(BenchmarkId::new("i8/cold", &id), &(I8, m, k, n, true), mat_vec);
    group.bench_with_input(BenchmarkId::new("i8/hot", &id), &(I8, m, k, n, false), mat_vec);
}

type ConvGeo = (usize, usize, usize, usize, usize);

fn direct_conv_geo(
    &(pulse, kern, ci, co, stride): &ConvGeo,
) -> (usize, usize, usize, Vec<isize>, Vec<isize>, usize) {
    let (m, k, n) = (co, kern * ci, pulse / stride);
    let rows_offsets: Vec<isize> =
        (0..ci).flat_map(move |ici| (0..kern).map(move |ik| (ik * ci + ici) as isize)).collect();
    let cols_offsets: Vec<isize> = (0..n).map(move |i| (i * ci * stride) as isize).collect();
    let b_len = cols_offsets.iter().max().unwrap() + rows_offsets.iter().max().unwrap() + 1;
    (m, k, n, rows_offsets, cols_offsets, b_len as usize)
}

fn direct_conv_mmm_f32(be: &mut Bencher, geo: &ConvGeo) {
    unsafe {
        let (m, k, n, rows_offsets, cols_offsets, b_len) = direct_conv_geo(geo);
        let mm = tract_linalg::ops().mmm(F32, F32, F32, m, k, n).unwrap();
        let pa =
            Tensor::zero_aligned::<f32>(&[mm.a_pack().len(m)], mm.a_pack().alignment()).unwrap();
        let pb = Tensor::zero_aligned::<f32>(&[b_len], mm.b_pack().alignment()).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, n]).unwrap();
        mm.b_from_data_and_offsets(&rows_offsets, &cols_offsets);
        be.iter(move || {
            mm.run(
                &mm.a_packed().wrap(&pa.view()),
                &mm.b_from_data_and_offsets(&rows_offsets, &cols_offsets).wrap(&pb.view()),
                &mut mm.c_view().wrap(&c.view_mut()),
                &[],
            )
        })
    }
}

fn direct_conv_i8(be: &mut Bencher, geo: &ConvGeo) {
    unsafe {
        let (m, k, n, rows_offsets, cols_offsets, b_len) = direct_conv_geo(geo);
        let mm = tract_linalg::ops().mmm(I8, I8, I8, m, k, n).unwrap();
        let pa =
            Tensor::zero_aligned::<i8>(&[mm.a_pack().len(m)], mm.a_pack().alignment()).unwrap();
        let pb = Tensor::zero_aligned::<i8>(&[b_len], mm.b_pack().alignment()).unwrap();
        let mut c = Tensor::zero::<i8>(&[m, n]).unwrap();
        mm.b_from_data_and_offsets(&rows_offsets, &cols_offsets);
        be.iter(move || {
            mm.run(
                &mm.a_packed().wrap(&pa.view()),
                &mm.b_from_data_and_offsets(&rows_offsets, &cols_offsets).wrap(&pb.view()),
                &mut mm.c_view().wrap(&c.view_mut()),
                &[],
            )
        })
    }
}

fn direct_conv(c: &mut Criterion, p: usize, kl: usize, ci: usize, co: usize, stride: usize) {
    let mut group = c.benchmark_group("conv");
    let id = format!("{}x{}x{}x{}", p, kl, ci, co);
    group.bench_with_input(
        BenchmarkId::new("f32", &id),
        &(p, kl, ci, co, stride),
        direct_conv_mmm_f32,
    );
    group.bench_with_input(BenchmarkId::new("i8", &id), &(p, kl, ci, co, stride), direct_conv_i8);
}

fn all(c: &mut Criterion) {
    direct_conv(c, 24, 5, 40, 200, 1); // lda
    packed_packed(c, 256, 200, 24); // tdnn1
    direct_conv(c, 24, 3, 256, 256, 1); // tdnn2
    direct_conv(c, 24, 3, 256, 256, 3); // tdnn3
    packed_packed(c, 256, 256, 8); // fastlstm1 and 2 (input) x 8 (4 prod x 2 layers)
    packed_packed(c, 256, 128, 1); // fastlstm1 and 2 (hidden) x 64 (4 prod x 2 layers x 8 loops)
    packed_packed(c, 256, 256, 1); // fastlstm1 and 2 (rp) x 16 (2 layers x 8 loops)
    direct_conv(c, 8, 3, 256, 256, 1); // tdnn4, tdd5 (x2)
    packed_packed(c, 1690, 256, 8); // output

    // 8M
    packed_packed(c, 512, 200, 24); // tdnn1
    packed_packed(c, 512, 512, 24); // tdnn2
    packed_packed(c, 512, 256, 1); // fastlstm1 and 2 (four parts, rec mat*vec)
    packed_vec(c, 512, 256, 1); // fastlstm1 and 2 (four parts, rec mat*vec)
}

criterion_group!(benches, all);
criterion_main!(benches);
