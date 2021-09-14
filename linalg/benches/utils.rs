#![allow(dead_code)]
use criterion::*;
use tract_data::internal::*;
use tract_linalg::frame::mmm::{FusedSpec, InputStore, PackedStore};
use tract_linalg::frame::MatMatMul;

use DatumType::*;

pub fn packed_packed(c: &mut Criterion, name: &str, m: usize, k: usize, n: usize) {
    let mut group = c.benchmark_group(format!("{}/packed_packed", name));
    group.throughput(Throughput::Elements((m * k * n) as u64));
    let id = format!("{}x{}x{}", m, k, n);
    group.bench_with_input(BenchmarkId::new("f32/cold", &id), &(F32, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("f32/hot", &id), &(F32, m, k, n, false), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/cold", &id), &(I8, m, k, n, true), mat_mat);
    group.bench_with_input(BenchmarkId::new("i8/hot", &id), &(I8, m, k, n, false), mat_mat);
}

pub fn packed_vec(c: &mut Criterion, name: &str, m: usize, k: usize, n: usize) {
    assert_eq!(n, 1);
    let mut group = c.benchmark_group(format!("{}/packed_vec", name));
    group.throughput(Throughput::Elements((m * k * n) as u64));
    let id = format!("{}x{}x{}", m, k, n);
    group.bench_with_input(BenchmarkId::new("f32/cold", &id), &(F32, m, k, n, true), mat_vec);
    group.bench_with_input(BenchmarkId::new("f32/hot", &id), &(F32, m, k, n, false), mat_vec);
    group.bench_with_input(BenchmarkId::new("i8/cold", &id), &(I8, m, k, n, true), mat_vec);
    group.bench_with_input(BenchmarkId::new("i8/hot", &id), &(I8, m, k, n, false), mat_vec);
}

pub fn direct_conv(
    c: &mut Criterion,
    name: &str,
    p: usize,
    kl: usize,
    ci: usize,
    co: usize,
    stride: usize,
) {
    let mut group = c.benchmark_group(format!("{}/conv", name));
    group.throughput(Throughput::Elements((kl * p * ci * co / stride) as u64));
    let id = format!("{}x{}x{}x{}", p, kl, ci, co);
    group.bench_with_input(
        BenchmarkId::new("f32", &id),
        &(p, kl, ci, co, stride),
        direct_conv_mmm_f32,
    );
    group.bench_with_input(BenchmarkId::new("i8", &id), &(p, kl, ci, co, stride), direct_conv_i8);
}

fn ruin_cache() {
    let _a = (0..1000000).collect::<Vec<i32>>();
}

unsafe fn run(
    m: usize,
    k: usize,
    n: usize,
    be: &mut Bencher,
    mm: &dyn MatMatMul,
    pa: PackedStore,
    pb: InputStore,
    cold: bool,
) {
    let mut scratch = mm.allocate_scratch_space();
    be.iter_custom(move |iters| {
        let mut dur = std::time::Duration::default();
        for _ in 0..iters {
            if cold {
                ruin_cache();
            }
            let instant = std::time::Instant::now();
            mm.run_with_scratch_space(
                m,
                n,
                scratch.as_mut(),
                &[FusedSpec::AddMatMul { a: pa, b: pb.clone(), k }],
            )
            .unwrap();
            let time = instant.elapsed();
            dur += time;
        }
        dur
    });
}

fn mat_mat(be: &mut Bencher, &(dt, m, k, n, cold): &(DatumType, usize, usize, usize, bool)) {
    let mm = tract_linalg::ops().mmm(dt, dt, dt, Some(m), Some(k), Some(n)).unwrap();
    let pa = Tensor::zero_aligned_dt(dt, &[mm.a_pack(k).len(m)], mm.a_pack(k).alignment()).unwrap();
    let pb = Tensor::zero_aligned_dt(dt, &[mm.b_pack(k).len(n)], mm.b_pack(k).alignment()).unwrap();
    unsafe {
        run(
            m,
            k,
            n,
            be,
            &*mm,
            mm.a_packed(dt.size_of(), k).wrap(&pa.view()),
            mm.b_packed(dt.size_of(), k).wrap(&pb.view()),
            cold,
        );
    }
}

fn mat_vec(be: &mut Bencher, &(dt, m, k, n, cold): &(DatumType, usize, usize, usize, bool)) {
    assert_eq!(n, 1);
    let mm = tract_linalg::ops().mmm(dt, dt, dt, Some(m), Some(k), Some(n)).unwrap();
    let pa = Tensor::zero_aligned_dt(dt, &[mm.a_pack(k).len(m)], mm.a_pack(k).alignment()).unwrap();
    let pb = Tensor::zero_dt(dt, &[k, 1]).unwrap();
    unsafe {
        run(
            m,
            k,
            n,
            be,
            &*mm,
            mm.a_packed(dt.size_of(), k).wrap(&pa.view()),
            mm.b_packed(dt.size_of(), k).wrap(&pb.view()),
            cold,
        );
    }
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
        let mm = tract_linalg::ops().mmm(F32, F32, F32, Some(m), Some(k), Some(n)).unwrap();
        let pa =
            Tensor::zero_aligned::<f32>(&[mm.a_pack(k).len(m)], mm.a_pack(k).alignment()).unwrap();
        let pb = Tensor::zero_aligned::<f32>(&[b_len], mm.b_pack(k).alignment()).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, n]).unwrap();
        mm.b_from_data_and_offsets(pb.datum_type().size_of(), &rows_offsets, &cols_offsets);
        be.iter(move || {
            mm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: mm.a_packed(f32::datum_type().size_of(), k).wrap(&pa.view()),
                        b: mm
                            .b_from_data_and_offsets(
                                c.datum_type().size_of(),
                                &rows_offsets,
                                &cols_offsets,
                            )
                            .wrap(&pb.view()),
                        k,
                    },
                    FusedSpec::Store(mm.c_view().wrap(&c.view_mut())),
                ],
            )
        })
    }
}

fn direct_conv_i8(be: &mut Bencher, geo: &ConvGeo) {
    unsafe {
        let (m, k, n, rows_offsets, cols_offsets, b_len) = direct_conv_geo(geo);
        let mm = tract_linalg::ops().mmm(I8, I8, I8, Some(m), Some(k), Some(n)).unwrap();
        let pa =
            Tensor::zero_aligned::<i8>(&[mm.a_pack(k).len(m)], mm.a_pack(k).alignment()).unwrap();
        let pb = Tensor::zero_aligned::<i8>(&[b_len], mm.b_pack(k).alignment()).unwrap();
        let mut c = Tensor::zero::<i8>(&[m, n]).unwrap();
        mm.b_from_data_and_offsets(pb.datum_type().size_of(), &rows_offsets, &cols_offsets);
        be.iter(move || {
            mm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: mm.a_packed(i8::datum_type().size_of(), k).wrap(&pa.view()),
                        b: mm
                            .b_from_data_and_offsets(
                                pb.datum_type().size_of(),
                                &rows_offsets,
                                &cols_offsets,
                            )
                            .wrap(&pb.view()),
                        k,
                    },
                    FusedSpec::Store(mm.c_view().wrap(&c.view_mut())),
                ],
            )
        })
    }
}
