use criterion::measurement::WallTime;
use criterion::*;
use tract_data::internal::*;

#[allow(dead_code)]
#[path = "../tests/virtual_im2col.rs"]
mod virtual_im2col;
use virtual_im2col::ConvProblem;

fn conv(
    c: &mut BenchmarkGroup<WallTime>,
    ci: usize,
    h: usize,
    w: usize,
    co: usize,
    kh: usize,
    kw: usize,
) {
    // CHW HWIO
    let input = Tensor::zero::<f32>(&[ci, h, w]).unwrap();
    let filters = Tensor::zero::<f32>(&[kh, kw, ci, co]).unwrap();
    let mut cv = ConvProblem { input, filters, lazy_im2col: false };
    c.bench_function("eager", |b| {
        b.iter(|| {
            cv.tract().unwrap();
        })
    });
    cv.lazy_im2col = true;
    c.bench_function("lazy", |b| {
        b.iter(|| {
            cv.tract().unwrap();
        })
    });
}

fn ex1(c: &mut Criterion) {
    let mut c = c.benchmark_group("ex1");
    conv(&mut c, 32, 256, 256, 32, 3, 3);
}

fn big(c: &mut Criterion) {
    let mut c = c.benchmark_group("big");
    conv(&mut c, 1, 1024, 1024, 99, 3, 3);
}

criterion_group!(benches, ex1, big);
criterion_main!(benches);
