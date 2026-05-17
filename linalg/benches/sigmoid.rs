#[macro_use]
extern crate criterion;
extern crate tract_linalg;
use criterion::Criterion;

fn ssigmoid(c: &mut Criterion, n: usize) {
    c.bench_function(&format!("ssigmoid_tract_{n}"), move |be| {
        let mut s = (0..n).map(|i| i as f32 / 10.0).collect::<Vec<f32>>();
        let op = &(tract_linalg::ops().sigmoid_f32)();
        be.iter(|| op.run(&mut s));
    });
}

#[inline(never)]
fn rust_sigmoid(x: &mut [f32]) {
    for v in x {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

fn ssigmoid_scalar(c: &mut Criterion, n: usize) {
    c.bench_function(&format!("ssigmoid_scalar_{n}"), move |be| {
        let mut s = (0..n).map(|i| i as f32 / 10.0).collect::<Vec<f32>>();
        be.iter(|| rust_sigmoid(&mut s));
    });
}

fn bs(c: &mut Criterion) {
    ssigmoid(c, 1024);
    ssigmoid_scalar(c, 1024);
}

criterion_group!(benches, bs);
criterion_main!(benches);
