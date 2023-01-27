#[macro_use]
extern crate criterion;
extern crate tract_linalg;
use criterion::Criterion;

fn ssigmoid(c: &mut Criterion, n: usize) {
    c.bench_function(&format!("ssigmoid_{n}"), move |be| {
        let mut s = (0..n).map(|i| i as f32 / 10.0).collect::<Vec<f32>>();
        let op = &(tract_linalg::ops().sigmoid_f32)();
        be.iter(|| op.run(&mut s));
    });
}

fn bs(c: &mut Criterion) {
    ssigmoid(c, 4);
    ssigmoid(c, 8);
    ssigmoid(c, 128);
    ssigmoid(c, 1024);
}

criterion_group!(benches, bs);
criterion_main!(benches);
