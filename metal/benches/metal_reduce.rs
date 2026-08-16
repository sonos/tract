use criterion::*;
use tract_core::internal::*;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};
use tract_metal::kernels::nn::reduce::{Reducer, metal_reduce_launch};
use tract_metal::with_metal_stream;

const DISPATCHES: usize = 20;

fn bench_shape(
    g: &mut BenchmarkGroup<measurement::WallTime>,
    shape: &[usize],
    axis: usize,
    name: &str,
) {
    with_metal_stream(|stream| {
        let len = shape.iter().product::<usize>();
        let a = Tensor::from_shape(shape, &vec![1.0f32; len])?.into_device()?;
        let mut o_shape = shape.to_vec();
        o_shape[axis] = 1;
        let output = unsafe { DeviceTensor::uninitialized_dt(a.datum_type(), &o_shape)? };
        metal_reduce_launch(&Reducer::Sum, &a, axis, &output)?;
        stream.wait_until_completed()?;

        g.bench_function(name, |b| {
            b.iter(|| {
                for _ in 0..DISPATCHES {
                    metal_reduce_launch(&Reducer::Sum, &a, axis, &output).unwrap();
                }
                stream.wait_until_completed().unwrap();
            })
        });
        Ok(())
    })
    .unwrap()
}

fn reduce(c: &mut Criterion) {
    let mut g = c.benchmark_group("reduce_sum_f32");
    g.sample_size(20);
    for (shape, axis, name) in [
        (tvec![64usize, 262144usize], 1, "64x262144"),
        (tvec![256, 65536], 1, "256x65536"),
        (tvec![1024, 16384], 1, "1024x16384"),
        (tvec![16384, 1024], 1, "16384x1024"),
        (tvec![32768, 333], 1, "32768x333"),
        (tvec![65536, 64], 1, "65536x64"),
        (tvec![64, 1024, 512], 1, "64x1024x512_strided"),
    ] {
        bench_shape(&mut g, &shape, axis, name);
    }
    g.finish();
}

criterion_group!(benches, reduce);
criterion_main!(benches);
