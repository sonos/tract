//! Four-channel along-W convolution kernels at contiguous and subsampled input strides.

use std::hint::black_box;
use std::time::Duration;

use criterion::*;
use tract_linalg::routines::conv_w4_f32;

fn conv_w4(c: &mut Criterion) {
    let Some(kernel) = conv_w4_f32() else { return };
    for taps in [9usize, 27, 64] {
        let mut group = c.benchmark_group(format!("conv_w4/{taps}"));
        for len in [128usize, 256, 767] {
            group.throughput(Throughput::Elements((4 * len * taps) as u64));
            for stride in [1isize, 2, 3, 4] {
                let row = len * stride as usize + taps;
                let offsets: Vec<isize> = (0..taps).map(|i| (i * row) as isize).collect();
                let extent = offsets[taps - 1] as usize + (len - 1) * stride as usize + 1;
                let input: Vec<f32> = (0..extent).map(|i| (i as f32 * 0.37).cos()).collect();
                let weights: Vec<f32> = (0..4 * taps).map(|i| (i as f32 * 0.7).sin()).collect();
                let bias = [0.25, -0.5, 0.125, 1.0];
                let mut output = vec![0f32; 4 * len];
                group.bench_function(format!("w{len}/s{stride}"), |b| {
                    b.iter(|| unsafe {
                        kernel(
                            black_box(input.as_ptr()),
                            output.as_mut_ptr(),
                            black_box(&weights),
                            black_box(&offsets),
                            &bias,
                            len,
                            stride,
                            len as isize,
                        );
                        black_box(&output);
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    targets = conv_w4
}
criterion_main!(benches);
