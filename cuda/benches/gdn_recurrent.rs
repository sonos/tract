//! CPU vs CUDA latency for GatedDeltaNetRecurrent.
//!
//! The CUDA kernel (unlike Metal) requires batch=1, width=128, ungrouped
//! heads (q/k/v share the same head count -- no GQA), and an f32 state; it
//! also has no chunked-prefill path, just a per-step host-side loop. Same
//! synthetic-data pattern as the Metal bench (metal/benches/gdn_recurrent.rs).

use criterion::measurement::WallTime;
use criterion::*;
use tract_core::internal::*;
use tract_cuda::kernels::gdn_recurrent::CudaGdnRecurrent;
use tract_cuda::with_cuda_stream;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};
use tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent;

struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
}

fn make_inputs(b: usize, s_len: usize, heads: usize, width: usize) -> Inputs {
    // Ungrouped: k_heads == heads (ungrouped is the only mode the CUDA
    // kernel supports).
    let n_qkv = b * s_len * heads * width;
    let n_gate = b * s_len * heads;
    let n_state = b * heads * width * width;
    let as_f16 = |v: Vec<f32>| v.into_iter().map(f16::from_f32).collect::<Vec<_>>();
    Inputs {
        q: Tensor::from_shape(
            &[b, s_len, heads, width],
            &as_f16((0..n_qkv).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect()),
        )
        .unwrap(),
        k: Tensor::from_shape(
            &[b, s_len, heads, width],
            &as_f16((0..n_qkv).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect()),
        )
        .unwrap(),
        v: Tensor::from_shape(
            &[b, s_len, heads, width],
            &as_f16((0..n_qkv).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect()),
        )
        .unwrap(),
        g: Tensor::from_shape(
            &[b, s_len, heads],
            &(0..n_gate).map(|i| -0.05 - 0.1 * (i % 7) as f32).collect::<Vec<f32>>(),
        )
        .unwrap(),
        beta: Tensor::from_shape(
            &[b, s_len, heads],
            &as_f16((0..n_gate).map(|i| 0.125 + 0.08 * (i % 9) as f32).collect()),
        )
        .unwrap(),
        state: Tensor::from_shape(
            &[b, heads, width, width],
            &(0..n_state).map(|i| ((i % 37) as f32 - 18.0) / 256.0).collect::<Vec<f32>>(),
        )
        .unwrap(),
    }
}

fn cpu_gdn(crit: &mut BenchmarkGroup<WallTime>, label: &str, inputs: &Inputs) {
    crit.bench_function(&format!("cpu_{label}"), |be| {
        be.iter(|| {
            GatedDeltaNetRecurrent::default()
                .eval(
                    &EvalContext::out_of_plan(),
                    tvec![
                        inputs.q.clone().into_tvalue(),
                        inputs.k.clone().into_tvalue(),
                        inputs.v.clone().into_tvalue(),
                        inputs.g.clone().into_tvalue(),
                        inputs.beta.clone().into_tvalue(),
                        inputs.state.clone().into_tvalue(),
                    ],
                )
                .unwrap()
        });
    });
}

fn cuda_gdn(crit: &mut BenchmarkGroup<WallTime>, label: &str, inputs: &Inputs) {
    with_cuda_stream(|stream| {
        let q = inputs.q.clone().into_device()?;
        let k = inputs.k.clone().into_device()?;
        let v = inputs.v.clone().into_device()?;
        let g = inputs.g.clone().into_device()?;
        let beta = inputs.beta.clone().into_device()?;
        let state = inputs.state.clone().into_device()?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F16, q.shape())?;
        let final_state = DeviceTensor::uninitialized_dt(DatumType::F32, state.shape())?;

        // dispatch_eval enqueues asynchronously; sync inside the timed
        // region for a fair per-call LATENCY number, not just launch
        // overhead.
        let run = |stream: &_| -> TractResult<()> {
            CudaGdnRecurrent.dispatch_eval(
                stream,
                &q,
                &k,
                &v,
                &g,
                &beta,
                &state,
                &output,
                &final_state,
            )?;
            Ok(stream.synchronize()?)
        };
        run(stream)?; // warmup

        crit.bench_function(&format!("cuda_{label}"), |be| {
            be.iter(|| run(stream).unwrap());
        });
        Ok(())
    })
    .unwrap();
}

/// Decode: one step, 32 heads, head width 128 (ungrouped -- matches the
/// Metal bench's total head count of 32, but CUDA cannot do GQA so there's
/// no separate k_heads/groups split here).
fn decode_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("gdn_decode_step_b1_s1_h32_w128");
    let inputs = make_inputs(1, 1, 32, 128);
    cpu_gdn(&mut g, "decode", &inputs);
    cuda_gdn(&mut g, "decode", &inputs);
    g.finish();
}

/// Prefill: 512 tokens at the same geometry. The CUDA kernel has no chunked
/// path (see cuda/src/kernels/gdn_recurrent.rs): dispatch_eval loops one
/// kernel launch per token host-side.
fn prefill_512(c: &mut Criterion) {
    let mut g = c.benchmark_group("gdn_prefill_s512_h32_w128");
    let inputs = make_inputs(1, 512, 32, 128);
    cpu_gdn(&mut g, "prefill", &inputs);
    cuda_gdn(&mut g, "prefill", &inputs);
    g.finish();
}

criterion_group!(benches, decode_step, prefill_512);
criterion_main!(benches);
