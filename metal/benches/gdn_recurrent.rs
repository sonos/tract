//! CPU (pre-existing reference) vs Metal (split E) for GatedDeltaNetRecurrent.
//!
//! Same synthetic-data pattern as
//! `metal/src/kernels/gdn_recurrent.rs`'s `multi_step_matches_cpu_op_len`
//! test (proven CPU/Metal-parity inputs), reused here purely for timing.

use criterion::measurement::WallTime;
use criterion::*;
use tract_core::internal::*;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};
use tract_metal::kernels::gdn_recurrent::metal_gdn_recurrent_launch;
use tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent;

struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
}

fn make_inputs(b: usize, s_len: usize, k_heads: usize, groups: usize, width: usize) -> Inputs {
    let heads = k_heads * groups;
    let n_qk = b * s_len * k_heads * width;
    let n_vec = b * s_len * heads * width;
    let n_gate = b * s_len * heads;
    let n_state = b * heads * width * width;
    let as_f16 = |v: Vec<f32>| v.into_iter().map(f16::from_f32).collect::<Vec<_>>();
    Inputs {
        q: Tensor::from_shape(
            &[b, s_len, k_heads, width],
            &as_f16((0..n_qk).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect()),
        )
        .unwrap(),
        k: Tensor::from_shape(
            &[b, s_len, k_heads, width],
            &as_f16((0..n_qk).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect()),
        )
        .unwrap(),
        v: Tensor::from_shape(
            &[b, s_len, heads, width],
            &as_f16((0..n_vec).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect()),
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
                .eval(tvec![
                    inputs.q.clone().into_tvalue(),
                    inputs.k.clone().into_tvalue(),
                    inputs.v.clone().into_tvalue(),
                    inputs.g.clone().into_tvalue(),
                    inputs.beta.clone().into_tvalue(),
                    inputs.state.clone().into_tvalue(),
                ])
                .unwrap()
        });
    });
}

fn metal_gdn(
    crit: &mut BenchmarkGroup<WallTime>,
    label: &str,
    inputs: &Inputs,
    disable_chunked: bool,
) {
    // Toggles dispatch_eval's own gate for the prefill chunking optimization
    // this PR adds (see metal/src/kernels/gdn_recurrent.rs): unset uses the
    // new chunked path, set forces the pre-existing threadgroup-parallel
    // kernel this PR's chunking sits on top of ("prior to this PR's optim").
    if disable_chunked {
        unsafe { std::env::set_var("TRACT_METAL_DISABLE_GDN_CHUNKED", "1") };
    } else {
        unsafe { std::env::remove_var("TRACT_METAL_DISABLE_GDN_CHUNKED") };
    }
    // First call initializes the global device context .into_device() below
    // requires, and reuses one thread-local MetalStream for every later call.
    tract_metal::with_metal_stream(|_| Ok(())).unwrap();
    let q = inputs.q.clone().into_device().unwrap();
    let k = inputs.k.clone().into_device().unwrap();
    let v = inputs.v.clone().into_device().unwrap();
    let g = inputs.g.clone().into_device().unwrap();
    let beta = inputs.beta.clone().into_device().unwrap();
    let state = inputs.state.clone().into_device().unwrap();
    let output = DeviceTensor::uninitialized_dt(DatumType::F16, v.shape()).unwrap();
    let next = DeviceTensor::uninitialized_dt(state.datum_type(), state.shape()).unwrap();

    // dispatch_eval/dispatch_chunked enqueue work asynchronously and do not
    // wait for the GPU before returning (by design, for real pipelining), so
    // a fair per-call LATENCY number needs an explicit wait in the timed
    // region -- otherwise this only measures CPU-side command encoding.
    let run = || -> TractResult<()> {
        metal_gdn_recurrent_launch(&q, &k, &v, &g, &beta, &state, &output, &next, false)?;
        tract_metal::with_metal_stream(|stream| stream.wait_until_completed())
    };

    run().unwrap(); // warmup: pays one-time pipeline/library load cost

    crit.bench_function(&format!("metal_{label}"), |be| {
        be.iter(|| run().unwrap());
    });
}

/// Decode: one step, the real Qwen3.5-35B geometry (16 k-heads, GQA groups=2,
/// head width 128) — the threadgroup-parallel kernel path (unaffected by
/// this PR's chunking optimization: chunking only engages at s_len >= 64).
fn decode_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("gdn_decode_step_b1_s1_h32_w128");
    let inputs = make_inputs(1, 1, 16, 2, 128);
    cpu_gdn(&mut g, "decode", &inputs);
    metal_gdn(&mut g, "decode", &inputs, false);
    g.finish();
}

/// Prefill: one 512-token chunk at the same geometry. Three points: the CPU
/// reference, this PR's new chunked gated-delta-rule path (`GDN_CHUNK = 64`,
/// 8 chunks), and the pre-existing threadgroup-parallel kernel looping one
/// token at a time -- i.e. what Metal prefill looked like immediately prior
/// to this PR's own chunking optimization.
fn prefill_chunk(c: &mut Criterion) {
    let mut g = c.benchmark_group("gdn_prefill_s512_h32_w128");
    let inputs = make_inputs(1, 512, 16, 2, 128);
    cpu_gdn(&mut g, "prefill", &inputs);
    metal_gdn(&mut g, "prefill_old_tg_per_token", &inputs, true);
    metal_gdn(&mut g, "prefill_chunked", &inputs, false);
    g.finish();
}

criterion_group!(benches, decode_step, prefill_chunk);
criterion_main!(benches);
