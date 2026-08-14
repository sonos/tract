//! Determinism canary for Metal transient buffer-pool recycling.
//!
//! Runs a fixed feed-forward chain repeatedly with identical inputs and
//! byte-compares every run against the first. Deterministic ops only, so any
//! mismatch is a scheduling/recycling bug, not a numerics trade-off.
//!
//! History: the original recycling race (pool_put at host-drop time while an
//! in-flight command buffer still referenced the pair, fixed by deferring
//! recycling to buffer completion) was only ever observed on full q40
//! transformer decodes at low in-flight depth; this synthetic chain never
//! reproduced it. Kept as a cheap bounded canary: it must stay at 0 dirty
//! runs. Knobs:
//!   arg1: TRACT_METAL_MAX_IN_FLIGHT        (default 2)
//!   arg2: TRACT_METAL_COMMIT_EVERY_N_DISPATCHES (default 10)
//!   arg3: iterations                        (default 40)
//!
//!   cargo run --release -p tract-metal --example pool_race -- 2 10 40

use tract_core::internal::*;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::math::add;
use tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};
use tract_core::transform::ModelTransform;
use tract_metal::MetalTransform;

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let depth = args.get(1).map(|s| s.as_str()).unwrap_or("2").to_string();
    let cadence = args.get(2).map(|s| s.as_str()).unwrap_or("10").to_string();
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);

    // Must precede any Metal dispatch: the tuning profile freezes at first
    // read (autotune off).
    unsafe {
        std::env::set_var("TRACT_METAL_AUTOTUNE", "0");
        std::env::set_var("TRACT_METAL_MAX_IN_FLIGHT", &depth);
        std::env::set_var("TRACT_METAL_COMMIT_EVERY_N_DISPATCHES", &cadence);
        // Match the dirty probe condition "arena disabled, pool on": every
        // transient is an individually pooled buffer.
        std::env::set_var("TRACT_GPU_DISABLE_MEMORY_ARENA", "1");
    }

    let env_usize = |name: &str, default: usize| -> usize {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let (layers, m, k) = (
        env_usize("POOL_RACE_LAYERS", 48),
        env_usize("POOL_RACE_M", 64),
        env_usize("POOL_RACE_K", 512),
    );
    let mut model = TypedModel::default();
    let mut x = model.add_source("x", f32::fact([1, m, k]))?;
    for l in 0..layers {
        let w: Vec<f32> = (0..k * k)
            .map(|i| ((((i + l * 7919) * 2654435761) % 2000) as f32 / 1000.0 - 1.0) / (k as f32))
            .collect();
        let w = model.add_const(format!("w{l}"), Tensor::from_shape(&[1, k, k], &w)?)?;
        let mm = model.wire_node(
            format!("mm{l}"),
            PrefixMatMul {
                transpose_a: false,
                transpose_b: false,
                transpose_c: false,
                quantize_output: None,
                operating_dt: Some(DatumType::F32),
            },
            &[x, w],
        )?[0];
        let b: Vec<f32> = (0..k).map(|i| ((i * 31 + l) % 100) as f32 / 100.0).collect();
        let b = model.add_const(format!("b{l}"), Tensor::from_shape(&[1, 1, k], &b)?)?;
        let biased = model.wire_node(format!("add{l}"), add(), &[mm, b])?[0];
        x = model.wire_node(
            format!("sm{l}"),
            Softmax::new(tvec![2], None, SoftmaxKind::Softmax(SoftmaxExp::Libc)),
            &[biased],
        )?[0];
    }
    model.select_output_outlets(&[x])?;

    let metal = MetalTransform::default().transform_into(model)?;
    let runnable = metal.into_optimized()?.into_runnable()?;

    let input: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) / 100.0).collect();
    let input = Tensor::from_shape(&[1, m, k], &input)?.into_tvalue();

    let bytes_of = |out: &TVec<TValue>| -> Vec<Vec<u8>> {
        out.iter().map(|t| t.clone().into_tensor().as_bytes().to_vec()).collect()
    };

    let reference = bytes_of(&runnable.run(tvec![input.clone()])?);
    let mut dirty = 0usize;
    for it in 0..iters {
        let out = bytes_of(&runnable.run(tvec![input.clone()])?);
        if out != reference {
            dirty += 1;
            let ix = reference
                .iter()
                .zip(out.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(usize::MAX);
            let byte = reference[ix]
                .iter()
                .zip(out[ix].iter())
                .position(|(a, b)| a != b)
                .unwrap_or(usize::MAX);
            println!("iter {it}: MISMATCH output #{ix} first differing byte {byte}");
        }
    }
    println!(
        "depth={depth} cadence={cadence}: {dirty}/{iters} dirty runs vs reference"
    );
    if dirty > 0 {
        std::process::exit(1);
    }
    Ok(())
}
