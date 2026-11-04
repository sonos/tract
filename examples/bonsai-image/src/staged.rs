//! Staged stand-in for Ternary Bonsai Image: a 4-bit encoder matmul, a ternary
//! diffusion matmul kept for four steps, then a float decode. Each runnable is
//! dropped and the device flushed before the next one is loaded.
//!
//! These graphs are not the Klein weights. A 512² picture still needs the
//! Qwen3 encoder, the ternary MMDiT and the Flux2 VAE as ONNX or NNEF.

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use tract_core::dyn_clone::clone_box;
use tract_core::internal::*;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::BlockQuant;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::BlockQuantStorage;
use tract_core::tract_linalg::block_quant::Q2_0_T;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_core::transform::ModelTransform;
use tract_gpu::device::DeviceContext;
use tract_wgpu::WgpuTransform;
use tract_wgpu::ops::matmul::WgpuGemm;

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 6.0 - 3.0
        })
        .collect()
}

fn pack(format: &dyn BlockQuant, n: usize, k: usize, data: &[f32]) -> TractResult<Const> {
    let blob = format.quant_f32(data)?;
    let storage = BlockQuantStorage::new(clone_box(format), n, k, Arc::new(blob))?;
    let tensor = storage.into_tensor_with_shape(f32::datum_type(), &[n, k]);
    let fact = BlockQuantFact::new(clone_box(format), tvec![n, k]);
    Const::new_with_exotic_fact(Arc::new(tensor), Box::new(fact))
}

fn matmul(m: usize, k: usize, n: usize, weight: Const) -> TractResult<TypedModel> {
    ensure!(weight.val().shape() == [n, k]);
    let mut model = TypedModel::default();
    let a = model.add_source("a", f32::fact([m, k]))?;
    let w = model.wire_node("w", weight, &[])?[0];
    let op = PrefixMatMul {
        transpose_a: false,
        transpose_b: true,
        transpose_c: false,
        quantize_output: None,
        operating_dt: None,
    };
    let y = model.wire_node("mm", op, &[a, w])?[0];
    model.select_output_outlets(&[y])?;
    Ok(model)
}

fn assert_on_gpu(model: &TypedModel) -> TractResult<()> {
    let mut check = model.clone();
    WgpuTransform.transform(&mut check)?;
    check = check.into_optimized()?;
    ensure!(check.nodes().iter().any(|n| n.op_is::<WgpuGemm>()), "stage did not reach WgpuGemm");
    ensure!(
        !check.nodes().iter().any(|n| n.op_is::<PrefixMatMul>()),
        "stage left a PrefixMatMul on the CPU"
    );
    drop(check);
    DeviceContext::synchronize(&tract_wgpu::wgpu_context())?;
    Ok(())
}

struct Gate {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl Gate {
    fn enter(&self) {
        let n = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(n, Ordering::SeqCst);
        assert_eq!(n, 1, "loaded a stage while another was still resident");
    }

    fn leave(&self) {
        let before = self.live.fetch_sub(1, Ordering::SeqCst);
        assert_eq!(before, 1, "releasing a stage that was not the only resident");
    }
}

fn stage(gate: &Gate, model: TypedModel, inputs: &[Tensor]) -> TractResult<Vec<Tensor>> {
    assert_on_gpu(&model)?;
    gate.enter();
    let rt =
        tract_core::runtime::runtime_for_name("wgpu")?.context("wgpu runtime is not linked")?;
    let runnable = rt.prepare(model)?;
    let mut outs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outs.push(runnable.run(tvec!(input.clone().into()))?.remove(0).into_tensor());
    }
    drop(runnable);
    DeviceContext::synchronize(&tract_wgpu::wgpu_context())?;
    gate.leave();
    Ok(outs)
}

fn cpu_one(model: &TypedModel, input: &Tensor) -> TractResult<Tensor> {
    let plan = SimplePlan::new(model.clone())?;
    Ok(plan.run(tvec!(input.clone().into()))?.remove(0).into_tensor())
}

fn expect_close(gpu: &[Tensor], model: &TypedModel, inputs: &[Tensor]) -> TractResult<()> {
    for (g, input) in gpu.iter().zip(inputs) {
        let c = cpu_one(model, input)?;
        g.close_enough(&c, Approximation::Approximate)?;
    }
    Ok(())
}

/// Encoder, four denoising steps, decode. Returns the peak number of resident stages.
pub fn run_schedule() -> TractResult<usize> {
    let gate = Gate { live: AtomicUsize::new(0), peak: AtomicUsize::new(0) };

    let encoder = matmul(13, 64, 17, pack(&Q4_0, 17, 64, &fill(1, 17 * 64))?)?;
    let enc_in = Tensor::from_shape(&[13, 64], &fill(2, 13 * 64))?;
    let enc = stage(&gate, encoder.clone(), std::slice::from_ref(&enc_in))?;
    expect_close(&enc, &encoder, &[enc_in])?;

    let dit = matmul(9, 128, 11, pack(&Q2_0_T, 11, 128, &fill(3, 11 * 128))?)?;
    let dit_in: Vec<Tensor> = (0..4)
        .map(|step| Tensor::from_shape(&[9, 128], &fill(10 + step, 9 * 128)))
        .collect::<TractResult<_>>()?;
    let denoised = stage(&gate, dit.clone(), &dit_in)?;
    ensure!(denoised.len() == 4, "the diffusion stage did not run four steps");
    expect_close(&denoised, &dit, &dit_in)?;

    let vae =
        matmul(3, 5, 7, Const::new(Arc::new(Tensor::from_shape(&[7, 5], &fill(4, 7 * 5))?))?)?;
    let vae_in = Tensor::from_shape(&[3, 5], &fill(5, 3 * 5))?;
    let decoded = stage(&gate, vae.clone(), std::slice::from_ref(&vae_in))?;
    expect_close(&decoded, &vae, &[vae_in])?;

    ensure!(gate.live.load(Ordering::SeqCst) == 0, "a stage was left resident");
    Ok(gate.peak.load(Ordering::SeqCst))
}

#[test]
fn load_then_free_peaks_at_one_stage() -> TractResult<()> {
    ensure!(run_schedule()? == 1);
    Ok(())
}
