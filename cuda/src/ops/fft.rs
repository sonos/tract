use crate::kernels::fft::{CudaStft as CudaStftKernel, FFT_LEN};
use tract_core::internal::*;
use tract_core::ops::fft::Stft;
use tract_gpu::tensor::{DeviceTensorExt, IntoDevice};
use tract_gpu::utils::facts_to_device_facts;

/// STFT with a fixed `FFT_LEN` frame, fused on the GPU (frame + window + FFT).
/// The window is pre-padded to `[FFT_LEN]` (all-ones when the source had none),
/// matching `core::ops::fft::Stft`'s symmetric padding. The time axis must sit
/// just before the trailing complex pair (`axis == rank - 2`).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CudaStft {
    pub axis: usize,
    pub stride: usize,
    pub window: Arc<Tensor>,
}

impl CudaStft {
    fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut shape: TVec<D> = input.into();
        let frames = (input[self.axis].clone() - FFT_LEN) / self.stride + 1;
        shape[self.axis] = frames;
        shape.insert(self.axis + 1, FFT_LEN.into());
        shape
    }
}

impl Op for CudaStft {
    fn name(&self) -> StaticName {
        "CudaStft".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaStft {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        crate::with_cuda_stream(|stream| {
            let input = inputs[0].to_device_tensor()?;
            let window = (*self.window).clone().into_device()?;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                input.datum_type(),
                &self.output_shape(input.shape()),
            )?;
            CudaStftKernel { stride: self.stride }
                .dispatch_eval(stream, input, &window, &output)?;
            Ok(tvec!(output.into_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for CudaStft {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |facts| {
            let input = facts[0];
            ensure!(
                input.rank() >= 2 && input.shape[input.rank() - 1] == 2.to_dim(),
                "CudaStft expects a complex input [.., T, 2]"
            );
            Ok(tvec!(input.datum_type.fact(self.output_shape(&input.shape.to_tvec()))))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

/// Pre-pad `window` (or all-ones) to `[FFT_LEN]` exactly as core `Stft` does:
/// symmetric padding when shorter than the frame.
fn padded_window(window: Option<&Arc<Tensor>>) -> TractResult<Arc<Tensor>> {
    let mut win = vec![0f32; FFT_LEN];
    match window {
        Some(w) => {
            let w = w.cast_to::<f32>()?;
            let w = w.try_as_plain()?;
            let w = w.as_slice::<f32>()?;
            ensure!(w.len() <= FFT_LEN, "STFT window longer than frame");
            let pad_left = (FFT_LEN - w.len()) / 2;
            win[pad_left..pad_left + w.len()].copy_from_slice(w);
        }
        None => win.fill(1.0),
    }
    Ok(Arc::new(tensor1(&win)))
}

crate::register_cuda_op!(Stft, |source, node, op| {
    let in_fact = &source.node_input_facts(node.id)?[0];
    rule_if!(op.frame == FFT_LEN);
    rule_if!(in_fact.rank() >= 2 && op.axis == in_fact.rank() - 2);
    rule_if!(in_fact.datum_type == DatumType::F32);
    let window = padded_window(op.window.as_ref())?;
    Ok(Some(Box::new(CudaStft { axis: op.axis, stride: op.stride, window })))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::CudaTransform;
    use tract_core::transform::ModelTransform;

    #[test]
    fn stft_lowers_and_matches_cpu() -> TractResult<()> {
        let (t, frame, stride, win_len) = (2000usize, FFT_LEN, 160usize, 400usize);
        let win: Vec<f32> = (0..win_len)
            .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / win_len as f32).cos())
            .collect();

        let build = || -> TractResult<TypedModel> {
            let mut m = TypedModel::default();
            let src = m.add_source("sig", f32::fact([1, t, 2]))?;
            let stft = Stft { axis: 1, frame, stride, window: Some(Arc::new(tensor1(&win))) };
            let out = m.wire_node("stft", stft, &[src])?;
            m.select_output_outlets(&out)?;
            Ok(m)
        };

        let mut sig = vec![0f32; t * 2];
        for i in 0..t {
            sig[i * 2] = (0.05 * i as f32).sin();
        }
        let input = Tensor::from_shape(&[1, t, 2], &sig)?;

        let cpu = build()?.into_runnable()?.run(tvec!(input.clone().into_tvalue()))?;

        let mut gpu_model = build()?;
        CudaTransform.transform(&mut gpu_model)?;
        assert!(
            gpu_model.nodes().iter().any(|n| n.op_as::<CudaStft>().is_some()),
            "transform did not lower Stft to CudaStft"
        );
        let gpu = gpu_model.into_runnable()?.run(tvec!(input.into_tvalue()))?;

        let cpu = cpu[0].to_plain_array_view::<f32>()?;
        let gpu = gpu[0].to_plain_array_view::<f32>()?;
        let (cpu, gpu) = (cpu.as_slice().unwrap(), gpu.as_slice().unwrap());
        assert_eq!(cpu.len(), gpu.len());
        let max_err = cpu.iter().zip(gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_err < 1e-2, "max err {max_err} between CPU and GPU STFT");
        Ok(())
    }
}
