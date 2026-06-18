use crate::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_core::internal::*;

/// The only FFT length the GPU STFT kernels support (the Nemotron/Parakeet STFT frame).
pub const FFT_LEN: usize = 512;

/// Per-backend STFT kernel launcher: `(stride, input, window, output)`. `input` is
/// interleaved-complex f32 `[lead.., T, 2]`, `window` the pre-padded real window
/// `[FFT_LEN]`, `output` `[lead.., frames, FFT_LEN, 2]`.
pub type DispatchStftFn = fn(usize, &DeviceTensor, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

/// Backend-agnostic fused STFT (frame + window + forward FFT) with a fixed `FFT_LEN`
/// frame. The window is pre-padded to `[FFT_LEN]` (all-ones when the source had none),
/// matching `core::ops::fft::Stft`'s symmetric padding. The time axis must sit just
/// before the trailing complex pair (`axis == rank - 2`). Each backend supplies its own
/// `dispatch` kernel; everything else (facts, output allocation, window upload) is shared.
#[derive(Clone)]
pub struct GpuStft {
    pub axis: usize,
    pub stride: usize,
    pub window: Arc<Tensor>,
    pub backend_name: &'static str,
    pub dispatch: DispatchStftFn,
}

impl GpuStft {
    fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut shape: TVec<D> = input.into();
        let frames = (input[self.axis].clone() - FFT_LEN) / self.stride + 1;
        shape[self.axis] = frames;
        shape.insert(self.axis + 1, FFT_LEN.into());
        shape
    }
}

impl std::fmt::Debug for GpuStft {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Stft(stride={})", self.backend_name, self.stride)
    }
}

impl PartialEq for GpuStft {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.axis == other.axis
            && self.stride == other.stride
            && self.window == other.window
    }
}

impl Eq for GpuStft {}

impl std::hash::Hash for GpuStft {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
        self.stride.hash(state);
        self.window.hash(state);
    }
}

impl Op for GpuStft {
    fn name(&self) -> StaticName {
        format!("{}Stft", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuStft {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let window = (*self.window).clone().into_device()?;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &self.output_shape(input.shape()),
        )?;
        (self.dispatch)(self.stride, input, &window, &output)
            .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuStft {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let input = facts[0];
            ensure!(
                input.rank() >= 2 && input.shape[input.rank() - 1] == 2.to_dim(),
                "{} expects a complex input [.., T, 2]",
                self.name()
            );
            Ok(tvec!(input.datum_type.fact(self.output_shape(&input.shape.to_tvec()))))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

/// Pre-pad `window` (or all-ones) to `[FFT_LEN]` exactly as core `Stft` does: symmetric
/// padding when shorter than the frame. Shared by every backend's lowering rule.
pub fn padded_window(window: Option<&Arc<Tensor>>) -> TractResult<Arc<Tensor>> {
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
