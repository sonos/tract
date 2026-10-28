use crate::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_core::internal::*;

/// The power-of-two lengths with a radix-2 Stockham FFT kernel (the kernel generalizes to
/// any pow2; these cover Kaldi-style featurizers at 8/16/32/48 kHz).
pub const STOCKHAM_LENS: [usize; 4] = [256, 512, 1024, 2048];

/// Longest transform the direct DFT kernels take: the frame and its twiddle table must fit
/// in 32 KB of block (threadgroup) memory.
pub const MAX_DFT_LEN: usize = 2048;

/// Whether this length has a Stockham FFT kernel.
pub fn is_stockham_len(n: usize) -> bool {
    STOCKHAM_LENS.contains(&n)
}

/// Whether the GPU FFT and STFT kernels handle this length: the Stockham kernels cover
/// [`STOCKHAM_LENS`], the direct DFT kernels any other length up to [`MAX_DFT_LEN`]
/// (Whisper-style featurizers use 400).
pub fn is_supported_len(n: usize) -> bool {
    is_stockham_len(n) || (2..=MAX_DFT_LEN).contains(&n)
}

/// Per-backend STFT kernel launcher: `(stride, input, window, output)`. `input` is
/// interleaved-complex f32 `[lead.., T, 2]`, `window` the pre-padded real window
/// `[frame]`, `output` `[lead.., frames, frame, 2]`. The kernel reads the frame length
/// from the output shape (`output[axis + 1]`).
pub type DispatchStftFn = fn(usize, &DeviceTensor, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

/// Backend-agnostic fused STFT (frame + window + forward FFT). `frame` satisfies
/// [`is_supported_len`]; the window is pre-padded to `[frame]` (all-ones when the source had
/// none), matching `core::ops::fft::Stft`'s symmetric padding. The time axis must sit just
/// before the trailing complex pair (`axis == rank - 2`). Each backend supplies its own
/// `dispatch` kernel; everything else (facts, output allocation, window upload) is shared.
#[derive(Clone)]
pub struct GpuStft {
    pub axis: usize,
    pub frame: usize,
    pub stride: usize,
    pub window: Arc<Tensor>,
    pub backend_name: &'static str,
    pub dispatch: DispatchStftFn,
}

impl GpuStft {
    fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut shape: TVec<D> = input.into();
        let frames = (input[self.axis].clone() - self.frame) / self.stride + 1;
        shape[self.axis] = frames;
        shape.insert(self.axis + 1, self.frame.into());
        shape
    }
}

impl std::fmt::Debug for GpuStft {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Stft(frame={}, stride={})", self.backend_name, self.frame, self.stride)
    }
}

impl PartialEq for GpuStft {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.axis == other.axis
            && self.frame == other.frame
            && self.stride == other.stride
            && self.window == other.window
    }
}

impl Eq for GpuStft {}

impl std::hash::Hash for GpuStft {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
        self.frame.hash(state);
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
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let window = (*self.window).clone().into_device()?;
        let output = crate::turn_handler::make_tensor_for_node(
            ctx,
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

/// Per-backend FFT kernel launcher: `(inverse, input, output)`, both interleaved-complex
/// f32 `[lead.., N, 2]` (N a supported power of two, transformed axis at `rank-2`). The
/// inverse is UNNORMALIZED, matching core `Fft` (rustfft). The kernel reads N from the
/// input shape.
pub type DispatchFftFn = fn(bool, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

/// Backend-agnostic complex FFT over the innermost-but-one axis (the trailing dim is the
/// `[re, im]` pair); forward or `inverse`, shape-preserving. Mirrors `core::ops::fft::Fft`
/// for a supported power-of-two length with `axis == rank - 2`.
#[derive(Clone)]
pub struct GpuFft {
    pub axis: usize,
    pub inverse: bool,
    pub backend_name: &'static str,
    pub dispatch: DispatchFftFn,
}

impl std::fmt::Debug for GpuFft {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Fft({})", self.backend_name, if self.inverse { "inverse" } else { "forward" })
    }
}

impl PartialEq for GpuFft {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
            && self.axis == other.axis
            && self.inverse == other.inverse
    }
}

impl Eq for GpuFft {}

impl std::hash::Hash for GpuFft {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
        self.inverse.hash(state);
    }
}

impl Op for GpuFft {
    fn name(&self) -> StaticName {
        format!("{}Fft", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuFft {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let output =
            crate::turn_handler::make_tensor_for_node(ctx, input.datum_type(), input.shape())?;
        (self.dispatch)(self.inverse, input, &output)
            .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuFft {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let input = facts[0];
            ensure!(
                input.rank() >= 2 && input.shape[input.rank() - 1] == 2.to_dim(),
                "{} expects a complex input [.., N, 2]",
                self.name()
            );
            Ok(tvec!(input.datum_type.fact(input.shape.clone())))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

/// Pre-pad `window` (or all-ones) to `[frame]` exactly as core `Stft` does: symmetric
/// padding when shorter than the frame. Shared by every backend's lowering rule.
pub fn padded_window(window: Option<&Arc<Tensor>>, frame: usize) -> TractResult<Arc<Tensor>> {
    let mut win = vec![0f32; frame];
    match window {
        Some(w) => {
            let w = w.cast_to::<f32>()?;
            let w = w.try_as_plain_ram()?;
            let w = w.as_slice::<f32>()?;
            ensure!(w.len() <= frame, "STFT window longer than frame");
            let pad_left = (frame - w.len()) / 2;
            win[pad_left..pad_left + w.len()].copy_from_slice(w);
        }
        None => win.fill(1.0),
    }
    Ok(Arc::new(tensor1(&win)))
}
