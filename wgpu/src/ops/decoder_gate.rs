use tract_core::internal::*;
use tract_core::ops::math::Mul;
use tract_gpu::ops::concat::GpuConcat;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::turn_handler::make_tensor_for_node;

use crate::kernels::bin_ops::wgpu_bin_op_dispatch;
use crate::kernels::shaders::EntryPoint;
use crate::kernels::shaders::GRID_LIMIT;
use crate::kernels::shaders::LayoutKind;
use crate::kernels::shaders::ShaderDtype;
use crate::kernels::shaders::WORKGROUP;
use crate::kernels::shaders::pack_u32s;
use crate::utils::element_offset;
use crate::utils::get_wgpu_buffer;

/// Applies the common sigmoid to rounded FP16 logits. This stays distinct
/// from an elementwise op that can absorb into a wider convolution epilogue.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct WgpuDecoderSigmoid(pub GpuElementWise);

impl Op for WgpuDecoderSigmoid {
    fn name(&self) -> StaticName {
        "WgpuDecoderSigmoid".into()
    }

    op_as_typed_op!();
}

impl TypedOp for WgpuDecoderSigmoid {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.0.output_facts(inputs)
    }
}

impl EvalOp for WgpuDecoderSigmoid {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.0.eval(ctx, inputs)
    }
}

/// Materializes `[features, reset_gate * previous_state]` along the channel
/// axis. Accepts batch-one FP16 inputs, with reset in the first channel half
/// of the shared sigmoid output and no broadcast in the multiplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WgpuPackedDecoderGate;

impl Op for WgpuPackedDecoderGate {
    fn name(&self) -> StaticName {
        "WgpuPackedDecoderGate".into()
    }

    op_as_typed_op!();
}

impl TypedOp for WgpuPackedDecoderGate {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [features, gates, previous] = inputs else { bail!("Expected three decoder inputs") };
        ensure!(
            matches!(features.shape.as_concrete(), Some([1, _, _, _]))
                && features.datum_type == DatumType::F16
                && features.datum_type == gates.datum_type
                && features.datum_type == previous.datum_type
                && features.shape == previous.shape
        );
        let concat = GpuConcat::new(1).output_facts(&[features, previous])?;
        ensure!(concat[0].shape == gates.shape);
        Ok(concat)
    }
}

impl EvalOp for WgpuPackedDecoderGate {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let [features, gates, previous] = inputs.as_slice() else {
            bail!("Expected three decoder inputs")
        };
        let (feature, gates, previous) =
            (features.to_device_tensor()?, gates.to_device_tensor()?, previous.to_device_tensor()?);
        let [1, c, h, w] = feature.shape() else { bail!("Expected batch-one NCHW decoder inputs") };
        let full = c.checked_mul(2).context("Too many decoder channels")?;
        ensure!(gates.shape() == [1, full, *h, *w]);
        ensure!(feature.shape() == previous.shape());
        ensure!([feature, gates, previous].iter().all(|t| t.datum_type() == DatumType::F16));
        let len = feature.len();
        let output = make_tensor_for_node(ctx, DatumType::F16, gates.shape())?;
        let tensors = [feature, gates, previous, &output];
        let offsets = tensors.map(|t| element_offset(t, 0));
        if !len.is_multiple_of(4)
            || !tensors.iter().zip(offsets).all(|(tensor, off)| {
                off.is_multiple_of(4)
                    && off.checked_add(tensor.len()).is_some_and(|end| end <= u32::MAX as usize)
                    && tensor.strides() == Tensor::natural_strides(tensor.shape()).as_slice()
            })
        {
            drop(output);
            // Keep the view inside this node and preserve the separate FP16
            // multiply when a vector load cannot represent an input's layout.
            let reset = gates.prefix_window(1, *c)?;
            let gated = DeviceTensor::uninitialized_dt(DatumType::F16, feature.shape())?;
            wgpu_bin_op_dispatch(&Mul, &reset, previous, &gated)?;
            return GpuConcat::new(1)
                .eval(ctx, tvec![features.clone(), gated.into_tensor().into_tvalue()]);
        }
        crate::with_wgpu_queue(|q| {
            for tensor in tensors {
                q.retain_tensor(tensor);
            }
            let name = "decoder_gate_concat_v4";
            let layout = LayoutKind::Chain(4);
            let pipeline = q.context().chain_pipeline(
                name,
                layout,
                EntryPoint::typed(name, ShaderDtype::F16),
                shader,
            )?;
            let bind =
                q.context().bind_group(layout, &tensors.map(get_wgpu_buffer), q.uniform())?;
            let [feature, gate, previous, dst] = offsets.map(|v| (v / 4) as u32);
            let uniform = q.alloc_uniform(&pack_u32s(&[
                feature,
                gate,
                previous,
                dst,
                (len / 4) as u32,
                0,
                0,
                0,
            ]))?;
            q.dispatch(name, &pipeline, &bind, uniform, (len / 2) as u64)
        })?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

fn shader() -> String {
    format!(
        r#"
enable f16;
struct Params {{
    off: vec4<u32>,
    half_len: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}}
@group(0) @binding(0) var<storage, read> feature: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> gates: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> previous: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size({WORKGROUP})
fn decoder_gate_concat_v4_f16(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i < params.half_len) {{
        dst[params.off.w + i] = feature[params.off.x + i];
    }} else if (i < 2u * params.half_len) {{
        let j = i - params.half_len;
        dst[params.off.w + i] = gates[params.off.y + j] * previous[params.off.z + j];
    }}
}}
"#
    )
}
