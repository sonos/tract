use crate::kernels::conv::winograd::{
    WINOGRAD_POSITIONS, WinogradGeometry, WinogradStage, dispatch_winograd_stage,
};
use crate::kernels::matmul::MlxGemm;
use crate::ops::MetalGemm;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

/// One transform of an F(6x6, 3x3) Winograd conv. See [`wire_winograd_conv`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalWinograd {
    pub stage: WinogradStage,
    pub geo: WinogradGeometry,
    pub output_dt: DatumType,
    pub output_shape: TVec<usize>,
}

impl Op for MetalWinograd {
    fn name(&self) -> StaticName {
        format!("MetalWinograd{:?}", self.stage).into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalWinograd {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.to_device_tensor()?;
        let output =
            tract_gpu::turn_handler::make_tensor_for_node(ctx, self.output_dt, &self.output_shape)?;
        crate::with_metal_stream(|stream| {
            dispatch_winograd_stage(stream, self.stage, &self.geo, input, &output)
        })?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalWinograd {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |_| {
            Ok(tvec!(self.output_dt.fact(&self.output_shape)))
        })
    }
}

/// Wires a 3x3 conv `geo` accepts as F(6x6, 3x3) Winograd: the kernel and the
/// data each go through their transform, an f32 MLX GEMM batched over the 64
/// transformed positions sums over the input channels, and the output
/// transform crops the tiles back to the conv's `output_shape`. The kernel
/// transform runs on every evaluation: kept as a constant, its 64 positions
/// would hold seven times the kernel, in f32.
pub fn wire_winograd_conv(
    target: &mut TypedModel,
    name: &str,
    geo: WinogradGeometry,
    inputs: &[OutletId],
    output_dt: DatumType,
    output_shape: &[usize],
) -> TractResult<OutletId> {
    let (c, o, tiles) = (geo.c as usize, geo.o as usize, geo.tiles());
    let stage = |stage, output_dt, output_shape: &[usize]| MetalWinograd {
        stage,
        geo,
        output_dt,
        output_shape: output_shape.into(),
    };
    let u = target.wire_node(
        format!("{name}.winograd_weights"),
        stage(WinogradStage::Weights, DatumType::F32, &[WINOGRAD_POSITIONS, c, o]),
        &[inputs[1]],
    )?[0];
    let v = target.wire_node(
        format!("{name}.winograd_input"),
        stage(WinogradStage::Input, DatumType::F32, &[WINOGRAD_POSITIONS, tiles, c]),
        &[inputs[0]],
    )?[0];
    let m = target.wire_node(
        format!("{name}.winograd_gemm"),
        MetalGemm::<MlxGemm>::new(false, false),
        &[v, u],
    )?[0];
    Ok(target.wire_node(name, stage(WinogradStage::Output, output_dt, output_shape), &[m])?[0])
}
