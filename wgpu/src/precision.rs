use std::collections::HashMap;

use tract_core::floats::FloatPrecisionTranslator;
use tract_core::internal::translator::Translate;
use tract_core::internal::*;
use tract_core::ops::cast::cast;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::source::TypedSource;
use tract_core::transform::ModelTransform;

const ACCUMULATE_F32: &str = "tract.wgpu.f16_accumulate_f32";

tract_core::declare_knob!(
    scopable TRACT_WGPU_F16_ACCUMULATE_F32,
    bool,
    true,
    "Accumulate FP16 WebGPU convolutions and matrix products in FP32. Captured when a model is prepared."
);

pub fn f16_accumulate_f32() -> bool {
    TRACT_WGPU_F16_ACCUMULATE_F32.get()
}

pub(crate) fn for_model(model: &TypedModel) -> TractResult<bool> {
    model
        .properties
        .get(ACCUMULATE_F32)
        .map(|tensor| tensor.try_as_plain_ram()?.to_scalar::<bool>().copied())
        .unwrap_or_else(|| Ok(f16_accumulate_f32()))
}

/// Opt-in FP16 internal storage with unchanged input and output types.
/// On devices without shader-f16, the graph retains its declared precision.
#[derive(Debug)]
pub struct WgpuFloatPrecision {
    pub accumulate_f32: bool,
}

impl ModelTransform for WgpuFloatPrecision {
    fn name(&self) -> StaticName {
        if self.accumulate_f32 { "wgpu_fp16_f32_accum" } else { "wgpu_fp16" }.into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        if !crate::wgpu_context().shader_f16() {
            return Ok(());
        }
        *model = InternalFloatPrecision.translate_model(model)?;
        model
            .properties
            .insert(ACCUMULATE_F32.into(), tensor0(self.accumulate_f32).into_arc_tensor());
        Ok(())
    }
}

tract_core::register_simple_model_transform!(
    "wgpu_fp16",
    WgpuFloatPrecision { accumulate_f32: false }
);
tract_core::register_simple_model_transform!(
    "wgpu_fp16_f32_accum",
    WgpuFloatPrecision { accumulate_f32: true }
);

#[derive(Debug)]
struct InternalFloatPrecision;

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>>
    for InternalFloatPrecision
{
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if node.op_is::<TypedSource>() {
            return target.wire_node(&node.name, node.op.clone(), &[]);
        }
        let outputs = if let Some(op) = node.op_as::<PrefixMatMul>() {
            let mut op = *op;
            if op.operating_dt == Some(DatumType::F32) {
                op.operating_dt = Some(DatumType::F16);
            }
            let inputs = node
                .inputs
                .iter()
                .enumerate()
                .map(|(slot, input)| {
                    let input = mapping[input];
                    if target.outlet_fact(input)?.datum_type == DatumType::F32 {
                        Ok(target.wire_node(
                            format!("{}.half-{slot}", node.name),
                            cast(DatumType::F16),
                            &[input],
                        )?[0])
                    } else {
                        Ok(input)
                    }
                })
                .collect::<TractResult<TVec<_>>>()?;
            target.wire_node(&node.name, op, &inputs)?
        } else {
            FloatPrecisionTranslator::new(DatumType::F32, DatumType::F16)
                .translate_node(source, node, target, mapping)?
        };
        outputs
            .into_iter()
            .enumerate()
            .map(|(slot, outlet)| {
                let original = OutletId::new(node.id, slot);
                let dt = node.outputs[slot].fact.datum_type;
                if source.outputs.contains(&original)
                    && target.outlet_fact(outlet)?.datum_type != dt
                {
                    Ok(target.wire_node(
                        format!("{}.public-{slot}", node.name),
                        cast(dt),
                        &[outlet],
                    )?[0])
                } else {
                    Ok(outlet)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_storage_preserves_model_interface_and_output_order() -> TractResult<()> {
        let mut original = TypedModel::default();
        let input = original.add_source("rgb", f32::fact([2]))?;
        let scale = original.add_const("scale", tensor1(&[0.3f32]))?;
        let result =
            original.wire_node("scaled", tract_core::ops::math::mul(), &[input, scale])?[0];
        original.select_output_outlets(&[result, input])?;
        let converted = InternalFloatPrecision.translate_model(&original)?;
        assert_eq!(converted.input_fact(0)?, original.input_fact(0)?);
        assert_eq!(converted.output_fact(0)?, original.output_fact(0)?);
        assert_eq!(converted.output_fact(1)?, original.output_fact(1)?);
        assert_eq!(converted.node_by_name("scaled")?.outputs[0].fact.datum_type, DatumType::F16);
        let input = tensor1(&[0.25f32, -0.5]);
        let output = converted.into_runnable()?.run(tvec!(input.clone().into()))?;
        assert_eq!(&*output[1], &input);
        assert_eq!(output[0].datum_type(), DatumType::F32);
        assert!((output[0].try_as_plain_ram()?.as_slice::<f32>()?[0] - 0.075).abs() < 0.0001);
        Ok(())
    }

    #[test]
    fn internal_storage_preserves_integer_values() -> TractResult<()> {
        let mut original = TypedModel::default();
        let input = original.add_source("index", i64::fact([2]))?;
        original.select_output_outlets(&[input])?;
        let converted = InternalFloatPrecision.translate_model(&original)?;
        let input = tensor1(&[16_777_217i64, i64::MAX]);
        let output = converted.into_runnable()?.run(tvec!(input.clone().into()))?;
        assert_eq!(&*output[0], &input);
        Ok(())
    }

    #[test]
    fn internal_storage_converts_explicit_matmul_operating_type() -> TractResult<()> {
        let mut original = TypedModel::default();
        let input = original.add_source("input", f32::fact([2, 4]))?;
        let weights = original.add_const("weights", Tensor::from_shape(&[4, 2], &[0.5f32; 8])?)?;
        let op = PrefixMatMul {
            transpose_a: false,
            transpose_b: false,
            transpose_c: false,
            quantize_output: None,
            operating_dt: Some(DatumType::F32),
        };
        let result = original.wire_node("product", op, &[input, weights])?[0];
        original.select_output_outlets(&[result])?;
        let converted = InternalFloatPrecision.translate_model(&original)?;
        let node = converted.node_by_name("product")?;
        assert_eq!(node.op_as::<PrefixMatMul>().unwrap().operating_dt, Some(DatumType::F16));
        assert_eq!(node.outputs[0].fact.datum_type, DatumType::F16);
        assert_eq!(converted.output_fact(0)?.datum_type, DatumType::F32);
        Ok(())
    }
}
