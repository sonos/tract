use crate::internal::translator::Translate;
use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::cast::{Cast, cast};
use crate::ops::einsum::EinSum;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::konst::Const;
use crate::ops::nn::resize::Resize;
use crate::ops::scan::Scan;
use crate::ops::source::TypedSource;
use crate::transform::ModelTransform;

pub struct FloatPrecisionTranslator {
    from_dt: DatumType,
    to_dt: DatumType,
    #[allow(clippy::type_complexity)]
    node_predicate: Option<Box<dyn Fn(&TypedNode) -> bool>>,
    keep_io: bool,
}

impl FloatPrecisionTranslator {
    pub fn new(from_dt: DatumType, to_dt: DatumType) -> Self {
        Self { from_dt, to_dt, node_predicate: None, keep_io: false }
    }

    pub fn with_filter(
        from_dt: DatumType,
        to_dt: DatumType,
        node_predicate: impl Fn(&TypedNode) -> bool + 'static,
    ) -> Self {
        Self { from_dt, to_dt, node_predicate: Some(Box::new(node_predicate)), keep_io: false }
    }

    /// With `keep_io`, the translated model keeps the original's input and
    /// output types: sources stay at their declared type and are cast where a
    /// translated node consumes them, and every model output is cast back to
    /// its declared type.
    pub fn keeping_io(self, keep_io: bool) -> Self {
        Self { keep_io, ..self }
    }

    fn should_translate_node(&self, node: &TypedNode) -> bool {
        self.node_predicate.as_ref().map(|it| (it)(node)).unwrap_or(true)
    }

    /// Cast node inputs to the working float precision for the operator
    /// Only input using float datumtype are impacted. This will add cast operations
    /// in the model. The function return the new input outlet ids.
    fn cast_inputs_if_required(
        &self,
        model: &mut TypedModel,
        node: &TypedNode,
        mapping: &HashMap<OutletId, OutletId>,
        op_float_dt: DatumType,
    ) -> TractResult<TVec<OutletId>> {
        let original_op_float_dt =
            if op_float_dt == self.from_dt { self.to_dt } else { self.from_dt };

        let mut mapped_inputs = tvec![];
        for (i_idx, i) in node.inputs.iter().enumerate() {
            let fact = model.outlet_fact(mapping[i])?;
            if fact.datum_type == original_op_float_dt && fact.is_plain() {
                let casted_mapped_input = model.wire_node(
                    format!("{}.cast-{i_idx}", node.name),
                    Cast { to: op_float_dt },
                    &[mapping[i]],
                )?[0];
                mapped_inputs.push(casted_mapped_input);
            } else {
                mapped_inputs.push(mapping[i])
            }
        }
        Ok(mapped_inputs)
    }

    /// Cast node output outlet ids to the destination float precision,
    /// after insertion in the target mode. This preserves the model output float
    /// precision.
    fn cast_model_outputs_if_required(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        target_node_outlet_ids: TVec<OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let mut outputs = tvec![];
        for (o_idx, o) in target_node_outlet_ids.into_iter().enumerate() {
            // Add Cast op for model output
            let is_source_output = source.outputs.contains(&OutletId::new(node.id, o_idx));
            let fact = target.outlet_fact(o)?;
            if fact.datum_type == self.from_dt && fact.is_plain() && is_source_output {
                let casted_output = target.wire_node(
                    format!("{}.cast-out-{o_idx}", node.name),
                    Cast { to: self.to_dt },
                    &[o],
                )?[0];
                outputs.push(casted_output);
            } else {
                outputs.push(o)
            }
        }
        Ok(outputs)
    }

    /// Resize scales determine the integer output shape: a constant scales
    /// input keeps its declared type so rounding cannot change the geometry.
    fn keep_resize_scales(
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        inputs: &mut TVec<OutletId>,
    ) -> TractResult<()> {
        if let Some(slot) = node.op_as::<Resize>().and_then(|op| op.optional_scales_input)
            && let Some(scales) = source.outlet_fact(node.inputs[slot])?.konst.clone()
        {
            inputs[slot] = target.add_const(format!("{}.scales", node.name), scales)?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for FloatPrecisionTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FloatPrecisionTranslator")
            .field("from", &self.from_dt)
            .field("to", &self.to_dt)
            .field("keep_io", &self.keep_io)
            .finish()
    }
}

impl ModelTransform for FloatPrecisionTranslator {
    fn name(&self) -> StaticName {
        format!("{:?}-to-{:?}", self.from_dt, self.to_dt).into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let new = self.translate_model(model)?;
        *model = new;
        Ok(())
    }
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>>
    for FloatPrecisionTranslator
{
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let outlets = self.wire_translated(source, node, target, mapping)?;
        if !self.keep_io {
            return Ok(outlets);
        }
        outlets
            .into_iter()
            .enumerate()
            .map(|(slot, outlet)| {
                let declared = node.outputs[slot].fact.datum_type;
                let fact = target.outlet_fact(outlet)?;
                if source.outputs.contains(&OutletId::new(node.id, slot))
                    && fact.datum_type != declared
                    && fact.is_plain()
                {
                    Ok(target.wire_node(
                        format!("{}.cast-out-{slot}", node.name),
                        cast(declared),
                        &[outlet],
                    )?[0])
                } else {
                    Ok(outlet)
                }
            })
            .collect()
    }
}

impl FloatPrecisionTranslator {
    fn wire_translated(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let is_source = node.op_as::<TypedSource>().is_some();
        if !self.should_translate_node(node) && !is_source {
            let new_op = node.op.clone();

            let mut casted_inputs =
                self.cast_inputs_if_required(target, node, mapping, self.from_dt)?;
            Self::keep_resize_scales(source, node, target, &mut casted_inputs)?;
            let target_node_outlet_ids = target.wire_node(&node.name, new_op, &casted_inputs)?;
            if self.keep_io {
                return Ok(target_node_outlet_ids);
            }
            self.cast_model_outputs_if_required(source, node, target, target_node_outlet_ids)
        } else {
            let mut casted_inputs =
                self.cast_inputs_if_required(target, node, mapping, self.to_dt)?;
            Self::keep_resize_scales(source, node, target, &mut casted_inputs)?;

            let new_op = if let Some(source_op) = node.op_as::<TypedSource>() {
                let mut fact = source_op.fact.clone();
                if fact.datum_type == self.from_dt && !self.keep_io {
                    fact.datum_type = self.to_dt;
                }
                Box::new(TypedSource::new(fact))
            } else if let Some(konst) = node.op_as::<Const>() {
                if konst.val().datum_type() == self.from_dt && konst.val().is_plain_ram() {
                    let wire = target.add_const(
                        format!("{}.{:?}", node.name, self.from_dt),
                        konst.val().clone(),
                    )?;
                    return target.wire_node(&node.name, cast(self.to_dt), &[wire]);
                } else {
                    node.op.clone()
                }
            } else if let Some(cast_op) = node.op_as::<Cast>() {
                if cast_op.to == self.from_dt {
                    Box::new(Cast { to: self.to_dt })
                } else {
                    node.op.clone()
                }
            } else if let Some(ew) = node.op_as::<ElementWiseOp>() {
                if ew.1 == Some(self.from_dt) {
                    Box::new(ElementWiseOp(ew.0.clone(), Some(self.to_dt)))
                } else {
                    node.op.clone()
                }
            } else if let Some(bin) = node.op_as::<TypedBinOp>() {
                if bin.1 == Some(self.from_dt) {
                    Box::new(TypedBinOp(bin.0.clone(), Some(self.to_dt)))
                } else {
                    node.op.clone()
                }
            } else if let Some(op) = node.op_as::<Scan>() {
                let body = FloatPrecisionTranslator::new(self.from_dt, self.to_dt)
                    .translate_model(&op.body)?;
                Box::new(Scan { body, ..op.clone() })
            } else if let Some(op) = node.op_as::<EinSum>() {
                let operating_dt =
                    if op.operating_dt == self.from_dt { self.to_dt } else { op.operating_dt };
                Box::new(EinSum { operating_dt, ..op.clone() })
            } else {
                node.op.clone()
            };
            target.wire_node(&node.name, new_op, &casted_inputs)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::math;
    use tract_data::prelude::f16;

    fn build_f32_model() -> TractResult<TypedModel> {
        // F32 model definition
        let mut model = TypedModel::default();
        let a = model.add_source("source", f32::fact([1])).unwrap();
        let multiplier = model.add_const("multiplier", tensor1(&[1.0f32]))?;
        let neg_infinity = model.add_const("neg_infinity", tensor1(&[f32::NEG_INFINITY]))?;
        let pow_factor = model.add_const("pow_factor", tensor1(&[10.0f32]))?;
        let add = model.wire_node("layer.0/add", math::add(), &[a, a]).unwrap()[0];
        let mul = model.wire_node("layer.0/mul", math::mul(), &[add, multiplier]).unwrap()[0];
        let pow = model.wire_node("layer.1/pow", math::pow(), &[mul, pow_factor]).unwrap()[0];
        let _output = model
            .wire_node("layer.1/add_neg_infinity", math::add(), &[pow, neg_infinity])
            .unwrap()[0];
        model.auto_outputs()?;
        Ok(model)
    }

    #[test]
    fn test_high_level_f16_transform_with_filter() -> TractResult<()> {
        // F32 model definition
        let model = build_f32_model()?;

        // Execution in F32
        let runnable_model = model.clone().into_runnable()?;
        assert_eq!(
            runnable_model.run(tvec![tensor1(&[5.0f32]).into()])?[0],
            tensor1(&[f32::NEG_INFINITY]).into()
        );

        // Execution in F16 with returns NaN
        let runnable_model = &crate::transform::get_transform("f32_to_f16")?
            .unwrap()
            .transform_into(model.clone())?
            .into_runnable()?;
        assert!(
            runnable_model.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
                .try_as_plain_ram()?
                .to_scalar::<f16>()?
                .is_nan()
        );

        // Execution in F16 with filter that returns the good output.
        let runnable_model = &crate::transform::build_float_translator(
            f32::datum_type(),
            f16::datum_type(),
            crate::transform::NodeFilter {
                exclude: Some(vec!["layer.1".into()]),
                ..Default::default()
            },
        )
        .transform_into(model.clone())?
        .into_runnable()?;
        assert_eq!(
            runnable_model.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0],
            tensor1(&[f16::NEG_INFINITY]).into()
        );

        // Execution in F16 with returns NaN despite the filter.
        let runnable_model = &crate::transform::build_float_translator(
            f32::datum_type(),
            f16::datum_type(),
            crate::transform::NodeFilter {
                exclude: Some(vec!["layer.0".into()]),
                ..Default::default()
            },
        )
        .transform_into(model)?
        .into_runnable()?;
        assert!(
            runnable_model.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
                .try_as_plain_ram()?
                .to_scalar::<f16>()?
                .is_nan()
        );

        Ok(())
    }

    #[test]
    fn f16_transform_keeps_resize_geometry() -> TractResult<()> {
        use crate::ops::nn::resize::{CoordTransformer, Interpolator, Nearest};
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::fact([1, 3, 1080, 1920]))?;
        let scales = model.add_const("scales", tensor1(&[1f32, 1., 0.4, 0.4]))?;
        let resize = Resize {
            coord_transformer: CoordTransformer::HalfPixel,
            interpolator: Interpolator::Linear,
            nearest: Nearest::RoundPreferCeil,
            optional_scales_input: Some(1),
            optional_sizes_input: None,
        };
        let output = model.wire_node("resize", resize, &[input, scales])?;
        model.select_output_outlets(&output)?;
        let expected = model.output_fact(0)?.shape.clone();
        assert_eq!(expected.as_concrete(), Some(&[1, 3, 432, 768][..]));

        let translated = crate::transform::get_transform("f32_to_f16")?
            .unwrap()
            .transform_into(model.clone())?;
        assert_eq!(translated.output_fact(0)?.shape, expected);
        assert_eq!(translated.output_fact(0)?.datum_type, f16::datum_type());

        let excluded =
            FloatPrecisionTranslator::with_filter(f32::datum_type(), f16::datum_type(), |node| {
                node.name != "resize"
            })
            .transform_into(model)?;
        assert_eq!(excluded.output_fact(0)?.shape, expected);
        Ok(())
    }

    #[test]
    fn test_f16_transform_with_filter() -> TractResult<()> {
        // F32 model definition
        let model = build_f32_model()?;

        // Execution in F32
        let runnable_model = model.clone().into_runnable()?;
        assert_eq!(
            runnable_model.run(tvec![tensor1(&[5.0f32]).into()])?[0],
            tensor1(&[f32::NEG_INFINITY]).into()
        );

        // Execution in F16 with returns NaN
        let mut model_f16 = model.clone();
        model_f16
            .transform(&FloatPrecisionTranslator::new(f32::datum_type(), f16::datum_type()))?;
        let runnable_model_f16 = model_f16.clone().into_runnable()?;
        assert!(
            runnable_model_f16.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
                .try_as_plain_ram()?
                .to_scalar::<f16>()?
                .is_nan()
        );

        // Execution in F16 with filter that returns the good output.
        let mut model_f16_with_filter = model.clone();
        model_f16_with_filter.transform(&FloatPrecisionTranslator::with_filter(
            f32::datum_type(),
            f16::datum_type(),
            |node| !node.name.contains("layer.1"),
        ))?;
        let runnable_model_f16 = model_f16_with_filter.clone().into_runnable()?;
        assert_eq!(
            runnable_model_f16.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0],
            tensor1(&[f16::NEG_INFINITY]).into()
        );
        let mut model_f16_with_filter = model.clone();
        model_f16_with_filter.transform(&FloatPrecisionTranslator::with_filter(
            f32::datum_type(),
            f16::datum_type(),
            |node| !node.name.contains("layer.0"),
        ))?;
        let runnable_model_f16 = model_f16_with_filter.clone().into_runnable()?;
        assert!(
            runnable_model_f16.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
                .try_as_plain_ram()?
                .to_scalar::<f16>()?
                .is_nan()
        );
        Ok(())
    }

    #[test]
    fn keep_io_preserves_model_interface() -> TractResult<()> {
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::fact([2]))?;
        let scale = model.add_const("scale", tensor1(&[0.25f32]))?;
        let scaled = model.wire_node("scaled", math::mul(), &[input, scale])?[0];
        model.select_output_outlets(&[scaled, input])?;

        let translated = FloatPrecisionTranslator::new(f32::datum_type(), f16::datum_type())
            .keeping_io(true)
            .transform_into(model.clone())?;
        assert_eq!(translated.input_fact(0)?, model.input_fact(0)?);
        assert_eq!(translated.output_fact(0)?, model.output_fact(0)?);
        assert_eq!(translated.output_fact(1)?, model.output_fact(1)?);
        let scaled = translated.node_by_name("scaled")?;
        assert_eq!(scaled.outputs[0].fact.datum_type, f16::datum_type());

        let input = tensor1(&[2.0f32, -6.0]);
        let outputs = translated.into_runnable()?.run(tvec![input.clone().into_tvalue()])?;
        assert_eq!(*outputs[0], tensor1(&[0.5f32, -1.5]));
        assert_eq!(*outputs[1], input);
        Ok(())
    }

    #[test]
    fn keep_io_leaves_excluded_outputs_alone() -> TractResult<()> {
        let model = build_f32_model()?;
        let translated =
            FloatPrecisionTranslator::with_filter(f32::datum_type(), f16::datum_type(), |node| {
                !node.name.contains("layer.1")
            })
            .keeping_io(true)
            .transform_into(model.clone())?;
        assert_eq!(translated.input_fact(0)?, model.input_fact(0)?);
        assert_eq!(translated.output_fact(0)?, model.output_fact(0)?);
        assert!(!translated.nodes.iter().any(|n| n.name.contains("cast-out")));
        let outputs = translated.into_runnable()?.run(tvec![tensor1(&[5.0f32]).into_tvalue()])?;
        assert_eq!(*outputs[0], tensor1(&[f32::NEG_INFINITY]));
        Ok(())
    }
}
