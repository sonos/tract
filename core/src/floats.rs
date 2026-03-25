use crate::internal::translator::Translate;
use crate::internal::*;
use crate::ops::array::{Pad, PadMode};
use crate::ops::binary::TypedBinOp;
use crate::ops::cast::{Cast, cast};
use crate::ops::einsum::EinSum;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::konst::Const;
use crate::ops::scan::Scan;
use crate::ops::source::TypedSource;
use crate::transform::ModelTransform;

pub struct FloatPrecisionTranslator {
    from_dt: DatumType,
    to_dt: DatumType,
    #[allow(clippy::type_complexity)]
    node_predicate: Option<Box<dyn Fn(&TypedNode) -> bool>>,
}

impl FloatPrecisionTranslator {
    pub fn new(from_dt: DatumType, to_dt: DatumType) -> Self {
        Self { from_dt, to_dt, node_predicate: None }
    }

    pub fn with_filter(
        from_dt: DatumType,
        to_dt: DatumType,
        node_predicate: impl Fn(&TypedNode) -> bool + 'static,
    ) -> Self {
        Self { from_dt, to_dt, node_predicate: Some(Box::new(node_predicate)) }
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
            if model.outlet_fact(mapping[i])?.datum_type == original_op_float_dt {
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
            if target.outlet_fact(o)?.datum_type == self.from_dt && is_source_output {
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
}

impl std::fmt::Debug for FloatPrecisionTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FloatPrecisionTranslator")
            .field("from", &self.from_dt)
            .field("to", &self.to_dt)
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
        let is_source = node.op_as::<TypedSource>().is_some();
        if !self.should_translate_node(node) && !is_source {
            let new_op = node.op.clone();

            let casted_inputs =
                self.cast_inputs_if_required(target, node, mapping, self.from_dt)?;
            let target_node_outlet_ids = target.wire_node(&node.name, new_op, &casted_inputs)?;

            self.cast_model_outputs_if_required(source, node, target, target_node_outlet_ids)
        } else {
            let casted_inputs = self.cast_inputs_if_required(target, node, mapping, self.to_dt)?;

            let new_op = if let Some(source_op) = node.op_as::<TypedSource>() {
                let mut fact = source_op.fact.clone();
                if fact.datum_type == self.from_dt {
                    fact.datum_type = self.to_dt;
                }
                Box::new(TypedSource::new(fact))
            } else if let Some(konst) = node.op_as::<Const>() {
                if konst.val().datum_type() == self.from_dt {
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
            } else if let Some(op) = node.op_as::<Pad>() {
                if let PadMode::Constant(t) = &op.mode {
                    let new_t = if t.datum_type() == self.from_dt {
                        t.cast_to_dt(self.to_dt)?.into_owned().into_arc_tensor()
                    } else {
                        Arc::clone(t)
                    };
                    Box::new(Pad { mode: PadMode::Constant(new_t), ..op.clone() })
                } else {
                    Box::new(op.clone())
                }
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
                .try_as_plain()?
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
                .try_as_plain()?
                .to_scalar::<f16>()?
                .is_nan()
        );

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
                .try_as_plain()?
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
                .try_as_plain()?
                .to_scalar::<f16>()?
                .is_nan()
        );
        Ok(())
    }
}
