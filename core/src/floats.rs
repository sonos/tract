use tract_num_traits::Float;

use crate::internal::translator::Translate;
use crate::internal::*;
use crate::ops::array::{Pad, PadMode};
use crate::ops::binary::TypedBinOp;
use crate::ops::cast::{cast, Cast};
use crate::ops::einsum::EinSum;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::konst::Const;
use crate::ops::scan::Scan;
use crate::ops::source::TypedSource;
use crate::transform::ModelTransform;

#[derive(Default)]
pub struct FloatPrecisionTranslator<T1: Datum + Float, T2: Datum + Float> {
    #[allow(clippy::type_complexity)]
    node_predicate: Option<Box<dyn Fn(&TypedNode) -> bool>>,
    _phantom: PhantomData<(T1, T2)>,
}

impl<T1: Datum + Float, T2: Datum + Float> FloatPrecisionTranslator<T1, T2> {
    pub fn with_filter(node_predicate: impl Fn(&TypedNode) -> bool + 'static) -> Self {
        Self { node_predicate: Some(Box::new(node_predicate)), _phantom: PhantomData }
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
            if op_float_dt == T1::datum_type() { T2::datum_type() } else { T1::datum_type() };

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
            if target.outlet_fact(o)?.datum_type == T1::datum_type() && is_source_output {
                let casted_output = target.wire_node(
                    format!("{}.cast-out-{o_idx}", node.name),
                    Cast { to: T2::datum_type() },
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

impl<T1: Datum + Float, T2: Datum + Float> std::fmt::Debug for FloatPrecisionTranslator<T1, T2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FloatPrecisionTranslator").field("_phantom", &self._phantom).finish()
    }
}

impl<T1: Datum + Float, T2: Datum + Float> ModelTransform for FloatPrecisionTranslator<T1, T2> {
    fn name(&self) -> Cow<str> {
        format!("{:?}-to-{:?}", T1::datum_type(), T2::datum_type()).into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let new = self.translate_model(model)?;
        *model = new;
        Ok(())
    }
}

impl<T1: Datum + Float, T2: Datum + Float>
    Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>>
    for FloatPrecisionTranslator<T1, T2>
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
                self.cast_inputs_if_required(target, node, mapping, T1::datum_type())?;
            let target_node_outlet_ids = target.wire_node(&node.name, new_op, &casted_inputs)?;

            self.cast_model_outputs_if_required(source, node, target, target_node_outlet_ids)
        } else {
            let casted_inputs =
                self.cast_inputs_if_required(target, node, mapping, T2::datum_type())?;

            let new_op = if let Some(source) = node.op_as::<TypedSource>() {
                Box::new(TypedSource::new(fact_float_precision_conversion::<T1, T2>(&source.fact)))
            } else if let Some(konst) = node.op_as::<Const>() {
                if konst.val().datum_type() == T1::datum_type() {
                    let wire = target.add_const(
                        format!("{}.{:?}", node.name, T1::datum_type()),
                        konst.val().clone(),
                    )?;
                    return target.wire_node(&node.name, cast(T2::datum_type()), &[wire]);
                } else {
                    node.op.clone()
                }
            } else if let Some(cast) = node.op_as::<Cast>() {
                if cast.to == T1::datum_type() {
                    Box::new(Cast { to: T2::datum_type() })
                } else {
                    node.op.clone()
                }
            } else if let Some(ew) = node.op_as::<ElementWiseOp>() {
                if ew.1 == Some(T1::datum_type()) {
                    Box::new(ElementWiseOp(ew.0.clone(), Some(T2::datum_type())))
                } else {
                    node.op.clone()
                }
            } else if let Some(bin) = node.op_as::<TypedBinOp>() {
                if bin.1 == Some(T1::datum_type()) {
                    Box::new(TypedBinOp(bin.0.clone(), Some(T2::datum_type())))
                } else {
                    node.op.clone()
                }
            } else if let Some(op) = node.op_as::<Scan>() {
                let body =
                    FloatPrecisionTranslator::<T1, T2>::default().translate_model(&op.body)?;
                Box::new(Scan { body, ..op.clone() })
            } else if let Some(op) = node.op_as::<EinSum>() {
                Box::new(EinSum {
                    operating_dt: dt_float_precision_conversion::<T1, T2>(op.operating_dt),
                    ..op.clone()
                })
            } else if let Some(op) = node.op_as::<Pad>() {
                if let PadMode::Constant(t) = &op.mode {
                    Box::new(Pad {
                        mode: PadMode::Constant(tensor_float_precision_conversion::<T1, T2>(t)),
                        ..op.clone()
                    })
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

fn dt_float_precision_conversion<T1: Datum + Float, T2: Datum + Float>(dt: DatumType) -> DatumType {
    if dt == T1::datum_type() {
        T2::datum_type()
    } else {
        dt
    }
}

fn fact_float_precision_conversion<T1: Datum + Float, T2: Datum + Float>(
    t: &TypedFact,
) -> TypedFact {
    if t.datum_type == T1::datum_type() {
        let mut t = t.clone();
        t.datum_type = T2::datum_type();
        t
    } else {
        t.clone()
    }
}

fn tensor_float_precision_conversion<T1: Datum + Float, T2: Datum + Float>(
    t: &Arc<Tensor>,
) -> Arc<Tensor> {
    if t.datum_type() == T1::datum_type() {
        t.cast_to::<T2>().unwrap().into_owned().into_arc_tensor()
    } else {
        Arc::clone(t)
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
        let runnable_model = &crate::transform::get_transform("f32-to-f16")
            .unwrap()
            .transform_into(model.clone())?
            .into_runnable()?;
        assert!(runnable_model.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());

        // Execution in F16 with filter that returns the good output.
        let runnable_model = &crate::transform::get_transform("f32-to-f16!=layer.1")
            .unwrap()
            .transform_into(model.clone())?
            .into_runnable()?;
        assert_eq!(
            runnable_model.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0],
            tensor1(&[f16::NEG_INFINITY]).into()
        );

        // Execution in F16 with returns NaN despite the filter.
        let runnable_model = &crate::transform::get_transform("f32-to-f16!=layer.0")
            .unwrap()
            .transform_into(model)?
            .into_runnable()?;
        assert!(runnable_model.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());

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
        model_f16.transform(&FloatPrecisionTranslator::<f32, f16>::default())?;
        let runnable_model_f16 = model_f16.clone().into_runnable()?;
        assert!(runnable_model_f16.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());

        // Execution in F16 with filter that returns the good output.
        let mut model_f16_with_filter = model.clone();
        model_f16_with_filter.transform(&FloatPrecisionTranslator::<f32, f16>::with_filter(
            |node| !node.name.contains("layer.1"),
        ))?;
        let runnable_model_f16 = model_f16_with_filter.clone().into_runnable()?;
        assert_eq!(
            runnable_model_f16.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0],
            tensor1(&[f16::NEG_INFINITY]).into()
        );
        let mut model_f16_with_filter = model.clone();
        model_f16_with_filter.transform(&FloatPrecisionTranslator::<f32, f16>::with_filter(
            |node| !node.name.contains("layer.0"),
        ))?;
        let runnable_model_f16 = model_f16_with_filter.clone().into_runnable()?;
        assert!(runnable_model_f16.run(tvec![tensor1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());
        Ok(())
    }
}
