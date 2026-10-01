use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::tract_core::ops::element_wise::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Cast", cast);
    reg.insert("CastLike", cast_like);
}

fn cast(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut to = node.get_attr::<DatumType>("to")?;
    if to == i64::datum_type() {
        to = TDim::datum_type();
    }
    Ok((expand(Cast { to }), vec![]))
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct Cast {
    to: DatumType,
}

impl Cast {
    fn is_string_parse(&self, from: DatumType) -> bool {
        from == String::datum_type() && self.to == f32::datum_type()
    }
}

impl Expansion for Cast {
    fn name(&self) -> StaticName {
        "onnx.Cast".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.to.bex())?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let from = model.outlet_fact(inputs[0])?.datum_type;
        if self.is_string_parse(from) {
            model.wire_node(
                prefix,
                ElementWiseOp(Box::new(StringParse { to: self.to }), None),
                &[inputs[0]],
            )
        } else {
            model.wire_node(prefix, tract_core::ops::cast::cast(self.to), &[inputs[0]])
        }
    }
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct StringParse {
    to: DatumType,
}

impl ElementWiseMiniOp for StringParse {
    fn name(&self) -> String {
        "onnx.Cast(StringParse)".into()
    }

    fn output_type(&self, _input_type: DatumType) -> Option<DatumType> {
        Some(self.to)
    }

    fn eval_out_of_place(&self, t: &Tensor, _out_dt: Option<DatumType>) -> TractResult<Tensor> {
        unsafe {
            let mut output = Tensor::uninitialized::<f32>(t.shape())?;
            let output_slice = output.as_slice_mut_unchecked();
            let input = t.as_slice_unchecked::<String>();
            for i in 0..input.len() {
                output_slice[i] = match &*input[i] {
                    "-INF" => f32::NEG_INFINITY,
                    "INF" | "+INF" => f32::INFINITY,
                    v => v.parse()?,
                };
            }
            Ok(output)
        }
    }

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}

fn cast_like(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(CastLike), vec![]))
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct CastLike;

impl Expansion for CastLike {
    fn name(&self) -> StaticName {
        "CastLike".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[1])?.datum_type;
        model.wire_node(prefix, tract_core::ops::cast::cast(dt), &[inputs[0]])
    }
}
