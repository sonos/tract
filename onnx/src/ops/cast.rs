use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::tract_core::ops::element_wise::*;

pub fn cast(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let to = node.get_attr::<DatumType>("to")?;
    Ok((Box::new(ElementWiseOp(Box::new(Cast::new(to)))), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Cast {
    to: DatumType,
}

tract_data::impl_dyn_hash!(Cast);

impl ElementWiseMiniOp for Cast {
    fn name(&self) -> String {
        "onnx.Cast".into()
    }

    fn output_type(&self, _input_type: DatumType) -> Option<DatumType> {
        Some(self.to)
    }

    fn eval_out_of_place(&self, t: &Tensor) -> TractResult<Tensor> {
        if t.datum_type() == String::datum_type() && self.to == f32::datum_type() {
            unsafe {
                let mut output = Tensor::uninitialized::<f32>(t.shape())?;
                let output_slice = output.as_slice_mut_unchecked();
                let input = t.as_slice_unchecked::<String>();
                for i in 0..input.len() {
                    output_slice[i] = match &*input[i] {
                       "-INF" => -std::f32::INFINITY,
                        "INF" | "+INF" => std::f32::INFINITY,
                        v => v.parse()?
                    };
                }
                Ok(output)
            }
        } else {
            t.cast_to_dt(self.to).map(|t| t.into_owned())
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let from = model.outlet_fact(node.inputs[0])?.datum_type;
        if from == self.to {
            Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
        } else if from == String::datum_type() && self.to == f32::datum_type() {
            Ok(None)
        } else {
            Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                tract_hir::ops::cast(self.to),
            )?))
        }
    }
}
