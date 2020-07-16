use crate::internal::*;
use crate::ops::element_wise::ElementWiseOp;

pub fn cast(to: DatumType) -> ElementWiseOp {
    ElementWiseOp(Box::new(Cast { to }))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Cast {
    pub to: DatumType,
}

tract_linalg::impl_dyn_hash!(Cast);

impl ElementWiseMiniOp for Cast {
    fn name(&self) -> String {
        "Cast".into()
    }

    fn output_type(&self, _input_type: DatumType) -> Option<DatumType> {
        Some(self.to)
    }

    fn eval_out_of_place(&self, t: &Tensor) -> TractResult<Tensor> {
        t.cast_to_dt(self.to).map(|t| t.into_owned())
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let from = model.outlet_fact(node.inputs[0])?.datum_type;
        if from == self.to {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
        }
        return Ok(None)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let from = model.outlet_fact(node.inputs[0])?.datum_type;
        if (from == i8::datum_type() || from == u8::datum_type()) && self.to == f32::datum_type() {
            if let Some(patch) = super::quant::quantize_section(model, node)? {
                return Ok(Some(patch))
            }
        }
        return Ok(None)
    }
}


