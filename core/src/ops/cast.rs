use crate::internal::*;
use crate::ops::element_wise::ElementWiseOp;

pub fn cast(to: DatumType) -> ElementWiseOp {
    ElementWiseOp(Box::new(Cast { to }))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Cast {
    pub to: DatumType,
}

tract_data::impl_dyn_hash!(Cast);

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
        if model.outlet_fact(node.inputs[0])?.datum_type == self.to {
            Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
        } else {
            Ok(None)
        }
    }
}
