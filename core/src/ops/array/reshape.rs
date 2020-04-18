use crate::internal::*;
use itertools::Itertools;

// FIXME: try to recanonicalize as flatten (maybe extended) / add_dims / rm_dims ?

#[derive(Debug, Clone, new, Default, Hash)]
pub struct TypedReshape {
    shape: TVec<TDim>,
}
tract_linalg::impl_dyn_hash!(TypedReshape);

impl Op for TypedReshape {
    fn name(&self) -> Cow<str> {
        "TypedReshape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("to shape: {}", self.shape.iter().join("x"))])
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for TypedReshape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let shape: TVec<usize> =
            self.shape.iter().map(|d| Ok(d.to_integer()? as usize)).collect::<TractResult<_>>()?;
        let o = unsafe { input.into_tensor().into_shape(&*shape)?.into_arc_tensor() };
        Ok(tvec!(o))
    }
}

impl TypedOp for TypedReshape {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.shape)?))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape.to_tvec() == self.shape {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        } else if let Ok(shape) =
            self.shape.iter().map(|d| Ok(d.to_integer()? as usize)).collect::<TractResult<_>>()
        {
            return Ok(Some(TypedModelPatch::single_unary_op(
                model,
                node,
                FiniteReshape::new(shape),
            )?));
        }
        Ok(None)
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct FiniteReshape {
    pub shape: TVec<usize>,
}

impl Op for FiniteReshape {
    fn name(&self) -> Cow<str> {
        "FiniteReshape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("to shape: {}", self.shape.iter().join("x"))])
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

tract_linalg::impl_dyn_hash!(FiniteReshape);

impl StatelessOp for FiniteReshape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let o = unsafe { input.into_tensor().into_shape(&*self.shape)?.into_arc_tensor() };
        Ok(tvec!(o))
    }
}

impl TypedOp for FiniteReshape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.shape)?))
    }

    as_op!();
}
