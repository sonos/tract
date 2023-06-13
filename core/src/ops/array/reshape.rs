use crate::internal::*;
use tract_itertools::Itertools;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct FiniteReshape {
    pub shape: TVec<usize>,
}

impl Op for FiniteReshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("to shape: {}", self.shape.iter().join(","))])
    }

    op_as_typed_op!();
}



impl EvalOp for FiniteReshape {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let mut tensor = input.into_tensor();
        unsafe {
            tensor.set_shape_unchecked(&self.shape);
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for FiniteReshape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(&self.shape)))
    }

    as_op!();
}
