use ndarray::*;

use crate::internal::*;
use crate::infer::*;

#[derive(Debug, Clone, new)]
pub struct ConstantOfShape {
    scalar: Arc<Tensor>,
}

impl Op for ConstantOfShape {
    fn name(&self) -> Cow<str> {
        "ConstantOfShape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.scalar)])
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for ConstantOfShape {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape = inputs[0].cast_to::<i64>().chain_err(|| TractErrorKind::StreamTensor)?;
        let shape: TVec<usize> = shape.as_slice::<i64>()?.iter().map(|&x| x as usize).collect();
        Ok(tvec!(dispatch_numbers!(make_from_shape(self.scalar.datum_type())(
            &shape,
            &*self.scalar
        ))?))
    }
}

impl InferenceRulesOp for ConstantOfShape {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.scalar.datum_type())?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[0].shape[0], outputs[0].rank.bex().to_dim())?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref fact) = target.outlet_fact(mapping[&node.inputs[0]])?.konst {
            let shape = fact.cast_to::<i32>()?;
            let shape = shape.as_slice::<i32>()?.iter().map(|&s| s as usize).collect::<TVec<_>>();
            let value =
                dispatch_copy!(make_from_shape(self.scalar.datum_type())(&*shape, &self.scalar))?;
            return target.wire_node(&*node.name, crate::ops::konst::Const::new(value), &[]);
        }
        bail!("shape input is variable")
    }

    as_op!();
}

fn make_from_shape<T>(shape: &[usize], scalar: &Tensor) -> TractResult<Arc<Tensor>>
where
    T: Datum + Copy,
{
    Ok(Array::<T, _>::from_elem(&*shape, *scalar.to_scalar()?).into_arc_tensor())
}
