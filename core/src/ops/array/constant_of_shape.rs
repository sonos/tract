use ndarray::*;

use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct ConstantOfShape {
    value: Arc<Tensor>,
}

impl ConstantOfShape {
    pub fn make_from_shape<T>(&self, shape: &[usize]) -> TractResult<Arc<Tensor>>
    where
        T: Datum + Copy,
    {
        Ok(Array::<T, _>::from_elem(&*shape, *self.value.to_scalar()?).into_arc_tensor())
    }
    pub fn make<T>(&self, shape: &Arc<Tensor>) -> TractResult<Arc<Tensor>>
    where
        T: Datum + Copy,
    {
        let shape: TVec<usize> =
            shape.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|&x| x as usize).collect();
        self.make_from_shape::<T>(&*shape)
    }
}

impl Op for ConstantOfShape {
    fn name(&self) -> Cow<str> {
        "ConstantOfShape".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for ConstantOfShape {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(dispatch_numbers!(Self::make(self.value.datum_type())(self, &inputs[0]))?))
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
        s.equals(&outputs[0].datum_type, self.value.datum_type())?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[0].shape[0], outputs[0].rank.bex().to_dim())?;
        // FIXME: inputs[0] is int64 and to_dim only works on i32.
        //        s.given(&outputs[0].rank, move |s, rank| {
        //            for idx in 0..(rank as usize) {
        //                s.equals(inputs[0].value[idx].bex().to_dim(), &outputs[0].shape[idx])?;
        //            }
        //            Ok(())
        //        })?;
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
            let value = dispatch_copy!(Self::make_from_shape(self.value.datum_type())(self, &*shape))?;
            return target.wire_node(&*node.name, crate::ops::konst::Const::new(value), &[]);
        }
        bail!("shape input is variable")
    }

    inference_op_as_op!();
}
