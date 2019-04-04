use crate::internal::*;

#[derive(Debug, Clone, new, Default)]
pub struct MultiBroadcastTo;

impl MultiBroadcastTo {
    fn eval_t<T: Datum>(input: &Tensor, shape: &[usize]) -> TractResult<TVec<SharedTensor>> {
        let input = input.to_array_view::<T>()?;
        let output = input.broadcast(&*shape).ok_or("incompatible shapes")?;
        Ok(tvec![output.to_owned().into()])
    }
}

impl Op for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }
}

impl StatelessOp for MultiBroadcastTo {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (input, dims) = args_2!(inputs);
        let dims: Vec<usize> = dims.to_array_view::<i64>()?.iter().map(|i| *i as usize).collect();
        let dims = crate::broadcast::multi_broadcast(&[&*dims, &*input.shape()])
            .ok_or("incompatible shapes")?;
        dispatch_datum!(Self::eval_t(input.datum_type())(input.as_tensor(), &*dims))
    }
}

impl InferenceRulesOp for MultiBroadcastTo {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[1].datum_type, DatumType::I64)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.given(&inputs[0].shape, move |s, shape| {
            s.given(&inputs[1].value, move |s, dims| {
                let dims: Vec<TDim> =
                    dims.to_array_view::<i64>().unwrap().iter().map(|i| TDim::from(*i)).collect();
                let dims = crate::broadcast::multi_broadcast(&[&*dims, &*shape])
                    .ok_or("incompatible shapes")
                    .unwrap();
                s.equals(&outputs[0].shape, ShapeFact::from(dims))
            })
        })
    }
}
