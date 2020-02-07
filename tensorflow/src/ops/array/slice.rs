use tract_core::internal::*;
use tract_core::infer::*;
use tract_core::ndarray;

#[derive(Debug, Clone, new, Default)]
pub struct Slice;

impl Slice {
    pub fn eval_t<T: Datum>(
        &self,
        input: Arc<Tensor>,
        begin: &[i32],
        size: &[i32],
    ) -> TractResult<Arc<Tensor>> {
        let mut input = input.into_tensor().into_array::<T>()?;
        for i in 0..begin.len() {
            let b = begin[i];
            let e = begin[i] + size[i];
            if e > input.shape()[i] as i32 {
                bail!(
                    "on axis {} of length {}, invalid slice required: begin={} size={}",
                    i,
                    input.shape()[i],
                    begin[i],
                    size[i]
                );
            }
            input.slice_axis_inplace(ndarray::Axis(i), (b..e).into());
        }
        Ok(input.into_arc_tensor())
    }
}

impl Op for Slice {
    fn name(&self) -> Cow<str> {
        "tf.Slice".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Slice {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, begin, size) = args_3!(inputs);
        let begin = begin.cast_to::<i32>()?;
        let size = size.cast_to::<i32>()?;
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(
            self,
            input,
            begin.as_slice::<i32>()?,
            size.as_slice::<i32>()?
        ))?))
    }
}

impl InferenceRulesOp for Slice {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].shape[0], inputs[0].rank.bex().to_dim())?;
        s.equals(&inputs[1].shape, &inputs[2].shape)?;
        s.given(&inputs[2].value, move |s, sizes| {
            let sizes = sizes.cast_to::<i32>()?;
            let sizes = sizes.as_slice::<i32>()?;
            sizes
                .iter()
                .enumerate()
                .try_for_each(|(axis, dim)| s.equals(&outputs[0].shape[axis], dim.to_dim()))
        })?;
        Ok(())
    }

    inference_op_as_op!();
}
