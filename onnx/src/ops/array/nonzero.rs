use tract_hir::internal::*;
use tract_ndarray::Dimension;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct NonZero;

tract_data::impl_dyn_hash!(NonZero);

impl NonZero {
    unsafe fn eval_t<T: Datum + tract_num_traits::Zero>(input: &Tensor) -> TractResult<Tensor> {
        let count = input.as_slice_unchecked::<T>().iter().filter(|d| !d.is_zero()).count();
        let view = input.to_array_view_unchecked::<T>();
        let mut output = Tensor::uninitialized::<i64>(&[input.rank(), count])?;
        let mut view_mut: tract_ndarray::ArrayViewMut2<i64> =
            output.to_array_view_mut_unchecked::<i64>().into_dimensionality().unwrap();
        let mut i = 0;
        for (coords, _) in view.indexed_iter().filter(|(_, value)| !value.is_zero()) {
            view_mut
                .index_axis_mut(tract_ndarray::Axis(1), i)
                .assign(&coords.as_array_view().map(|d| *d as i64));
            i += 1;
        }
        Ok(output)
    }
}

impl Op for NonZero {
    fn name(&self) -> Cow<str> {
        "NonZero".into()
    }

    op_onnx!();
    not_a_typed_op!();
}

impl EvalOp for NonZero {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unsafe {
            let input = args_1!(inputs);
            let output = if input.datum_type() == bool::datum_type() {
                Self::eval_t::<u8>(input.as_ref())?
            } else {
                dispatch_numbers!(Self::eval_t(input.datum_type())(input.as_ref()))?
            };
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}

impl InferenceRulesOp for NonZero {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, i64::datum_type())?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], inputs[0].rank.bex().to_dim())?;
        Ok(())
    }

    as_op!();
}
