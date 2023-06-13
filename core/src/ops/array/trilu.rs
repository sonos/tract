use crate::internal::*;

#[derive(Debug, Clone)]
pub struct Trilu {
    pub upper: bool,
}

impl Op for Trilu {
    fn name(&self) -> Cow<str> {
        "Trilu".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Trilu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, k) = args_2!(inputs);
        let mut input = input.into_tensor();
        let k = *k.to_scalar::<i64>()?;
        fn eval_t<T: Datum>(tensor: &mut Tensor, upper: bool, k: i64) -> TractResult<()> {
            let mut view = tensor.to_array_view_mut::<T>()?;
            for coords in tract_ndarray::indices(view.shape()) {
                let row = coords[view.ndim() - 2] as i64;
                let col = coords[view.ndim() - 1] as i64;
                if upper {
                    if col < row + k {
                        view[coords] = T::default();
                    }
                } else if col > row + k {
                    view[coords] = T::default();
                }
            }
            Ok(())
        }
        dispatch_datum!(eval_t(input.datum_type())(&mut input, self.upper, k))?;
        Ok(tvec!(input.into_tvalue()))
    }
}

impl TypedOp for Trilu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}
