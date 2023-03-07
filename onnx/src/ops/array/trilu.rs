use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn trilu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let upper: i64 = node.get_attr_opt("upper")?.unwrap_or(1);
    let has_k = node.input.len() == 2;
    Ok((Box::new(Trilu { upper: upper == 1, has_k }), vec![]))
}

#[derive(Debug, Clone)]
struct Trilu {
    upper: bool,
    has_k: bool,
}

impl Op for Trilu {
    fn name(&self) -> Cow<str> {
        "Trilu".into()
    }

    not_a_typed_op!();
}

impl EvalOp for Trilu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (mut input, k) = if self.has_k {
            let (input, k) = args_2!(inputs);
            let k = *k.to_scalar::<i64>()?;
            (input.into_tensor(), k)
        } else {
            (args_1!(inputs).into_tensor(), 0)
        };
        fn eval_t<T: Datum>(tensor: &mut Tensor, upper: bool, k: i64) -> TractResult<()> {
            let mut view = tensor.to_array_view_mut::<T>()?;
            for coords in tract_ndarray::indices(view.shape()) {
                let row = coords[view.ndim() - 2] as i64;
                let col = coords[view.ndim() - 1] as i64;
                if upper {
                    if col < row + k {
                        view[coords] = T::default();
                    }
                } else {
                    if col > row + k {
                        view[coords] = T::default();
                    }
                }
            }
            Ok(())
        }
        dispatch_datum!(eval_t(input.datum_type())(&mut input, self.upper, k))?;
        Ok(tvec!(input.into_tvalue()))
    }
}

impl InferenceRulesOp for Trilu {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1 + self.has_k as usize)?;
        check_output_arity(outputs, 1)?;
        solver.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        solver.equals(&inputs[0].shape, &outputs[0].shape)?;
        if self.has_k {
            solver.equals(&inputs[1].datum_type, i64::datum_type())?;
            solver.equals(&inputs[1].rank, 0)?;
        }
        Ok(())
    }

    as_op!();
}
