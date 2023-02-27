use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn trilu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let upper: i64 = node.get_attr_opt("upper")?.unwrap_or(1);
    Ok((Box::new(Trilu { upper: upper == 1 }), vec![]))
}

#[derive(Debug, Clone)]
struct Trilu {
    upper: bool,
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
        let mut input = args_1!(inputs).into_tensor();
        let mut view = input.to_array_view_mut::<i64>()?;
        for coords in tract_ndarray::indices(view.shape()) {
            let row = coords[view.ndim() - 2];
            let col = coords[view.ndim() - 1];
            if self.upper {
                if col < row {
                    view[coords] = 0;
                }
            } else {
                if col > row {
                    view[coords] = 0;
                }
            }
        }
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
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        solver.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        solver.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    as_op!();
}
