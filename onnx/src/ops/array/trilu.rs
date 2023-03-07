use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn trilu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let upper: i64 = node.get_attr_opt("upper")?.unwrap_or(1);
    let has_k = node.input.len() == 2;
    Ok((expand(Trilu { upper: upper == 1, has_k }), vec![]))
}

#[derive(Debug, Clone)]
struct Trilu {
    upper: bool,
    has_k: bool,
}

impl Expansion for Trilu {
    fn name(&self) -> Cow<str> {
        "Trilu".into()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let augmented_inputs = if self.has_k {
            inputs.into()
        } else {
            let k = model.add_const(format!("{prefix}.k"), tensor0(0i64))?;
            tvec!(inputs[0], k)
        };
        model.wire_node(
            prefix,
            tract_core::ops::array::Trilu { upper: self.upper },
            &augmented_inputs,
        )
    }

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
}
