use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn topk(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(-1i64);
    let largest = node.get_attr_opt("largest")?.unwrap_or(1i64) == 1;
    Ok((expand(Topk { axis, largest }), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash)]
struct Topk {
    axis: i64,
    largest: bool,
}

impl Expansion for Topk {
    fn name(&self) -> Cow<str> {
        "Topk".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_input_arity(outputs, 2)?;

        solver.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        solver.equals(&inputs[1].datum_type, i64::datum_type())?;
        solver.equals(&outputs[1].datum_type, i64::datum_type())?;

        solver.equals(&inputs[0].rank, &outputs[0].rank)?;
        solver.equals(&inputs[0].rank, &outputs[1].rank)?;
        solver.equals(&inputs[1].rank, 1)?;

        solver.equals(&inputs[1].shape[0], 1.to_dim())?;

        solver.given(&inputs[0].rank, move |s, rank| {
            let axis = if self.axis >= 0 { self.axis } else { self.axis + rank } as usize;
            for ix in 0..rank as usize {
                if ix != axis {
                    s.equals(&inputs[0].shape[ix], &outputs[0].shape[ix])?;
                    s.equals(&inputs[0].shape[ix], &outputs[1].shape[ix])?;
                } else {
                    s.given(&inputs[1].value[0], move |s, k| {
                        s.equals(&outputs[0].shape[ix], k.to_dim())?;
                        s.equals(&outputs[1].shape[ix], k.to_dim())?;
                        Ok(())
                    })?;
                }
            }
            Ok(())
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = model.outlet_fact(inputs[0])?;
        let rank = input.rank();
        let axis = if self.axis >= 0 { self.axis } else { self.axis + rank as i64 } as usize;
        let fallback_k = model.symbols.new_with_prefix("k").into();
        model.wire_node(
            prefix,
            tract_core::ops::array::Topk { axis, fallback_k, largest: self.largest },
            &[inputs[0], inputs[1]],
        )
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }
}
