use std::ops::Range;

use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn shape(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let start = node.get_attr_opt("start")?.unwrap_or(0);
    let end = node.get_attr_opt("end")?;
    Ok((expand(Shape { start, end }), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
struct Shape {
    start: i64,
    end: Option<i64>,
}

impl Shape {
    fn resolve(&self, rank: i64) -> Range<usize> {
        let clamp =
            |axis: i64| (if axis >= 0 { axis } else { axis + rank }).clamp(0, rank) as usize;
        let start = clamp(self.start);
        let end = self.end.map(clamp).unwrap_or(rank as usize);
        // Both bounds are clamped to [0, rank]; a start past the end selects nothing.
        start..end.max(start)
    }
}

impl Expansion for Shape {
    fn name(&self) -> StaticName {
        "Shape".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].rank, 1)?;
        s.equals(&outputs[0].datum_type, TDim::datum_type())?;
        s.given(&inputs[0].shape, |s, shape| {
            let rank = shape.len() as i64;
            let range = self.resolve(rank);
            s.equals(&outputs[0].value, rctensor1(&shape[range]))?;
            Ok(())
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(inputs[0])?;
        let range = self.resolve(fact.rank() as i64);
        let shape = fact.shape.to_tvec();
        let wire = model.add_const(prefix, tensor1(&shape[range]))?;
        Ok(tvec!(wire))
    }
}
