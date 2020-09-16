use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::pb::NodeProto;

pub fn one_hot(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(-1);
    Ok((expand(OneHot::new(axis)), vec![]))
}

#[derive(Debug, PartialEq, Clone, new, Hash)]
struct OneHot {
    axis: i64,
}

tract_linalg::impl_dyn_hash!(OneHot);

impl Expansion for OneHot {
    fn name(&self) -> Cow<str> {
        "OneHot".into()
    }

    op_onnx!();

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        todo!();
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.equals(inputs[0].rank.bex() + 1,  &outputs[0].rank)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(&inputs[2].shape[0], 2.to_dim())?;
        s.given(&inputs[0].rank, move |s, irank| {
            let axis = if self.axis < 0 { self.axis + rank } else { self.axis };
            todo!();
            Ok(())
        })?;
        Ok(())
    }
}
