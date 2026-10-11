use tract_hir::internal::*;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("CumSum", cumsum);
}

fn cumsum(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let reverse = node.get_attr_opt::<i64>("reverse")? == Some(1);
    let exclusive = node.get_attr_opt::<i64>("exclusive")? == Some(1);
    Ok((expand(CumSum { reverse, exclusive }), vec![]))
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CumSum {
    pub reverse: bool,
    pub exclusive: bool,
}

impl Expansion for CumSum {
    fn name(&self) -> StaticName {
        "CumSum".into()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let axis =
            model.outlet_fact(inputs[1])?.konst.as_ref().context("Axis expected to be a const")?;
        let axis = axis.cast_to_scalar::<i64>()?;
        let rank = model.outlet_fact(inputs[0])?.rank() as i64;
        let axis = if axis < 0 { axis + rank } else { axis } as usize;
        model.wire_node(
            prefix,
            tract_core::ops::array::CumSum {
                axis,
                exclusive: self.exclusive,
                reverse: self.reverse,
            },
            &[inputs[0]],
        )
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[1].rank, 0)?;
        Ok(())
    }
}
