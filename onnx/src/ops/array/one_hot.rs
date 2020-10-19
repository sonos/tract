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

tract_data::impl_dyn_hash!(OneHot);

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
        let dim = model.outlet_fact(inputs[1])?;
        let values = model.outlet_fact(inputs[2])?;
        if let (Some(dim), Some(values)) = (&dim.konst, &values.konst) {
            let rank = model.outlet_fact(inputs[0])?.rank();
            let axis = if self.axis < 0 { self.axis + rank as i64 + 1 } else { self.axis } as usize;
            let dim = dim.cast_to::<i64>()?;
            let dim = dim.as_slice::<i64>()?[0];
            if dim < 0 {
                bail!("Expected positive dimension, got {}", dim)
            }
            let off = values.nth(0)?;
            let on = values.nth(1)?;
            let op = tract_onnx_opl::one_hot::OneHot {
                axis,
                dim: dim as usize,
                off: off.into_arc_tensor(),
                on: on.into_arc_tensor(),
            };
            model.wire_node(prefix, op, &[inputs[0]])
        } else {
            bail!("Expected dim and value to be determined, got {:?} and {:?}", dim, values)
        }
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
        s.equals(inputs[0].rank.bex() + 1, &outputs[0].rank)?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(&inputs[2].shape[0], 2.to_dim())?;
        s.given(&inputs[0].rank, move |s, irank| {
            let axis = if self.axis < 0 { self.axis + irank + 1 } else { self.axis } as usize;
            for ix in 0..axis {
                s.equals(&inputs[0].shape[ix], &outputs[0].shape[ix])?;
            }
            for ix in axis + 1..irank as usize + 1 {
                s.equals(&inputs[0].shape[ix - 1], &outputs[0].shape[ix])?;
            }
            s.given(&inputs[1].value, move |s, value| {
                let dim = value.cast_to_scalar::<i64>()?;
                s.equals(&outputs[0].shape[axis], dim.to_dim())
            })
        })
    }
}
