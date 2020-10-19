use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;

pub fn build(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(expand(ConcatV2))
}

#[derive(Debug, Clone, new, Hash)]
pub struct ConcatV2;

tract_data::impl_dyn_hash!(ConcatV2);

impl Expansion for ConcatV2 {
    fn name(&self) -> Cow<str> {
        "ConcatV2".into()
    }

    op_tf!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        let n = inputs.len() - 1;
        s.equals_all((0..n).map(|i| (&inputs[i].datum_type).bex()).collect())?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[n].datum_type, DatumType::I32)?;
        s.equals_all((0..n).map(|i| (&inputs[i].rank).bex()).collect())?;
        s.equals(&inputs[n].rank, 0)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[n].value, move |s, axis| {
            let axis = *axis.to_scalar::<i32>()? as usize;
            trace!("axis for ConcatV2: {}", axis);
            for d in 0..axis {
                s.equals_all((0..n).map(|i| (&inputs[i].shape[d]).bex()).collect())?;
            }
            for d in 0..axis {
                s.equals(&inputs[0].shape[d], &outputs[0].shape[d])?;
            }
            s.given(&inputs[0].rank, move |s, rank| {
                trace!("Given rank {}", rank);
                for d in (axis + 1)..(rank as usize) {
                    s.equals(&inputs[0].shape[d], &outputs[0].shape[d])?;
                }
                for d in (axis + 1)..(rank as usize) {
                    s.equals_all((0..n).map(|i| (&inputs[i].shape[d]).bex()).collect())?;
                }
                Ok(())
            })?;

            let mut concat_dim = -1 * outputs[0].shape[axis].bex();
            for i in 0..n {
                concat_dim = concat_dim + inputs[i].shape[axis].bex();
            }
            s.equals_zero(concat_dim)
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref axis) = model.outlet_fact(*inputs.last().unwrap())?.konst {
            let axis = *axis.to_scalar::<i32>()? as usize;
            let inputs = inputs.into_iter().copied().rev().skip(1).rev().collect::<TVec<_>>();
            model.wire_node(
                prefix,
                tract_hir::tract_core::ops::array::TypedConcat::concat_vars(axis, inputs.len()),
                &inputs,
            )
        } else {
            bail!("Except axis to be a constant")
        }
    }
}
