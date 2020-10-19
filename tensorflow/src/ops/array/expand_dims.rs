use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn build(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(expand(ExpandDims))
}

#[derive(Debug, Clone, Hash)]
pub struct ExpandDims;

tract_data::impl_dyn_hash!(ExpandDims);

impl Expansion for ExpandDims {
    fn name(&self) -> Cow<str> {
        "ExpandDims".into()
    }

    op_tf!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let data = &inputs[0];
        let dims = &inputs[1];
        let output = &outputs[0];

        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&dims.datum_type, DatumType::I32)?;
        s.equals(&data.datum_type, &output.datum_type)?;
        s.equals(data.rank.bex() + 1, &output.rank)?;
        s.given_2(&dims.value, &data.rank, move |s, index, rank| {
            let mut index = index.cast_to_scalar::<i64>()?;
            if index < 0 {
                index += rank + 1
            }
            let index = index as usize;

            for i in 0..index {
                s.equals(&output.shape[i], &data.shape[i])?;
            }

            s.equals(output.shape[index].bex(), 1i64.to_dim().bex())?;

            s.given(&data.rank, move |s, rank| {
                for i in index..(rank as usize) {
                    s.equals(&output.shape[i + 1], &data.shape[i])?;
                }
                Ok(())
            })
        })
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref axes) = target.outlet_fact(inputs[1])?.konst {
            let mut axes = axes
                .cast_to::<i32>()?
                .as_slice::<i32>()?
                .iter()
                .map(|&axis| {
                    Ok(if axis < 0 {
                        axis + target.outlet_fact(inputs[0])?.shape.rank() as i32
                    } else {
                        axis
                    })
                })
                .collect::<TractResult<Vec<_>>>()?;
            axes.sort();
            let mut wire = inputs[0];
            for axis in axes.iter().rev() {
                wire = target.wire_node(
                    format!("{}.axis-{}", prefix, axis),
                    AxisOp::Add(*axis as _),
                    &[wire],
                )?[0];
            }
            Ok(tvec!(wire))
        } else {
            bail!("Need axes to be const")
        }
    }
}
