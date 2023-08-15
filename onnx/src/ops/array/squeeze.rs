use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::pb::*;

pub fn squeeze(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    if ctx.onnx_operator_set_version < 13 {
        let axes = node.get_attr_vec::<i64>("axes")?.into_iter().map(|x| x as isize).collect();
        Ok((expand(tract_hir::ops::array::Squeeze::new(Some(axes))), vec![]))
    } else {
        Ok((expand(Squeeze13), vec![]))
    }
}

#[derive(Debug, Clone, Hash)]
struct Squeeze13;

impl Expansion for Squeeze13 {
    fn name(&self) -> Cow<str> {
        "Squeeze13".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        if inputs.len() == 2 {
            s.given_2(&inputs[0].shape, &inputs[1].value, move |s, shape, axes| {
                let axes =
                    axes.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|i| *i as isize).collect();
                let op = tract_hir::ops::array::Squeeze::new(Some(axes));
                let out_shape = op.output_shape(&shape)?;
                s.equals(&outputs[0].shape, out_shape)
            })
        } else {
            s.given(&inputs[0].shape, move |s, shape| {
                let axes = shape
                    .iter()
                    .enumerate()
                    .filter(|(_, dim)| dim.is_one())
                    .map(|(pos, _)| pos as isize)
                    .collect();
                let op = tract_hir::ops::array::Squeeze::new(Some(axes));
                let out_shape = op.output_shape(&shape)?;
                s.equals(&outputs[0].shape, out_shape)
            })
        }
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if inputs.len() == 2 {
            if let Some(axes) = model.outlet_fact(inputs[1])?.konst.as_ref() {
                let axes =
                    axes.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|i| *i as isize).collect();
                let op = tract_hir::ops::array::Squeeze::new(Some(axes));
                op.wire(prefix, model, &inputs[0..1])
            } else {
                bail!("Need axes to be a constant")
            }
        } else {
            let axes = model
                .outlet_fact(inputs[0])?
                .shape
                .iter()
                .enumerate()
                .filter(|(_, dim)| dim.is_one())
                .map(|(pos, _)| pos as isize)
                .collect();
            let op = tract_hir::ops::array::Squeeze::new(Some(axes));
            op.wire(prefix, model, &inputs[0..1])
        }
    }
}
