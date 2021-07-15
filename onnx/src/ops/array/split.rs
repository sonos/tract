use tract_hir::internal::*;
use tract_hir::ops::array;

use crate::model::ParsingContext;
use crate::pb::*;

pub fn split(
    ctx: &ParsingContext,
    node: &NodeProto,
    ) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    if ctx.onnx_operator_set_version < 13 || node.input.len() == 1 {
        let split = node.get_attr_opt_vec("split")?;
        Ok((expand(array::Split::new(axis, node.output.len(), split)), vec![]))
    } else {
        Ok((expand(Split13 { axis, outputs: node.output.len() }), vec![]))
    }
}

#[derive(Debug, Clone, Hash)]
struct Split13 {
    axis: isize,
    outputs: usize,
}

impl_dyn_hash!(Split13);

impl Expansion for Split13 {
    fn name(&self) -> Cow<str> {
        "Split13".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
        ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        s.given_2(&inputs[0].shape, &inputs[1].value, move |s, shape, splits| {
            let splits = splits.cast_to::<TDim>()?;
            let splits = splits.as_slice::<TDim>()?;
            let axis = self.axis + if self.axis < 0 { shape.len() as isize } else { 0 };
            for (o, dim) in outputs.iter().zip(splits.iter()) {
                s.equals(&o.shape[axis as usize], dim)?;
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
        if let Some(splits) = model.outlet_fact(inputs[1])?.konst.as_ref() {
            let axis = self.axis + if self.axis < 0 { model.outlet_fact(inputs[0])?.rank() as isize } else { 0 };
            let splits = splits.cast_to::<i64>()?;
            let splits = splits.as_slice::<i64>()?.into_iter().map(|i| *i as usize).collect::<Vec<_>>();
            let op = tract_hir::ops::array::Split::new(axis, splits.len(), Some(splits));
            return op.wire(prefix, model, &inputs[0..1]);
        }
        bail!("Need splits to be a constant and explicit (constant integers)")
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.outputs)
    }

    op_onnx!();
}
