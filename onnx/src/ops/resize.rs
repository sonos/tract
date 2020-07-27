use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use std::hash::Hash;
use tract_hir::internal::*;
use tract_hir::tract_core::itertools::Itertools;

pub fn resize(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let input_2_is_scales = node.input[2] != "";
    Ok((Box::new(Resize { input_2_is_scales }), vec![]))
}

#[derive(Clone, new, Debug, Hash)]
struct Resize {
    input_2_is_scales: bool,
}

tract_linalg::impl_dyn_hash!(Resize);

impl Op for Resize {
    fn name(&self) -> Cow<str> {
        "Resize".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl StatelessOp for Resize {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!();
    }
}

impl InferenceRulesOp for Resize {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(2 * inputs[0].rank.bex().to_dim(), inputs[1].shape[0].bex())?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(inputs[0].rank.bex().to_dim(), inputs[2].shape[0].bex())?;
        s.given(&inputs[0].rank, move |s, rank| {
            let rank = rank as usize;
            if self.input_2_is_scales {
                s.given_3(
                    &inputs[0].shape,
                    &inputs[1].value,
                    &inputs[2].value,
                    move |s, input_shape, rois, scales| {
                        let rois = rois.cast_to::<f32>()?;
                        let rois = rois.as_slice::<f32>()?;
                        let scales = scales.cast_to::<f32>()?;
                        let scales = scales.as_slice::<f32>()?;
                        for i in 0..rank {
                            let cropped = if rois.len() == 2 * rank {
                                rois[i + rank] - rois[i]
                            } else {
                                1.0f32
                            };
                            if let Ok(len) = input_shape[i].to_integer() {
                                let output_len =
                                    (len as f32 * cropped * scales[i]).round() as usize;
                                s.equals(&outputs[0].shape[i], output_len.to_dim())?;
                            }
                        }
                        Ok(())
                    },
                )?;
            } else {
                for i in 0..(rank as usize) {
                    s.equals(&outputs[0].shape[i], inputs[2].value[i].bex().to_dim())?;
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for Resize {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = inputs[0].shape.iter().enumerate().map|(ix, dim)| 
    }

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}
