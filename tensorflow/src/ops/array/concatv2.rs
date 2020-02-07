use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_core::internal::*;
use tract_core::infer::*;

pub fn build(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(Box::new(ConcatV2))
}

#[derive(Debug, Clone, new)]
pub struct ConcatV2;

impl StatelessOp for ConcatV2 {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let axis: i32 = *inputs.pop().unwrap().to_scalar::<i32>()?;
        tract_core::ops::array::Concat::new(axis as _).eval(inputs)
    }
}

impl Op for ConcatV2 {
    fn name(&self) -> Cow<str> {
        "tf.ConcatV2".into()
    }

    not_a_typed_op!();
}

impl InferenceRulesOp for ConcatV2 {
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

    inference_op_as_op!();

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        if let Some(ref axis) = model.outlet_fact(node.inputs[node.inputs.len() - 1])?.value.concretize() {
            let axis = axis.to_scalar::<i32>()?;
            Ok(Some(InferenceModelPatch::replace_single_op(
                model,
                node,
                &node.inputs[..node.inputs.len() - 1],
                tract_core::ops::array::Concat::new(*axis as _),
            )?))
        } else {
            Ok(None)
        }
    }
}
