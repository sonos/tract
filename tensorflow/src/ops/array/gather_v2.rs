use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_core::infer::*;
use tract_core::internal::*;

pub fn gather_v2(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(Box::new(GatherV2::new()))
}

#[derive(Debug, Clone, new)]
pub struct GatherV2 {}

impl Op for GatherV2 {
    fn name(&self) -> Cow<str> {
        "tf.GatherV2".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for GatherV2 {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, indices, axis) = args_3!(inputs);
        let op = tract_core::ops::array::Gather::new(*axis.to_scalar::<i32>()? as i64);
        op.eval(tvec!(input, indices))
    }
}

impl InferenceRulesOp for GatherV2 {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, i32::datum_type())?;
        s.equals(&inputs[2].datum_type, i32::datum_type())?;
        s.equals(&inputs[2].rank, 0)?;
        s.given_3(
            &inputs[0].shape,
            &inputs[1].shape,
            &inputs[2].value,
            move |s, input_shape, indices_shape, axis| {
                let op = tract_core::ops::array::Gather::new(*axis.to_scalar::<i32>()? as i64);
                let output_shape = op.compute_output_shape(&input_shape, &indices_shape)?;
                s.equals(&outputs[0].shape, output_shape)
            },
        )
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(axis) = target.outlet_fact(mapping[&node.inputs[2]])?.konst.as_ref() {
            let op = tract_core::ops::array::Gather::new(*axis.to_scalar::<i32>()? as i64);
            target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]], mapping[&node.inputs[1]]])
        } else {
            bail!("Need to know axis to type GatherV2")
        }
    }

    as_op!();
}
