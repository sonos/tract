use crate::infer::*;
use crate::internal::*;

use tract_core::ops::array::MultiBroadcastTo as Typed;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct MultiBroadcastTo;

impl MultiBroadcastTo {
    fn wire_with_known_target_shape(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        target_shape: &[TDim],
    ) -> TractResult<TVec<OutletId>> {
        let left_shape = model.outlet_fact(inputs[0])?.shape.to_tvec();
        let dims = tract_core::broadcast::multi_broadcast(&[&*left_shape, target_shape])?;
        let op = Typed::new(dims.into());
        model.wire_node(prefix, op, &[inputs[0]])
    }
}

impl Expansion for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.given(&inputs[0].shape, move |s, shape| {
            s.given(&inputs[1].value, move |s, dims| {
                let dims = dims.cast_to::<TDim>()?;
                let dims =
                    tract_core::broadcast::multi_broadcast(&[dims.as_slice::<TDim>()?, &shape])?;
                s.equals(&outputs[0].shape, ShapeFactoid::from(dims))
            })
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(shape) = model.outlet_fact(inputs[1])?.konst.clone() {
            let shape = shape.cast_to::<TDim>()?;
            self.wire_with_known_target_shape(prefix, model, inputs, shape.as_slice()?)
        } else {
            bail!("shape input is variable")
        }
    }

    fn wire_with_inference_model_and_node(
        &self,
        prefix: &str,
        source: &InferenceModel,
        node: &InferenceNode,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(shape) = model.outlet_fact(inputs[1])?.konst.clone() {
            let shape = shape.cast_to::<TDim>()?;
            self.wire_with_known_target_shape(prefix, model, inputs, shape.as_slice()?)
        } else if let Some(shape) = source.outlet_fact(node.id.into())?.shape.concretize() {
            let op = Typed::new(shape.into());
            model.wire_node(prefix, op, &[inputs[0]])
        } else {
            bail!("shape input is variable, of variable length (output can not have variable rank)")
        }
    }
}
