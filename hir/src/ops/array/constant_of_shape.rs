use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct ConstantOfShape {
    scalar: Arc<Tensor>,
}



impl Expansion for ConstantOfShape {
    fn name(&self) -> Cow<str> {
        "ConstantOfShape".into()
    }


    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.scalar.datum_type())?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[0].shape[0], outputs[0].rank.bex().to_dim())?;
        s.given(&inputs[0].value, move |s, shape| {
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            for (axis, dim) in shape.iter().enumerate() {
                s.equals(&outputs[0].shape[axis], dim)?;
            }
            Ok(())
        })?;
        /* does not work .value assumes ints
        s.given(&outputs[0].rank, move |s, rank| {
            for axis in 0..rank as usize {
                s.equals(&outputs[0].shape[axis], &inputs[0].value[axis].bex())?;
            }
            Ok(())
        })?;
        */
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(shape) = target.outlet_fact(inputs[0])?.konst.clone() {
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            let scalar = target.add_const(format!("{prefix}.scalar"), self.scalar.clone())?;
            let op = tract_core::ops::array::MultiBroadcastTo::new(shape.into());
            return target.wire_node(prefix, op, &[scalar]);
        }
        bail!("shape input is variable")
    }
}
