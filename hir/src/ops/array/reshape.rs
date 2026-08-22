use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct Reshape {
    /// ONNX's `allowzero`, added in opset 14. When set, a 0 in the target shape is a literal
    /// zero-length dimension instead of a copy of the input dimension at that position. The
    /// TensorFlow frontend always passes false, which is its only convention.
    pub allowzero: bool,
}

impl Expansion for Reshape {
    fn name(&self) -> StaticName {
        "Reshape".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        let allowzero = self.allowzero;
        s.given_2(&inputs[0].shape, &inputs[1].value, move |s, ishape, shape| {
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.try_as_plain()?.as_slice::<TDim>()?;
            let oshape = tract_core::ops::change_axes::compute_shape_with_onnx_rules(
                &ishape, shape, allowzero,
            )
            .with_context(|| format!("Reshaping {ishape:?} to {shape:?}"))?;
            s.equals(&outputs[0].shape, ShapeFactoid::from(oshape))
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref shape) = model.outlet_fact(inputs[1])?.konst {
            let input_shape: TVec<TDim> = model.outlet_fact(inputs[0])?.shape.to_tvec();
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.try_as_plain()?.as_slice::<TDim>()?;
            let mut wire = tvec!(inputs[0]);
            for (ix, op) in to_axis_ops_with_onnx_rules(&input_shape, shape, self.allowzero)?
                .into_iter()
                .enumerate()
            {
                wire = model.wire_node(format!("{prefix}.{ix}"), op, &wire)?;
            }
            return Ok(wire);
        }
        bail!("shape input is variable")
    }
}
