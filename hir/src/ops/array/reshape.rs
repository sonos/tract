use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Reshape {}

impl Expansion for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given_2(&inputs[0].shape, &inputs[1].value, move |s, ishape, shape| {
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            let oshape = tract_core::ops::change_axes::compute_shape_with_tf_rules(&ishape, shape)
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
            let shape = shape.as_slice::<TDim>()?;
            let mut wire = tvec!(inputs[0]);
            for (ix, op) in to_axis_ops_with_tf_rules(&input_shape, shape)?.into_iter().enumerate() {
                wire = model.wire_node(format!("{prefix}.{ix}"), op, &wire)?;
            }
            return Ok(wire);
        }
        bail!("shape input is variable")
    }
}
