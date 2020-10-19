use crate::infer::*;
use crate::internal::*;

use tract_core::ops::array::MultiBroadcastTo as Typed;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct MultiBroadcastTo;
tract_data::impl_dyn_hash!(MultiBroadcastTo);

impl Expansion for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.given(&inputs[0].shape, move |s, shape| {
            s.given(&inputs[1].value, move |s, dims| {
                let dims = dims.cast_to::<TDim>()?;
                let dims =
                    tract_core::broadcast::multi_broadcast(&[&*dims.as_slice::<TDim>()?, &*shape])
                        .ok_or("incompatible shapes")
                        .unwrap();
                s.equals(&outputs[0].shape, ShapeFactoid::from(dims))
            })
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(shape) = model.outlet_fact(inputs[1])?.konst.clone() {
            let input_shape = model.outlet_fact(inputs[0])?.shape.to_tvec();
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            let dims = tract_core::broadcast::multi_broadcast(&[&*input_shape, &*shape])
                .context("incompatible shapes")?;
            let op = Typed::new(dims.into());
            model.wire_node(prefix, op, &[inputs[0]])
        } else {
            bail!("shape input is variable")
        }
    }
}
