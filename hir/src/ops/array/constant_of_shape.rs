use tract_ndarray::*;

use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct ConstantOfShape {
    scalar: Arc<Tensor>,
}

tract_linalg::impl_dyn_hash!(ConstantOfShape);

impl Expansion for ConstantOfShape {
    fn name(&self) -> Cow<str> {
        "ConstantOfShape".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.scalar.datum_type())?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[0].shape[0], outputs[0].rank.bex().to_dim())?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref fact) = target.outlet_fact(inputs[0])?.konst {
            let shape = fact.cast_to::<i32>()?;
            let shape = shape.as_slice::<i32>()?.iter().map(|&s| s as usize).collect::<TVec<_>>();
            let value =
                dispatch_copy!(make_from_shape(self.scalar.datum_type())(&*shape, &self.scalar))?;
            return target.wire_node(&*prefix, crate::ops::konst::Const::new(value), &[]);
        }
        bail!("shape input is variable")
    }
}

fn make_from_shape<T>(shape: &[usize], scalar: &Tensor) -> TractResult<Arc<Tensor>>
where
    T: Datum + Copy,
{
    Ok(Array::<T, _>::from_elem(&*shape, *scalar.to_scalar()?).into_arc_tensor())
}
