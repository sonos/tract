use ndarray::prelude::*;
use tract_core::ops::prelude::*;
use num_traits::AsPrimitive;
use std::ops::{ Add, Div, Mul, Sub };

#[derive(Debug, Clone, new)]
pub struct Range {
    dtype: DatumType,
}

pub fn range(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let dtype = pb.get_attr_datum_type("Tidx")?;
    Ok(Box::new(Range::new(dtype)))
}

impl Range {
    fn eval_t<T>(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> 
        where T: Datum + AsPrimitive<usize> + Add<T, Output=T> + Div<T, Output=T> + Mul<T,Output=T> + Sub<T, Output=T>,
              usize: AsPrimitive<T>
    {
        let (start, limit, delta) = args_3!(inputs);
        let start = *start.to_scalar::<T>()?;
        let limit = *limit.to_scalar::<T>()?;
        let delta = *delta.to_scalar::<T>()?;
        let value = Array1::from_shape_fn(((limit-start) / delta).as_(), |ix| ix.as_() * delta + start);
        Ok(tvec![value.into()])
    }
}

impl Op for Range {
    fn name(&self) -> Cow<str> {
        "tf.Range".into()
    }
}

impl StatelessOp for Range {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        dispatch_numbers!(Self::eval_t(self.dtype)(self, inputs))
    }
}

impl InferenceRulesOp for Range {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, self.dtype)?;
        s.equals(&inputs[1].datum_type, self.dtype)?;
        s.equals(&inputs[2].datum_type, self.dtype)?;
        s.equals(&outputs[0].datum_type, self.dtype)?;
        s.equals(&inputs[0].rank, 0)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(&inputs[2].rank, 0)?;
        s.equals(&outputs[0].rank, 1)?;
        Ok(())
    }
}
