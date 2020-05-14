use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct ArgMaxMin {
    pub max: bool,
    pub axis: usize,
    pub keepdims: bool,
}

tract_linalg::impl_dyn_hash!(ArgMaxMin);

impl ArgMaxMin {
    fn eval_t<T: Datum + PartialOrd>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        use std::cmp::Ordering;
        let array = input.to_array_view::<T>()?;
        let f: fn(&(usize, &T), &(usize, &T)) -> Ordering = if self.max {
            |a, b| a.1.partial_cmp(&b.1).unwrap_or(a.0.cmp(&b.0))
        } else {
            |a, b| b.1.partial_cmp(&a.1).unwrap_or(a.0.cmp(&b.0))
        };
        let mut values = array
            .map_axis(Axis(self.axis), |row| row.iter().enumerate().max_by(f).unwrap().0 as i64);
        if self.keepdims {
            values = values.insert_axis(Axis(self.axis));
        }
        Ok(Tensor::from(values).into())
    }
}

impl Op for ArgMaxMin {
    fn name(&self) -> Cow<str> {
        "ArgMaxMin".into()
    }

    op_core_mir!();
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for ArgMaxMin {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl TypedOp for ArgMaxMin {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.to_tvec();
        if self.keepdims {
            shape[self.axis] = 1.into()
        } else {
            shape.remove(self.axis);
        }
        Ok(tvec!(TypedFact::dt_shape(i64::datum_type(), &*shape)?))
    }

    as_op!();
}
