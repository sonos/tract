use crate::internal::*;
use crate::infer::*;
use ndarray::*;

#[derive(Debug, Clone, new, Default)]
pub struct ArgMaxMin {
    max: bool,
    axis: usize,
    keepdims: bool,
}

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

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for ArgMaxMin {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl InferenceRulesOp for ArgMaxMin {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, DatumType::I64)?;
        if self.keepdims {
            s.equals(&outputs[0].rank, &inputs[0].rank)?;
            for i in 0..self.axis {
                s.equals(&outputs[0].shape[i], &inputs[0].shape[i])?;
            }
            s.equals(&outputs[0].shape[self.axis], 1.to_dim())?;
            s.given(&inputs[0].rank, move |s, rank| {
                for i in (self.axis + 1)..(rank as usize) {
                    s.equals(&outputs[0].shape[i], &inputs[0].shape[i])?;
                }
                Ok(())
            })?;
        } else {
            s.equals(&outputs[0].rank, inputs[0].rank.bex() - 1)?;
            for i in 0..self.axis {
                s.equals(&outputs[0].shape[i], &inputs[0].shape[i])?;
            }
            s.given(&inputs[0].rank, move |s, rank| {
                for i in (self.axis + 1)..(rank as usize - 1) {
                    s.equals(&outputs[0].shape[i], &inputs[0].shape[i + 1])?;
                }
                Ok(())
            })?;
        };
        Ok(())
    }

    as_op!();
    to_typed!();
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
