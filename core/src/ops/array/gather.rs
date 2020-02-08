use crate::internal::*;
use crate::infer::*;
use ndarray::*;

#[derive(Debug, Clone, new)]
pub struct Gather {
    axis: i64,
}

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl Gather {
    fn resolved_axis(&self, rank: usize) -> TractResult<usize> {
        if 0 <= self.axis && self.axis <= rank as i64 - 1 {
            Ok(self.axis as usize)
        } else if -(rank as i64) <= self.axis && self.axis < 0 {
            Ok((self.axis + rank as i64) as usize)
        } else {
            bail!("Illegal combination of values for rank and axis")
        }
    }

    pub fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        indices_shape: &[D],
    ) -> TractResult<TVec<D>> {
        let axis = self.resolved_axis(input_shape.len())?;
        let mut output_shape = tvec![];
        for (idx, dim) in input_shape.iter().enumerate() {
            if idx != axis {
                output_shape.push(dim.clone());
            } else {
                for idx2 in indices_shape {
                    output_shape.push(idx2.clone());
                }
            }
        }
        Ok(output_shape)
    }

    fn eval_t<T: Datum>(
        &self,
        data: Arc<Tensor>,
        indices: &Arc<Tensor>,
    ) -> TractResult<Arc<Tensor>> {
        let data_view = data.to_array_view::<T>()?;
        let axis = self.resolved_axis(data.shape().len())?;
        let indices = indices.cast_to::<i64>()?;
        if indices.shape().len() == 0 {
            let mut index = *indices.to_scalar::<i64>()?;
            if index < 0 {
                index += data_view.shape()[0] as i64;
            }
            return Ok(data_view
                .index_axis(Axis(axis), index as usize)
                .to_owned()
                .into_arc_tensor());
        }

        let mut output: Array<T, _> = unsafe {
            T::uninitialized_array(&*self.compute_output_shape(data.shape(), indices.shape())?)
        };
        for (pattern, index) in indices.to_array_view::<i64>()?.indexed_iter() {
            {
                let mut to_update = output.index_axis_mut(Axis(axis), pattern[0]);
                for idx in 1..pattern.ndim() {
                    to_update = to_update.index_axis_move(Axis(0), pattern[idx]);
                }

                to_update.assign(&data_view.index_axis(Axis(axis), *index as usize));
            }
        }
        Ok(output.into_arc_tensor())
    }
}

impl TypedOp for Gather {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            &*self
                .compute_output_shape(&*inputs[0].shape.to_tvec(), &*inputs[1].shape.to_tvec())?
        )?))
    }
}

impl StatelessOp for Gather {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, indices) = args_2!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(data.datum_type())(&self, data, &indices))?))
    }
}

impl InferenceRulesOp for Gather {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, i64::datum_type())?;
        s.equals(inputs[0].rank.bex() - 1 + inputs[1].rank.bex(), outputs[0].rank.bex())?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, input_shape, indices_shape| {
            let output_shape = self.compute_output_shape(&*input_shape, &*indices_shape)?;
            s.equals(&outputs[0].shape, output_shape)?;
            Ok(())
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_gather_scalar_index() {
        let data = Tensor::from(arr1(&[1i64, 2, 3]));
        let gatherer = Gather::new(0);
        for idx in 2..3 {
            let index = Tensor::from(arr0(idx as i64));
            let outputs = gatherer.eval(tvec![data.clone().into(), index.into()]).unwrap();
            let output = &outputs[0];
            assert_eq!(output.shape().len(), 0);
            assert_eq!(*output.to_scalar::<i64>().unwrap(), idx + 1);
        }
    }
}
