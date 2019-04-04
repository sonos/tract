use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new)]
pub struct Gather {
    axis: i64,
}

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }
}

impl Gather {
    fn eval_t<T: Datum>(
        &self,
        data: SharedTensor,
        indices: &SharedTensor,
    ) -> TractResult<SharedTensor> {
        let data_view = data.to_array_view::<T>()?;
        let rank = data.shape().len() as i64;
        let axis = {
            let axis_res: TractResult<i64> = {
                if 0 <= self.axis && self.axis <= rank - 1 {
                    Ok(self.axis)
                } else if -rank <= self.axis && self.axis < 0 {
                    Ok(self.axis + rank)
                } else {
                    bail!("Illegal combination of values for rank and axis")
                }
            };
            axis_res? as usize
        };

        if indices.shape().len() == 0 {
            return Ok(data_view
                .index_axis(Axis(axis), *indices.to_scalar::<i64>()? as usize)
                .to_owned()
                .into());
        }

        let mut output_shape: Vec<usize> = vec![];
        for (idx, dim) in data_view.shape().to_vec().iter().enumerate() {
            if idx != axis {
                output_shape.push(*dim);
            } else {
                for idx2 in indices.shape() {
                    output_shape.push(*idx2);
                }
            }
        }
        let mut output: Array<T, _> = Array::default(output_shape);
        for (pattern, index) in indices.to_array_view::<i64>()?.indexed_iter() {
            {
                let mut to_update = output.index_axis_mut(Axis(axis), pattern[0]);
                for idx in 1..pattern.ndim() {
                    to_update = to_update.index_axis_move(Axis(0), pattern[idx]);
                }

                to_update.assign(&data_view.index_axis(Axis(axis), *index as usize));
            }
        }
        Ok(output.into())
    }
}

impl StatelessOp for Gather {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
        s.equals(&inputs[1].datum_type, i64::datum_type())?;
        s.equals(inputs[0].rank.bex() - 1 + inputs[1].rank.bex(), outputs[0].rank.bex())?;
        Ok(())
    }
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
            //            println!("{:?}", output);
            assert_eq!(output.shape().len(), 0);
            assert_eq!(*output.to_scalar::<i64>().unwrap(), idx + 1);
        }
    }
}
