use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Gather {
    pub axis: usize,
}
tract_linalg::impl_dyn_hash!(Gather);

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl Gather {
    pub fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        indices_shape: &[D],
    ) -> TractResult<TVec<D>> {
        let mut output_shape = tvec![];
        for (idx, dim) in input_shape.iter().enumerate() {
            if idx != self.axis {
                output_shape.push(dim.clone());
            } else {
                for idx2 in indices_shape {
                    output_shape.push(idx2.clone());
                }
            }
        }
        Ok(output_shape)
    }

    unsafe fn eval_t<T: Datum>(
        &self,
        data: Arc<Tensor>,
        indices: &Arc<Tensor>,
    ) -> TractResult<Arc<Tensor>> {
        let data_view = data.to_array_view_unchecked::<T>();
        let indices = indices.cast_to::<i64>()?;
        if indices.shape().len() == 0 {
            let mut index = *indices.to_scalar::<i64>()?;
            if index < 0 {
                index += data_view.shape()[0] as i64;
            }
            let mut tensor =
                data_view.index_axis(Axis(self.axis), index as usize).to_owned().into_tensor();
            tensor.set_datum_type(data.datum_type());
            return Ok(tensor.into_arc_tensor());
        }

        let mut output = Tensor::uninitialized_dt(
            data.datum_type(),
            &*self.compute_output_shape(data.shape(), indices.shape())?,
        )?;
        let mut view = output.to_array_view_mut_unchecked::<T>();
        for (pattern, index) in indices.to_array_view::<i64>()?.indexed_iter() {
            let mut to_update = view.index_axis_mut(Axis(self.axis), pattern[0]);
            for idx in 1..pattern.ndim() {
                to_update = to_update.index_axis_move(Axis(0), pattern[idx]);
            }

            to_update.assign(&data_view.index_axis(Axis(self.axis), *index as usize));
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
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                &self, data, &indices
            ))?))
        }
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
            assert_eq!(output.shape().len(), 0);
            assert_eq!(*output.to_scalar::<i64>().unwrap(), idx + 1);
        }
    }
}
