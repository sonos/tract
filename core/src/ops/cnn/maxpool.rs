use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::pools::PoolSpec;

#[derive(Debug, Clone, new, Default)]
pub struct MaxPool {
    pool_spec: PoolSpec,
    with_index_outputs: Option<DatumType>,
}

impl Op for MaxPool {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }
}

impl StatelessOp for MaxPool {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<f32> = input.to_array_view()?;
        let input_ptr = input.as_ptr();
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(&*input.shape());

        let mut values = unsafe { ArrayD::<f32>::uninitialized(&*output_shape.shape) };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::<i32>::uninitialized(&*output_shape.shape) })
        } else {
            None
        };
        for n in 0..input_shape.n() {
            let input_offset = input_shape.n_stride() * n;
            let output_offset = output_shape.n_stride() * n;
            for c in 0..input_shape.c() {
                let input_offset = input_offset + input_shape.c_stride() * c;
                let output_offset = output_offset + output_shape.c_stride() * c;
                unsafe {
                    patch.visit_output(|visitor| {
                        let max = visitor
                            .valid_offsets()
                            .map(|v| (v, *input_ptr.offset(v + input_offset as isize)))
                            .fold((0, ::std::f32::MIN), |acc, v| if acc.1 < v.1 { v } else { acc });
                        *values
                            .as_mut_ptr()
                            .offset(output_offset as isize + visitor.output_offset) = max.1;
                        if let Some(ref mut indices) = indices {
                            *indices
                                .as_mut_ptr()
                                .offset(output_offset as isize + visitor.output_offset) =
                                max.0 as i32 / patch.spec.output_inner_stride as i32;
                        }
                    })
                }
            }
        }
        if let Some(dt) = self.with_index_outputs {
            Ok(tvec!(
                values.into(),
                Tensor::from(indices.unwrap()).cast_to_dt(dt)?.into_owned().into_tensor()
            ))
        } else {
            Ok(tvec!(values.into()))
        }
    }
}

impl InferenceRulesOp for MaxPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1 + self.with_index_outputs.is_some() as usize)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        if let Some(idt) = self.with_index_outputs {
            s.equals(&outputs[1].datum_type, idt)?;
            s.equals(&outputs[1].shape, &outputs[0].shape)?;
        }
        self.pool_spec.rules_for_shape(s, inputs, outputs)
    }
}
