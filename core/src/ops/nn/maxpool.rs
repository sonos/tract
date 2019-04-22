use crate::internal::*;
use ndarray::prelude::*;

use super::{DataFormat, PaddingSpec, Patch, PatchSpec};

#[derive(Debug, Clone, new, Default)]
pub struct MaxPool {
    data_format: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
    with_index_outputs: Option<DatumType>,
}

impl MaxPool {
    fn patch(&self, input_full_shape: &[usize]) -> Patch {
        let mut spec = PatchSpec::for_full_shape(self.data_format.clone(), input_full_shape)
            .with_kernel_shape(self.kernel_shape.clone())
            .with_padding(self.padding.clone());
        if let Some(strides) = self.strides.clone() {
            spec = spec.with_strides(strides);
        }
        spec.into_patch()
    }
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
        let shape = self.data_format.shape(input.shape());

        let patch = self.patch(input.shape());
        let output_shape = shape.fmt.from_n_c_hw(shape.n(), shape.c(), &*patch.output_shape);

        let mut values = unsafe { ArrayD::<f32>::uninitialized(&*output_shape.shape) };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::uninitialized(&*output_shape.shape) })
        } else {
            None
        };
        ::ndarray::indices(&*output_shape.shape).into_iter().for_each(|coords| unsafe {
            let input_ptr = input_ptr.offset((shape.n_stride() * coords[shape.n_axis()]) as isize);
            let input_ptr = input_ptr.offset((shape.c_stride() * coords[shape.c_axis()]) as isize);
            let max = patch
                .at(&coords.slice()[shape.hw_axes()])
                .enumerate()
                .filter_map(|(ix, v)| v.map(|v| (ix, *input_ptr.offset(v))))
                .fold((0, ::std::f32::MIN), |acc, v| if acc.1 < v.1 { v } else { acc });
            values[&coords] = max.1;
            if self.with_index_outputs.is_some() {
                indices.as_mut().unwrap()[coords] =
                    patch.global_offset_for(&coords.slice()[shape.hw_axes()], max.0) as i32;
            }
        });
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
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        if let Some(idt) = self.with_index_outputs {
            s.equals(&outputs[1].datum_type, idt)?;
            s.equals(&outputs[1].rank, &inputs[0].rank)?;
        }
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_format.shape(ishape);
            let ones = tvec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for o in 0..outputs.len() {
                for (ix, d) in computed.iter().enumerate() {
                    s.equals(&outputs[o].shape[ix + ishape.h_axis()], d.output)?;
                }
                s.equals(&outputs[o].shape[ishape.n_axis()], ishape.n_dim())?;
                s.equals(&outputs[o].shape[ishape.c_axis()], ishape.c_dim())?;
            }
            Ok(())
        })
    }
}
