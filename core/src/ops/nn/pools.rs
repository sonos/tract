use crate::internal::*;

pub mod avgpool;
pub mod maxpool;

use super::{DataFormat, DataShape, PaddingSpec, Patch, PatchSpec};

type Shape = DataShape<usize, TVec<usize>>;

#[derive(Debug, Clone, new, Default)]
pub struct PoolSpec {
    data_format: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
}

impl PoolSpec {
    fn compute_geo(&self, input_full_shape: &[usize]) -> (Shape, Patch, Shape) {
        let input_shape = self.data_format.shape(input_full_shape.into());
        let mut spec = PatchSpec::for_full_shape(self.data_format, input_full_shape)
            .with_output_inner_stride(input_shape.w_stride())
            .with_kernel_shape(self.kernel_shape.clone())
            .with_padding(self.padding.clone());
        if let Some(strides) = self.strides.clone() {
            spec = spec.with_strides(strides);
        }
        let patch = spec.into_patch();
        let output_shape =
            input_shape.fmt.from_n_c_hw(input_shape.n(), input_shape.c(), &*patch.output_shape);
        (input_shape, patch, output_shape)
    }

    fn rules_for_shape<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
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

