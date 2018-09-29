use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use dim::DimLike;

use super::{PaddingSpec, Patch};

#[derive(Debug, Clone, new, Default)]
pub struct AvgPool {
    data_is_nhwc: bool, // default is nchw (onnx)
    kernel_shape: Vec<usize>,
    padding: PaddingSpec,
    strides: Option<Vec<usize>>,
    count_include_pad: bool,
}

impl AvgPool {
    fn patch<D: DimLike>(&self, input_full_shape: &[D]) -> Patch<D> {
        let spatial_rank = input_full_shape.len() - 2;
        let strides: Vec<usize> = self.strides.clone().unwrap_or(vec![1; spatial_rank]);
        let dilations = vec![1; spatial_rank];
        let kernel_shape: Vec<D> = self.kernel_shape.iter().map(|i| D::from(*i)).collect();
        assert_eq!(spatial_rank, kernel_shape.len());
        assert_eq!(spatial_rank, strides.len());
        Patch::new(
            self.data_is_nhwc,
            dilations,
            kernel_shape,
            &self.padding,
            strides,
            input_full_shape.to_vec(),
        )
    }
}

impl Op for AvgPool {
    fn name(&self) -> &str {
        "AvgPool"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<f32> = input.to_array_view()?;

        let mut patch = self.patch(input.shape());
        patch.cache_data_field();
        let channels = input.shape()[patch.axis_data_channel()];
        let shape: Vec<usize> = patch.output_full_shape(channels);

        let output = ArrayD::from_shape_fn(shape, |coords| -> f32 {
            let pair = patch
                .patch_data_iter(&input, coords.slice())
                .map(|ov| ov.map(|v| (v, true)).unwrap_or((0.0, false)))
                .filter(|pair| pair.1 || self.count_include_pad)
                .fold((0.0, 0), |acc, pair| (acc.0 + pair.0, acc.1 + 1));
            pair.0 / pair.1 as f32
        });
        Ok(tvec!(output.into()))
    }
}

impl InferenceRulesOp for AvgPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&outputs.len, 1)
            .equals(&outputs[0].datum_type, &inputs[0].datum_type)
            .given(&inputs[0].shape, move |solver, ishape| {
                let patch = self.patch(&*ishape);
                let channels = ishape[patch.axis_data_channel()];
                let shape = patch.output_full_shape(channels);
                solver.equals(&outputs[0].shape, shape.bex());
            });
    }
}
