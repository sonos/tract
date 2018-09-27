use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use dim::DimLike;

use super::patches::{PaddingSpec, Patch};

#[derive(Debug, Clone, new, Default)]
pub struct MaxPool {
    data_is_nhwc: bool, // default is nchw (onnx)
    kernel_shape: Vec<usize>,
    padding: PaddingSpec,
    strides: Option<Vec<usize>>,
}

impl MaxPool {
    fn patch<D: DimLike>(&self, input_full_shape: &[D]) -> Patch<D> {
        let spatial_rank = input_full_shape.len() - 2;
        let strides:Vec<usize> = self.strides.clone().unwrap_or(vec![1; spatial_rank]);
        let dilations = vec![1; spatial_rank];
        let kernel_shape:Vec<D> = self.kernel_shape.iter().map(|i| D::from(*i)).collect();
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

impl Op for MaxPool {
    fn name(&self) -> &str {
        "MaxPool"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let input:ArrayViewD<f32> = input.to_array_view()?;

        let mut patch = self.patch(input.shape());
        let channels = input.shape()[patch.axis_data_channel()];
        let shape: Vec<usize> = patch.output_full_shape(channels);

        let output = ArrayD::from_shape_fn(shape, |coords| -> f32 {
            patch.patch_data_iter(&input, coords.slice())
                .filter_map(|pair| pair)
                .fold(::std::f32::MIN, |acc, v| acc.max(v))
        });
        Ok(tvec!(output.into()))
    }
}

impl InferenceRulesOp for MaxPool {
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
