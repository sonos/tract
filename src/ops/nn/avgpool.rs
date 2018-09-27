use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use dim::DimLike;

use super::patches::{PaddingSpec, Patch};

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
        let strides:Vec<usize> = self.strides.clone().unwrap_or(vec![1; spatial_rank]);
        let dilations = vec![1; spatial_rank];
        let kernel_shape:Vec<D> = self.kernel_shape.iter().map(|i| D::from(*i)).collect();
        let first_spatial_data_axis = 1 + (!self.data_is_nhwc as usize);
        let input_spatial_shape = &input_full_shape[first_spatial_data_axis..][..spatial_rank];
        assert_eq!(spatial_rank, kernel_shape.len());
        assert_eq!(spatial_rank, input_spatial_shape.len());
        assert_eq!(spatial_rank, strides.len());
        let geo = self
            .padding
            .compute(input_spatial_shape, &kernel_shape, &*dilations, &*strides);
        Patch::new(
            self.data_is_nhwc,
            dilations,
            kernel_shape,
            geo.pad_before.clone(),
            geo.pad_after.clone(),
            strides,
            input_full_shape.to_vec(),
            geo.output_spatial_shape
        )
    }

    fn output_shape<D: DimLike>(&self, ishape: &[D]) -> Vec<D> {
        let patch = self.patch(ishape);
        patch.output_full_shape(ishape[patch.axis_data_channel()])
    }
}

impl Op for AvgPool {
    fn name(&self) -> &str {
        "AvgPool"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let input = input.to_array_view()?;

        let patch = self.patch(input.shape());

        let shape: Vec<usize> = self.output_shape(input.shape());

        let data_field = patch.mk_data_field();

        let output = ArrayD::from_shape_fn(shape, |coords| -> f32 {
            let output = patch.split_data_coords(coords.slice());
            let pair = data_field.outer_iter().map(|imgf| {
                let i_coords: Vec<usize> = izip!(output.space.iter(), imgf.iter(), patch.strides.iter())
                    .map(|(x, i, s)| (x * s).wrapping_add(*i))
                    .collect();
                input
                    .subview(Axis(patch.axis_data_channel()), output.chan)
                    .subview(Axis(patch.axis_data_batch()), output.n) // careful, need to start with higher ranking
                    .get(&*i_coords)
                    .map(|&v| (v, true))
                    .unwrap_or((0.0, false))
            })
            .filter(|pair| pair.1 || self.count_include_pad)
            .fold((0.0,0), |acc, pair| (acc.0 + pair.0, acc.1 + 1));
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
                solver.equals(&outputs[0].shape, self.output_shape(&*ishape));
            });
    }
}
