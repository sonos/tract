use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use super::{DataFormat, PaddingSpec, Patch};

#[derive(Debug, Clone, new, Default)]
pub struct AvgPool {
    data_fmt: DataFormat,
    kernel_shape: Vec<usize>,
    padding: PaddingSpec,
    strides: Option<Vec<usize>>,
    count_include_pad: bool,
}

impl AvgPool {
    fn patch(&self, input_full_shape: &[usize]) -> Patch {
        let hw_rank = self.data_fmt.shape(input_full_shape).hw_rank();
        Patch::new(
            self.data_fmt,
            vec![1; hw_rank],
            self.kernel_shape.clone(),
            &self.padding,
            self.strides.clone().unwrap_or_else(|| vec![1; hw_rank]),
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


        let patch = self.patch(input.shape());
        let shape: Vec<usize> = patch.output_full_shape(patch.input_shape.c_dim());
        let visitor = patch.wrap(&input);

        let output = ArrayD::from_shape_fn(shape, |coords| -> f32 {
            let pair = visitor.at(&coords.slice())
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
                let ishape = self.data_fmt.shape(ishape);
                let ones = vec!(1; ishape.hw_rank());
                let (_, _, out_geo_shape) = self.padding.compute(ishape.hw_dims(), &*self.kernel_shape, &ones, self.strides.as_ref().unwrap_or(&ones));
                for (ix, &s) in out_geo_shape.iter().enumerate() {
                    solver.equals(&outputs[0].shape[ix+ishape.h_axis()], s);
                }
                solver.equals(&outputs[0].shape[ishape.n_axis()], ishape.n_dim());
                solver.equals(&outputs[0].shape[ishape.c_axis()], ishape.c_dim());
            });
    }
}
