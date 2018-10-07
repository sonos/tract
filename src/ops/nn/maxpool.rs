use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use super::{DataFormat, PaddingSpec, Patch};

#[derive(Debug, Clone, new, Default)]
pub struct MaxPool {
    data_fmt: DataFormat,
    kernel_shape: Vec<usize>,
    padding: PaddingSpec,
    strides: Option<Vec<usize>>,
}

impl MaxPool {
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

impl Op for MaxPool {
    fn name(&self) -> &str {
        "MaxPool"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<f32> = input.to_array_view()?;

        let patch = self.patch(input.shape());
        let shape: Vec<usize> = patch.output_full_shape(patch.input_shape.c_dim());
        let visitor = patch.wrap(&input);

        let output = ArrayD::from_shape_fn(shape, |coords| -> f32 {
            visitor.at(&coords.slice())
                .filter_map(|pair| pair)
                .fold(::std::f32::MIN, |acc, v| acc.max(v))
        });
        Ok(tvec!(output.into()))
    }
}

impl InferenceRulesOp for MaxPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_fmt.shape(ishape);
            let ones = vec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for (ix, &d) in computed.output.iter().enumerate() {
                s.equals(&outputs[0].shape[ix + ishape.h_axis()], d)?;
            }
            s.equals(&outputs[0].shape[ishape.n_axis()], ishape.n_dim())?;
            s.equals(&outputs[0].shape[ishape.c_axis()], ishape.c_dim())?;
            Ok(())
        })
    }
}
