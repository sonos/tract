use ndarray::prelude::*;
use ops::prelude::*;

use super::{DataFormat, PaddingSpec, Patch};

#[derive(Debug, Clone, new, Default)]
pub struct AvgPool {
    data_fmt: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
    count_include_pad: bool,
}

impl AvgPool {
    fn patch(&self, input_full_shape: &[usize]) -> Patch {
        let hw_rank = self.data_fmt.shape(input_full_shape).hw_rank();
        Patch::new(
            self.data_fmt,
            tvec![1; hw_rank],
            self.kernel_shape.clone(),
            &self.padding,
            self.strides.clone().unwrap_or_else(|| tvec![1; hw_rank]),
            input_full_shape.into(),
        )
    }
}

impl Op for AvgPool {
    fn name(&self) -> &str {
        "AvgPool"
    }
}

impl StatelessOp for AvgPool {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<f32> = input.to_array_view()?;

        let patch = self.patch(input.shape());
        let shape: TVec<usize> = patch.output_full_shape(patch.input_shape.c_dim());
        let visitor = patch.wrap(&input);

        let output = ArrayD::from_shape_fn(&*shape, |coords| -> f32 {
            let pair = visitor
                .at(&coords.slice())
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
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_fmt.shape(ishape);
            let ones = tvec![1; ishape.hw_rank()];
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
