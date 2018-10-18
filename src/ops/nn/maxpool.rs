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
    with_index_outputs: bool,
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

    fn noutputs(&self) -> usize {
        if self.with_index_outputs {
            2
        } else {
            1
        }
    }
}

impl StatelessOp for MaxPool {
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<f32> = input.to_array_view()?;

        let patch = self.patch(input.shape());
        let shape: Vec<usize> = patch.output_full_shape(patch.input_shape.c_dim());
        let visitor = patch.wrap(&input);

        let mut values = unsafe { ArrayD::uninitialized(shape.clone()) };
        let mut indices = if self.with_index_outputs {
            Some(unsafe { ArrayD::uninitialized(shape.clone()) })
        } else {
            None
        };
        ::ndarray::indices(shape).into_iter().for_each(|coords| {
            let max = visitor
                .at(&coords.slice())
                .enumerate()
                .filter_map(|(ix, v)| v.map(|v| (ix, v)))
                .fold(
                    (0, ::std::f32::MIN),
                    |acc, v| if acc.1 < v.1 { v } else { acc },
                );
            values[&coords] = max.1;
            if self.with_index_outputs {
                indices.as_mut().unwrap()[coords] =
                    visitor.global_offset_for(&coords.slice(), max.0) as i64;
            }
        });
        if self.with_index_outputs {
            Ok(tvec!(values.into(), indices.unwrap().into()))
        } else {
            Ok(tvec!(values.into()))
        }
    }
}

impl InferenceRulesOp for MaxPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, self.noutputs() as i64)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        if self.with_index_outputs {
            s.equals(&outputs[1].datum_type, i64::datum_type())?;
            s.equals(&outputs[1].rank, &inputs[0].rank)?;
        }
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_fmt.shape(ishape);
            let ones = vec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for o in 0..self.noutputs() {
                for (ix, &d) in computed.output.iter().enumerate() {
                    s.equals(&outputs[o].shape[ix + ishape.h_axis()], d)?;
                }
                s.equals(&outputs[o].shape[ishape.n_axis()], ishape.n_dim())?;
                s.equals(&outputs[o].shape[ishape.c_axis()], ishape.c_dim())?;
            }
            Ok(())
        })
    }
}
