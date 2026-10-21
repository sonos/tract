use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::cnn::*;
use tract_ndarray::*;

use crate::max_pool::MaxPoolProblem;
use crate::max_pool::MaxPoolProblemParams;

#[derive(Debug, Clone, Default)]
pub struct SumPoolProblemParams {}

/// A sum or average pool over the input and geometry of a [`MaxPoolProblem`].
#[derive(Debug, Clone)]
pub struct SumPoolProblem {
    pub pool: MaxPoolProblem,
    pub count_include_pad: bool,
    pub normalize: bool,
}

impl Arbitrary for SumPoolProblem {
    type Strategy = BoxedStrategy<SumPoolProblem>;
    type Parameters = SumPoolProblemParams;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (any_with::<MaxPoolProblem>(MaxPoolProblemParams::default()), any::<bool>(), any::<bool>())
            .prop_map(|(pool, count_include_pad, normalize)| SumPoolProblem {
                pool,
                count_include_pad,
                normalize,
            })
            .boxed()
    }
}

impl SumPoolProblem {
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(self.pool.input.shape()))?;
        let op = SumPool::new(self.pool.pool_spec()?, self.count_include_pad, self.normalize);
        let output = model.wire_node("sum_pool", op, &[src])?;
        model.select_output_outlets(&output)?;
        Ok(model)
    }

    fn reference(&self) -> TractResult<ArrayD<f32>> {
        let p = &self.pool;
        let input_shape = p.data_format.shape(p.input.shape())?;
        let n = input_shape.n().copied().unwrap_or(1);
        let c = *input_shape.c();
        let geometry = p.geometry()?;
        let output_hw: TVec<usize> = geometry.iter().map(|g| g.0).collect();
        let output_shape = p.data_format.from_n_c_hw(n, c, &*output_hw)?;
        let mut output = ArrayD::<f32>::zeros(&*output_shape.shape);
        let window = p.kernel_shape.iter().product::<usize>();
        for n in 0..n {
            for c in 0..c {
                for out in indices(&*output_hw) {
                    let mut sum = 0f32;
                    let mut valid = 0;
                    for tap in indices(&*p.kernel_shape) {
                        let pos: Option<TVec<usize>> = tract_itertools::izip!(
                            out.slice(),
                            tap.slice(),
                            &geometry,
                            &p.strides,
                            &p.dilations,
                            input_shape.hw_dims()
                        )
                        .map(|(o, t, (_, before), s, d, i)| {
                            (o * s + t * d).checked_sub(*before).filter(|p| p < i)
                        })
                        .collect();
                        if let Some(pos) = pos {
                            sum += p.input[&*p.data_format.from_n_c_hw(n, c, &*pos)?.shape];
                            valid += 1;
                        }
                    }
                    if self.normalize {
                        sum /= if self.count_include_pad { window } else { valid } as f32;
                    }
                    output[&*p.data_format.from_n_c_hw(n, c, out.slice())?.shape] = sum;
                }
            }
        }
        Ok(output)
    }
}

impl Test for SumPoolProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().context("Running reference")?.into_tensor();
        let mut model = self.tract().context("Generating model")?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output =
            runtime.prepare(model)?.run(tvec![self.pool.input.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<SumPoolProblem>("proptest", SumPoolProblemParams::default());
    Ok(suite)
}
