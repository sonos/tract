use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::Downsample;

#[derive(Debug, Clone, Default)]
struct DownsampleProblem {
    input_shape: Vec<usize>,
    op: Downsample,
}

impl Arbitrary for DownsampleProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<DownsampleProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(1..10usize, 1..5usize)
            .prop_flat_map(|input_shape| {
                let rank = input_shape.len();
                (Just(input_shape), 0..rank, 1..4isize, any::<bool>())
            })
            .prop_map(|(input_shape, axis, stride, dir)| DownsampleProblem {
                input_shape,
                op: Downsample { axis, stride: if dir { -stride } else { stride }, modulo: 0 },
            })
            .boxed()
    }
}

impl Test for DownsampleProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut input = Tensor::zero::<f32>(&self.input_shape)?;
        input.as_slice_mut::<f32>()?.iter_mut().enumerate().for_each(|(ix, x)| *x = ix as f32);

        let mut slices = vec![];
        let mut current =
            if self.op.stride > 0 { 0isize } else { input.shape()[self.op.axis] as isize - 1 };
        while current >= 0 && current < input.shape()[self.op.axis] as isize {
            slices.push(input.slice(self.op.axis, current as usize, current as usize + 1)?);
            current += self.op.stride;
        }
        let reference = Tensor::stack_tensors(self.op.axis, &slices)?;

        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", TypedFact::from(&input).without_value())?;
        let output = model.wire_node("downsample", self.op.clone(), &[wire])?;
        model.set_output_outlets(&output)?;
        let mut output = runtime.prepare(model)?.run(tvec![input.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<DownsampleProblem>("proptest", ());

    suite.add_test(
        "neg_0",
        DownsampleProblem {
            input_shape: vec![1],
            op: Downsample { axis: 0, stride: -1, modulo: 0 },
        },
        false,
    );
    Ok(suite)
}
