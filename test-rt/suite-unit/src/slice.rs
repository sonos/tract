use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::array::Slice;

#[derive(Debug, Clone, Default)]
struct SliceProblem {
    input_shape: TVec<usize>,
    op: Slice,
}

impl Arbitrary for SliceProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<SliceProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(1..10usize, 1..5usize)
            .prop_flat_map(|input_shape| {
                let rank = input_shape.len();
                (Just(input_shape), 0..rank)
            })
            .prop_flat_map(|(shape, axis)| {
                let b0 = 0..=shape[axis];
                let b1 = 0..=shape[axis];
                (Just(shape), Just(axis), b0, b1)
            })
            .prop_filter("non empty slice", |(_, _, b0, b1)| b0 != b1)
            .prop_map(|(input_shape, axis, b0, b1)| {
                let start = b0.min(b1).to_dim();
                let end = b0.max(b1).to_dim();
                SliceProblem { input_shape: input_shape.into(), op: Slice { axis, start, end } }
            })
            .boxed()
    }
}

impl Test for SliceProblem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut input = Tensor::zero::<f32>(self.input_shape.clone())?;
        input.as_slice_mut::<f32>()?.iter_mut().enumerate().for_each(|(ix, x)| *x = ix as f32);
        let reference = input.slice(
            self.op.axis,
            self.op.start.to_usize().unwrap(),
            self.op.end.to_usize().unwrap(),
        )?;
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;
        let output = model.wire_node("slice", self.op.clone(), &[wire])?;
        model.set_output_outlets(&output)?;
        let mut output = runtime.prepare(model)?.run(tvec![input.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<SliceProblem>("proptest", ());
    Ok(suite)
}
