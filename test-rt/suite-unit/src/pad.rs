use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};
use tract_ndarray::{ArrayD, Slice};

#[derive(Debug, Clone, Default)]
struct PadProblem {
    input_shape: Vec<usize>,
    pads: Vec<(usize, usize)>,
    value: f32,
}

impl Arbitrary for PadProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<PadProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(1..5usize, 1..5usize)
            .prop_flat_map(|input_shape| {
                let rank = input_shape.len();
                (
                    Just(input_shape),
                    vec((0..3usize, 0..3usize), rank..=rank),
                    prop_oneof![Just(0i8), -3i8..3i8].prop_map(|v| v as f32),
                )
            })
            .prop_map(|(input_shape, pads, value)| PadProblem { input_shape, pads, value })
            .boxed()
    }
}

impl PadProblem {
    fn reference(&self, input: &Tensor) -> TractResult<Tensor> {
        let input = input.to_plain_array_view::<f32>()?;
        let shape: Vec<usize> =
            self.input_shape.iter().zip(&self.pads).map(|(d, (b, a))| d + b + a).collect();
        let mut output = ArrayD::from_elem(shape, self.value);
        let mut view = output.view_mut();
        for (ax, (&(before, _), &dim)) in self.pads.iter().zip(&self.input_shape).enumerate() {
            view.slice_axis_inplace(tract_ndarray::Axis(ax), Slice::from(before..before + dim));
        }
        view.assign(&input);
        Ok(output.into_tensor())
    }
}

impl Test for PadProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut input = Tensor::zero::<f32>(&self.input_shape)?;
        input
            .try_as_plain_ram_mut()?
            .as_slice_mut::<f32>()?
            .iter_mut()
            .enumerate()
            .for_each(|(ix, x)| *x = ix as f32 + 1.0);
        let reference = self.reference(&input)?;
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;
        let op = Pad::new(self.pads.clone(), PadMode::Constant(rctensor0(self.value)));
        let output = model.wire_node("pad", op, &[wire])?;
        model.select_output_outlets(&output)?;
        let prepared = runtime.prepare(model)?;
        let mut output = prepared.run(tvec![input.into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<PadProblem>("proptest", ());

    suite.add(
        "inner_axis_0",
        PadProblem {
            input_shape: vec![1, 1, 3, 32],
            pads: vec![(0, 0), (0, 0), (0, 0), (1, 1)],
            value: 0.0,
        },
    );

    suite.add(
        "outer_axis_nonzero_0",
        PadProblem { input_shape: vec![2, 3, 2], pads: vec![(1, 2), (0, 0), (0, 0)], value: 1.5 },
    );

    suite.add(
        "no_pad_0",
        PadProblem { input_shape: vec![2, 3], pads: vec![(0, 0), (0, 0)], value: 0.0 },
    );

    Ok(suite)
}
