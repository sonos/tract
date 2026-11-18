use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::array::Tile;
use tract_ndarray::ArrayD;

#[derive(Debug, Clone, Default)]
struct TileProblem {
    input_shape: Vec<usize>,
    multipliers: Vec<usize>,
}

impl Arbitrary for TileProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<TileProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(1..4usize, 1..4usize)
            .prop_flat_map(|input_shape| {
                let rank = input_shape.len();
                (Just(input_shape), vec(1..4usize, rank..=rank))
            })
            .prop_map(|(input_shape, multipliers)| TileProblem { input_shape, multipliers })
            .boxed()
    }
}

impl Test for TileProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let len = self.input_shape.iter().product::<usize>();
        let data: Vec<f32> = (0..len).map(|x| x as f32 + 1.0).collect();
        let input = ArrayD::from_shape_vec(self.input_shape.clone(), data)?;
        let shape: Vec<usize> =
            self.input_shape.iter().zip(&self.multipliers).map(|(d, m)| d * m).collect();
        let reference = ArrayD::from_shape_fn(shape, |coords| {
            let source: Vec<usize> =
                (0..self.input_shape.len()).map(|ax| coords[ax] % self.input_shape[ax]).collect();
            input[&*source]
        })
        .into_tensor();
        let input = input.into_tensor();
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;
        let op = Tile::new(self.multipliers.iter().map(|m| m.to_dim()).collect());
        let output = model.wire_node("tile", op, &[wire])?;
        model.select_output_outlets(&output)?;
        let prepared = runtime.prepare(model)?;
        let mut output = prepared.run(tvec![input.into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<TileProblem>("proptest", ());
    suite.add(
        "outer_axis_0",
        TileProblem { input_shape: vec![1, 1, 1], multipliers: vec![2, 1, 1] },
    );
    suite.add(
        "every_axis_0",
        TileProblem { input_shape: vec![2, 3, 2], multipliers: vec![2, 3, 2] },
    );
    Ok(suite)
}
