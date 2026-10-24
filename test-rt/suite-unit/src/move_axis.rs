use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct MoveAxisProblem {
    pub input_shape: Vec<usize>,
    pub dt: DatumType,
    pub from: usize,
    pub to: usize,
}

impl Arbitrary for MoveAxisProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<MoveAxisProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        let dts = prop_oneof![
            Just(DatumType::F32),
            Just(DatumType::F16),
            Just(DatumType::I8),
            Just(DatumType::I64)
        ];
        (vec(prop_oneof![1..5usize, 28..70usize], 2..5usize), dts)
            .prop_filter("bounded size", |(shape, _)| shape.iter().product::<usize>() <= 1 << 18)
            .prop_flat_map(|(input_shape, dt)| {
                let rank = input_shape.len();
                (Just(input_shape), Just(dt), 0..rank, 0..rank)
            })
            .prop_filter("an actual move", |(_, _, from, to)| from != to)
            .prop_map(|(input_shape, dt, from, to)| MoveAxisProblem { input_shape, dt, from, to })
            .boxed()
    }
}

impl Test for MoveAxisProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let len = self.input_shape.iter().product::<usize>();
        let input = tensor1(&(0..len).map(|ix| (ix % 127) as f32).collect::<Vec<_>>())
            .into_shape(&self.input_shape)?
            .cast_to_dt(self.dt)?
            .into_owned();
        let op = AxisOp::Move(self.from, self.to);
        let mut reference = input.clone();
        op.change_tensor(&mut reference, false)?;
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;
        let output = model.wire_node("move", op, &[wire])?;
        model.select_output_outlets(&output)?;
        let prepared = runtime.prepare(model)?;
        let mut output = prepared.run(tvec![input.into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<MoveAxisProblem>("proptest", ());

    let mut case = |name: &str, shape: &[usize], dt: DatumType, from: usize, to: usize| {
        suite.add(name, MoveAxisProblem { input_shape: shape.to_vec(), dt, from, to });
    };
    // Extents off the transpose tile on both sides of the swap, each direction.
    case("nchw_to_nhwc", &[1, 64, 33, 35], DatumType::F16, 1, 3);
    case("nhwc_to_nchw", &[1, 33, 35, 64], DatumType::F32, 3, 1);
    // A batch of transposes, and every element size the copy kernels take.
    case("batched_u8", &[3, 40, 70], DatumType::I8, 1, 2);
    case("batched_u64", &[2, 33, 65], DatumType::I64, 2, 1);
    // Axes that merge into the transposed pair, as a depth-to-space spells them.
    case("merged_tail", &[2, 48, 5, 2, 7, 2], DatumType::F16, 1, 5);
    // Narrower than a tile on one side.
    case("narrow", &[1, 4, 100], DatumType::F32, 1, 2);
    Ok(suite)
}
