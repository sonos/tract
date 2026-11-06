use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::IxDyn;
use tract_core::ops::nn::Reduce;
use tract_core::ops::nn::Reducer;

/// ArgMax / ArgMin over one axis, checked against core's reducer. Values come
/// from a handful of levels plus NaN and infinities, so ties and skipped NaNs
/// are the common case.
#[derive(Debug, Clone)]
struct ReduceArgProblem {
    input: ArrayD<f32>,
    axis: usize,
    reducer: Reducer,
}

fn level(ix: u8) -> f32 {
    match ix {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        n => n as f32 - 6.0,
    }
}

fn reducer() -> impl Strategy<Value = Reducer> {
    (any::<bool>(), any::<bool>())
        .prop_map(|(max, last)| if max { Reducer::ArgMax(last) } else { Reducer::ArgMin(last) })
}

impl Arbitrary for ReduceArgProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<ReduceArgProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(1usize..6, 1..4usize)
            .prop_flat_map(|shape| {
                let len = shape.iter().product::<usize>();
                let rank = shape.len();
                (Just(shape), vec(0u8..10, len..=len), 0..rank, reducer())
            })
            .prop_map(|(shape, levels, axis, reducer)| ReduceArgProblem {
                input: ArrayD::from_shape_vec(
                    IxDyn(&shape),
                    levels.into_iter().map(level).collect(),
                )
                .unwrap(),
                axis,
                reducer,
            })
            .boxed()
    }
}

impl Test for ReduceArgProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        _approx: Approximation,
    ) -> infra::TestResult {
        let input = self.input.clone().into_tensor();
        let expected = self.reducer.reduce(&[self.axis], &input)?;
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", f32::fact(self.input.shape()))?;
        let op = Reduce { axes: tvec![self.axis], reducer: self.reducer };
        let output = model.wire_node("reduce", op, &[wire])?;
        model.select_output_outlets(&output)?;
        let mut output = runtime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        output.remove(0).into_tensor().close_enough(&expected, Approximation::Exact)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<ReduceArgProblem>("proptest", ());
    let long_row = |len: usize, reducer| {
        let input =
            ArrayD::from_shape_fn(IxDyn(&[2, len]), |ix| ((ix[0] * 31 + ix[1] * 7919) % 97) as f32);
        ReduceArgProblem { input, axis: 1, reducer }
    };
    for (name, reducer) in [
        ("argmax", Reducer::ArgMax(false)),
        ("argmax_last", Reducer::ArgMax(true)),
        ("argmin", Reducer::ArgMin(false)),
        ("argmin_last", Reducer::ArgMin(true)),
    ] {
        suite.add_test(format!("{name}_long_row"), long_row(3000, reducer));
        let all_nan = ArrayD::from_elem(IxDyn(&[3, 4]), f32::NAN);
        suite.add_test(
            format!("{name}_all_nan"),
            ReduceArgProblem { input: all_nan, axis: 1, reducer },
        );
    }
    Ok(suite)
}
