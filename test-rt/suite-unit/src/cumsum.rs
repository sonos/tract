use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::Axis;
use tract_core::ndarray::Dimension;
use tract_core::ndarray::IxDyn;
use tract_core::ops::array::CumSum;

#[derive(Debug, Clone)]
struct CumSumProblem {
    input: ArrayD<f32>,
    op: CumSum,
}

impl Arbitrary for CumSumProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<CumSumProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(0..6usize, 1..5usize)
            .prop_flat_map(|shape| {
                let len = shape.iter().product::<usize>();
                let rank = shape.len();
                (Just(shape), vec(-5i32..5, len..=len), 0..rank, any::<bool>(), any::<bool>())
            })
            .prop_map(|(shape, values, axis, exclusive, reverse)| CumSumProblem {
                input: ArrayD::from_shape_vec(
                    IxDyn(&shape),
                    values.into_iter().map(|v| v as f32).collect(),
                )
                .unwrap(),
                op: CumSum { axis, exclusive, reverse },
            })
            .boxed()
    }
}

impl CumSumProblem {
    fn reference(&self) -> ArrayD<f32> {
        let mut output = ArrayD::zeros(self.input.shape());
        let len = self.input.shape()[self.op.axis];
        for (mut out, lane) in output
            .lanes_mut(Axis(self.op.axis))
            .into_iter()
            .zip(self.input.lanes(Axis(self.op.axis)))
        {
            for i in 0..len {
                let range = match (self.op.reverse, self.op.exclusive) {
                    (false, false) => 0..i + 1,
                    (false, true) => 0..i,
                    (true, false) => i..len,
                    (true, true) => i + 1..len,
                };
                out[i] = range.map(|j| lane[j]).sum();
            }
        }
        output
    }
}

impl Test for CumSumProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", f32::fact(self.input.shape()))?;
        let output = model.wire_node("cumsum", self.op.clone(), &[wire])?;
        model.select_output_outlets(&output)?;
        let mut output =
            runtime.prepare(model)?.run(tvec![self.input.clone().into_tensor().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&self.reference().into_tensor(), approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<CumSumProblem>("proptest", ());
    let case = |shape: &[usize], axis, exclusive, reverse| CumSumProblem {
        input: ArrayD::from_shape_fn(IxDyn(shape), |ix| {
            (0..ix.ndim()).map(|k| (k + 1) * ix[k]).sum::<usize>() as f32 - 2.0
        }),
        op: CumSum { axis, exclusive, reverse },
    };
    suite.add_test("inner_axis", case(&[2, 7], 1, false, false));
    suite.add_test("outer_axis", case(&[5, 3, 2], 0, false, false));
    suite.add_test("exclusive_reverse", case(&[3, 4, 2], 1, true, true));
    suite.add_test("single_step", case(&[1, 4], 0, true, false));
    suite.add_test("single_step_inclusive", case(&[1, 4], 0, false, true));
    Ok(suite)
}
