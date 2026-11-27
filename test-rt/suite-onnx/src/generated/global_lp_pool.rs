use infra::{Test, TestSuite};
use tract_hir::internal::*;
use tract_onnx::pb::NodeProto;

#[derive(Debug, Clone)]
struct GlobalLpPool {
    order: i64,
    expected: [f32; 2],
}

impl Test for GlobalLpPool {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approximation: Approximation,
    ) -> TractResult<()> {
        super::check(
            NodeProto {
                op_type: "GlobalLpPool".into(),
                input: vec!["x".into()],
                output: vec!["y".into()],
                attribute: vec![super::int_attr("p", self.order)],
                ..Default::default()
            },
            vec![("x", Tensor::from_shape(&[1, 2, 2], &[-3f32, 4., -2., 0.])?)],
            vec![],
            vec![Tensor::from_shape(&[1, 2, 1], &self.expected)?],
            runtime,
            approximation,
        )
    }
}

pub fn suite() -> TestSuite {
    let mut suite = TestSuite::default();
    suite.add_test("l1", GlobalLpPool { order: 1, expected: [7., 2.] });
    suite.add_test("l2", GlobalLpPool { order: 2, expected: [5., 2.] });
    suite.add_test("l3", GlobalLpPool { order: 3, expected: [91f32.cbrt(), 2.] });
    suite
}
