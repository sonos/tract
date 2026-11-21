use infra::Test;
use infra::TestSuite;
use tract_hir::internal::*;
use tract_onnx::pb::*;

#[derive(Debug, Clone)]
struct SeluDefaults {
    opset: i64,
    alpha: f64,
    gamma: f64,
}

impl Test for SeluDefaults {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        _approximation: Approximation,
    ) -> TractResult<()> {
        let values = [-1.0f32, 1.0];
        let expected = [(self.alpha * self.gamma * (-1.0f64).exp_m1()) as f32, self.gamma as f32];
        super::check_with_opset(
            NodeProto {
                op_type: "Selu".into(),
                input: vec!["x".into()],
                output: vec!["y".into()],
                ..Default::default()
            },
            vec![("x", Tensor::from_shape(&[2], &values)?)],
            vec![],
            vec![Tensor::from_shape(&[2], &expected)?],
            runtime,
            Approximation::Custom(1e-6, 1e-6, 0.0),
            self.opset,
        )
    }
}

pub fn suite() -> TestSuite {
    let mut suite = TestSuite::default();
    suite.add_test("opset_1", SeluDefaults { opset: 1, alpha: 1.6732, gamma: 1.0507 });
    suite.add_test("opset_6", SeluDefaults { opset: 6, alpha: 1.6732632, gamma: 1.050701 });
    suite
}
