use infra::Test;
use infra::TestSuite;
use tract_hir::internal::*;
use tract_onnx::pb::*;

#[derive(Debug, Clone)]
struct CeluCase {
    alpha: f32,
}

impl Test for CeluCase {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approximation: Approximation,
    ) -> TractResult<()> {
        let values = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 2.0];
        let expected: Vec<f32> = values
            .iter()
            .map(|&x| {
                let alpha = f64::from(self.alpha);
                f64::from(x).max(0.0) as f32
                    + (alpha * (f64::from(x) / alpha).exp_m1()).min(0.0) as f32
            })
            .collect();
        super::check(
            NodeProto {
                op_type: "Celu".into(),
                input: vec!["x".into()],
                output: vec!["y".into()],
                attribute: vec![AttributeProto {
                    name: "alpha".into(),
                    r#type: 1,
                    f: self.alpha,
                    ..Default::default()
                }],
                ..Default::default()
            },
            vec![("x", Tensor::from_shape(&[2, 3], &values)?)],
            vec![],
            vec![Tensor::from_shape(&[2, 3], &expected)?],
            runtime,
            approximation,
        )
    }
}

pub fn suite() -> TestSuite {
    let mut suite = TestSuite::default();
    for (name, alpha) in [("half", 0.5), ("unit", 1.0), ("double", 2.0)] {
        suite.add_test(name, CeluCase { alpha });
    }
    suite
}
