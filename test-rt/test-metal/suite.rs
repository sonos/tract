use infra::Test;

pub fn suite() -> &'static infra::TestSuite {
    lazy_static::lazy_static! {
        static ref SUITE: infra::TestSuite  = mk_suite();
    };
    &SUITE
}

#[allow(clippy::needless_update)]
fn mk_suite() -> infra::TestSuite {
    let mut onnx = suite_onnx::suite().clone();
    onnx.ignore(&ignore_onnx);

    let mut unit = suite_unit::suite().unwrap().clone();
    unit.ignore_case(&ignore_unit);

    infra::TestSuite::default().with("onnx", onnx).with("unit", unit)
}

fn ignore_unit(_t: &[String], _case: &dyn Test) -> bool {
    false
}

fn ignore_onnx(_t: &[String]) -> bool {
    false
}
