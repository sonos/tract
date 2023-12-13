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

fn ignore_unit(_t: &[String], _tc: &dyn Test) -> bool {
    false
}

fn ignore_onnx(t: &[String]) -> bool {
    t.last().unwrap().starts_with("test_logsoftmax_large_number")
        || t.last().unwrap().starts_with("test_softmax_large_number")
        || t.last().unwrap().starts_with("test_resize")
        || t.last().unwrap() == "test_reduce_prod_default_axes_keepdims_example"
}
