use infra::Test;
use suite_unit::conv_f32::{ConvProblem, ConvProblemParams};
use suite_unit::conv_q::{QConvProblem, QConvProblemParams};
use tract_core::internal::*;

pub fn suite() -> &'static infra::TestSuite {
    lazy_static::lazy_static! {
        static ref SUITE: infra::TestSuite  = mk_suite();
    };
    &SUITE
}

#[allow(clippy::needless_update)]
fn mk_suite() -> infra::TestSuite {
    let mut onnx = suite_onnx::suite().clone();
    /*
    onnx.ignore(&ignore_onnx);
    onnx.skip(&skip_onnx);
    */

    let mut unit = suite_unit::suite().unwrap().clone();
    unit.ignore_case(&ignore_unit);
    /*
    let cv =
        ConvProblemParams { no_group: true, geo_rank: Some(1..3), ..ConvProblemParams::default() };
    unit.get_sub_mut("conv_f32").add_arbitrary::<ConvProblem>("proptest", cv.clone());
    unit.get_sub_mut("conv_q").add_arbitrary_with_filter::<QConvProblem>(
        "proptest",
        QConvProblemParams { conv: cv, tflite_rules: true, ..QConvProblemParams::default() },
        compatible_conv_q,
    );
    */
    infra::TestSuite::default().with("onnx", onnx).with("unit", unit)
}

fn ignore_unit(t: &[String], tc: &dyn Test) -> bool {
    t[0] == "conv_q"
}
