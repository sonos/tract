use std::vec;

use infra::Test;
use suite_unit::bin_einsum::{BinEinsumProblem, BinEinsumProblemParams};
use suite_unit::conv_f32::{ConvProblem, ConvProblemParams};
use suite_unit::sdpa::{SdpaProblem, SdpaProblemParams};
use tract_core::num_traits::Float;
use tract_core::prelude::Datum;
use tract_core::tract_data::half;

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

    unit.get_sub_mut("bin_einsum").add_arbitrary::<BinEinsumProblem>(
        "proptest",
        BinEinsumProblemParams {
            force_unique_non_trivial_m_n: true,
            max_dims: 6,
            ..BinEinsumProblemParams::default()
        },
    );

    unit.get_sub_mut("conv_f32").add_arbitrary::<ConvProblem>(
        "proptest",
        ConvProblemParams { no_group: true, ..ConvProblemParams::default() },
    );

    unit.get_sub_mut("sdpa").add_arbitrary::<SdpaProblem<half::f16>>(
        "proptest_f16",
        SdpaProblemParams { embed_dims: vec![64, 128] },
    );
    infra::TestSuite::default().with("onnx", onnx).with("unit", unit)
}

fn ignore_unit(t: &[String], case: &dyn Test) -> bool {
    if let Some(sdpab) = case.downcast_ref::<SdpaProblem<f32>>() {
        return !compatible_sdpa::<f32>(sdpab);
    }

    if let Some(sdpab) = case.downcast_ref::<SdpaProblem<half::f16>>() {
        return !compatible_sdpa::<half::f16>(sdpab);
    }
    t[0] == "sdpa" && t[1] == "proptest_f32"
}

fn compatible_sdpa<F: Datum + Float>(sdpap: &SdpaProblem<F>) -> bool {
    matches!(sdpap.k.shape().last().unwrap(), 64 | 80 | 96 | 112 | 128 | 256)
}

fn ignore_onnx(t: &[String]) -> bool {
    r#"
    test_slice_start_out_of_bounds
    test_nllloss_NCd1d2d3d4d5_mean_weight_expanded
    test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded
    test_tril_zero
    test_triu_zero
    "#
    .trim()
    .lines()
    .any(|s| t.last().unwrap() == s.trim())
}
