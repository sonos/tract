use infra::Test;
use suite_unit::bin_einsum::{BinEinsumProblem, BinEinsumProblemParams};

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
    infra::TestSuite::default().with("onnx", onnx).with("unit", unit)
}

fn ignore_unit(_t: &[String], _case: &dyn Test) -> bool {
    false
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
