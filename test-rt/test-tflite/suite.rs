use infra::Test;
use regex::Regex;
use suite_unit::bin_einsum::{BinEinsumProblem, BinEinsumProblemParams};
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
    onnx.ignore(&ignore_onnx);
    onnx.skip(&skip_onnx);

    let mut unit = suite_unit::suite().unwrap().clone();
    unit.ignore_case(&ignore_unit);
    let cv =
        ConvProblemParams { no_group: true, geo_rank: Some(1..3), ..ConvProblemParams::default() };
    unit.get_sub_mut("conv_f32").add_arbitrary::<ConvProblem>("proptest", cv.clone());
    unit.get_sub_mut("conv_q").add_arbitrary_with_filter::<QConvProblem>(
        "proptest",
        QConvProblemParams { conv: cv, tflite_rules: true, ..QConvProblemParams::default() },
        compatible_conv_q,
    );

    let einsum_params = BinEinsumProblemParams { max_dims: 4, ..BinEinsumProblemParams::default() };
    unit.get_sub_mut("bin_einsum")
        .add_arbitrary::<BinEinsumProblem>("proptest", einsum_params.clone());
    infra::TestSuite::default().with("onnx", onnx).with("unit", unit)
}

fn patterns(s: &str) -> Vec<Regex> {
    s.trim()
        .lines()
        .map(|s| s.split_once('#').map(|(left, _)| left).unwrap_or(s).trim())
        .filter(|s| !s.is_empty())
        .map(|pat| Regex::new(pat).unwrap())
        .collect()
}

fn ignore_onnx(t: &[String]) -> bool {
    let name = t.last().unwrap();
    let included = patterns(
        "
        _conv_
        Conv1d
        Conv2d

        test_averagepool_2d
        test_maxpool_2d

        squeeze
        _transpose_
        test_concat
        test_flatten
        test_reshape
        test_slice
        test_split

        test_where
        test_less
        test_greater
        test_equal
        test_not

        test_add
        test_mul
        test_sub
        test_div
        test_and
        test_or

        test_reduce
        test_softmax

        test_abs
        test_ceil
        test_exp
        test_floor
        test_log
        test_reciprocal
        test_square
        test_sqrt
        test_rsqrt

        test_cos
        test_sin
        # lol, no tan :)

        test_clip
        test_batchnorm
        test_hardswish
        test_leakyrelu
        test_prelu
        test_relu
        test_selu
        test_sigmoid
        test_tanh
        test_thresholdrelu
        ",
    );
    let excluded = patterns(
        "
            test_slice_start_out_of_bounds
            test_Conv1d_groups
            test_Conv2d_groups
            test_Conv1d_depthwise_with_multiplier
            test_Conv2d_depthwise_with_multiplier
            test_Conv2d_groups_thnn
            test_reshape_allowzero_reordered
            test_split_zero_size
            test_mul_uint8
            test_div_uint8
            test_reduce_log_sum_exp.*           # tflite does not support f64 reducers 🤷
            pool_2d_ceil
            pool_2d_pads
            pool_2d_precomputed_pads_count_include_pad
            pool_2d_same_lower
            test_cosh.*
            test_sinh.*
            ",
    );
    !included.iter().any(|pat| pat.is_match(name)) || excluded.iter().any(|pat| pat.is_match(name))
}

// We must *never* run these, even in --ignored mode, as they trigger buggy aborts in tflite runtime!
fn skip_onnx(t: &[String]) -> bool {
    let name = t.last().unwrap();
    let excluded = "
            test_clip_default_int8_max_expanded
            test_clip_default_int8_min_expanded
            test_BatchNorm3d_eval
            test_BatchNorm3d_momentum_eval
            test_PReLU_3d
            ";
    excluded.split_whitespace().any(|s| s == name)
}

fn ignore_unit(t: &[String], case: &dyn Test) -> bool {
    if let Some(cp) = case.downcast_ref::<ConvProblem>() {
        if !compatible_conv_f32(cp) {
            return true;
        }
    }
    if let Some(qcp) = case.downcast_ref::<QConvProblem>() {
        if !compatible_conv_q(qcp) {
            return true;
        }
    }

    if t[0] == "bin_einsum" && t[1] == "proptest" {
        return true;
    }

    let [section, _unit] = t else { return false };
    [
        "apply_rope",
        "deconv",
        "gelu_approximate",
        "q_flavours",
        "q_binary",
        "q_elmwise",
        "matmul_q40",
        "rms_norm",
        "scaled_masked_softmax",
        "silu",
    ]
    .contains(&&**section)
}

fn compatible_conv_f32(qcp: &ConvProblem) -> bool {
    qcp.group == 1 && (qcp.kernel.ndim() == 4 || qcp.kernel.ndim() == 3)
}

fn compatible_conv_q(qcp: &QConvProblem) -> bool {
    if qcp.group != 1 {
        return false;
    }
    let idt = qcp.data.datum_type();
    let kdt = qcp.kernel.datum_type();
    let odt = qcp.raw_output_dt;
    if odt != idt.unquantized() {
        return false;
    }

    // all u8 and per-layer
    if idt.unquantized() == u8::datum_type()
        && kdt.unquantized() == u8::datum_type()
        && qcp.qp.iter().all(|qp| qp.is_uniform())
    {
        return true;
    }
    // all i8 and no zero_point
    if idt.unquantized() == i8::datum_type()
        && kdt.unquantized() == i8::datum_type()
        && qcp.qp[0].is_zero().unwrap()
        && qcp.qp[2].is_zero().unwrap()
        && qcp.qp[4].is_zero().unwrap()
    {
        return true;
    }
    false
}
