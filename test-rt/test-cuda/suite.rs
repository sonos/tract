use std::vec;

use infra::Test;
use suite_unit::bin_einsum::{BinEinsumProblem, BinEinsumProblemParams};
use suite_unit::conv_f16::ConvProblemF16;
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

    unit.get_sub_mut("conv_f32")
        .add_arbitrary::<ConvProblem>("proptest", ConvProblemParams::default());

    unit.get_sub_mut("conv_f16")
        .add_arbitrary::<ConvProblemF16>("proptest", ConvProblemParams::default());

    unit.get_sub_mut("sdpa").add_arbitrary::<SdpaProblem<half::f16>>(
        "proptest_f16",
        SdpaProblemParams { embed_dims: vec![64, 128] },
    );
    let pulse = suite_pulse::suite().unwrap().clone();

    infra::TestSuite::default().with("onnx", onnx).with("unit", unit).with("pulse", pulse)
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

/// Cases the CUDA runtime declines: the SDPA kernel takes only a few head dims, so every
/// Attention case that keeps an SDPA node rejects the small ones the ONNX suite uses, and
/// CudaGgmlGemm has no broadcast batch.
fn ignore_onnx(t: &[String]) -> bool {
    r#"
    test_slice_start_out_of_bounds
    test_nllloss_NCd1d2d3d4d5_mean_weight_expanded
    test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded
    test_tril_zero
    test_triu_zero
    test_attention_3d
    test_attention_3d_attn_mask
    test_attention_3d_causal
    test_attention_3d_diff_heads_sizes
    test_attention_3d_diff_heads_sizes_attn_mask
    test_attention_3d_diff_heads_sizes_causal
    test_attention_3d_diff_heads_sizes_scaled
    test_attention_3d_diff_heads_with_past_and_present
    test_attention_3d_gqa
    test_attention_3d_gqa_attn_mask
    test_attention_3d_gqa_causal
    test_attention_3d_gqa_scaled
    test_attention_3d_gqa_with_past_and_present
    test_attention_3d_scaled
    test_attention_3d_transpose_verification
    test_attention_3d_with_past_and_present
    test_attention_4d
    test_attention_4d_attn_mask
    test_attention_4d_attn_mask_3d
    test_attention_4d_attn_mask_3d_causal
    test_attention_4d_attn_mask_4d
    test_attention_4d_attn_mask_4d_causal
    test_attention_4d_causal
    test_attention_4d_diff_heads_sizes
    test_attention_4d_diff_heads_sizes_attn_mask
    test_attention_4d_diff_heads_sizes_causal
    test_attention_4d_diff_heads_sizes_scaled
    test_attention_4d_diff_heads_with_past_and_present
    test_attention_4d_diff_heads_with_past_and_present_mask3d
    test_attention_4d_diff_heads_with_past_and_present_mask4d
    test_attention_4d_fp16
    test_attention_4d_gqa
    test_attention_4d_gqa_attn_mask
    test_attention_4d_gqa_causal
    test_attention_4d_gqa_scaled
    test_attention_4d_gqa_with_past_and_present
    test_attention_4d_gqa_with_past_and_present_fp16
    test_attention_4d_scaled
    test_attention_4d_with_past_and_present
    test_matmul_bcast
    "#
    .trim()
    .lines()
    .any(|s| t.last().unwrap() == s.trim())
}
