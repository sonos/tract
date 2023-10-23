fn main() {
    let mut suite = suite_onnx::suite().clone();
    suite.ignore(&ignore);
    suite.test_runtime("nnef_cycle", "suite_onnx::suite()", "nnef_cycle()", "Approximation::Approximate");
}

fn ignore(t: &[String]) -> bool {
    r#"
test_averagepool_2d_ceil
test_averagepool_2d_pads_count_include_pad
test_averagepool_2d_precomputed_pads_count_include_pad
test_averagepool_2d_same_lower
test_cast_STRING_to_FLOAT
test_castlike_STRING_to_FLOAT_expanded
test_constantlike_ones_with_input
test_constantlike_threes_with_shape_and_dtype
test_constantlike_zeros_without_input_dtype
test_cumsum_1d_exclusive
test_cumsum_1d_reverse_exclusive
test_cumsum_2d
test_dequantizelinear
test_dropout_random
test_dynamicquantizelinear
test_dynamicquantizelinear_max_adjusted
test_dynamicquantizelinear_min_adjusted
test_gemm_broadcast
test_gemm_nobroadcast
test_maxpool_2d_ceil
test_maxpool_2d_same_lower
test_maxpool_with_argmax_2d_precomputed_pads
test_mod_broadcast
test_mod_int64_fmod
test_mod_mixed_sign_float16
test_mod_mixed_sign_float32
test_mod_mixed_sign_float64
test_mod_mixed_sign_int16
test_mod_mixed_sign_int32
test_mod_mixed_sign_int64
test_mod_mixed_sign_int8
test_mod_uint16
test_mod_uint32
test_mod_uint64
test_mod_uint8
test_matmulinteger
test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded
test_nonzero_example
test_quantizelinear
test_qlinearmatmul_2D
test_qlinearmatmul_3D
test_reshape_reordered_dims
test_resize_upsample_scales_linear_align_corners
test_resize_downsample_scales_linear
test_unsqueeze
"#
    .trim()
    .lines()
    .any(|s| t.last().unwrap() == s.trim())
}
