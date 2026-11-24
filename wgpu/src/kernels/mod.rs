pub mod apply_rope;
pub mod bin_ops;
pub mod cast;
pub mod causal_conv1d_update;
pub mod chain;
pub mod compare_select;
pub mod conv;
pub mod conv_splitk;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod coop_gemm;
pub mod copy;
pub mod deconv;
pub mod diag_gather;
pub mod element_wise;
pub mod gather;
pub mod gather_nd;
pub mod gelu_approximate;
pub mod gemv_pair;
pub mod group_norm;
pub mod ingest;
pub mod layer_norm;
pub mod leaky_relu;
pub mod mamba2_recurrent;
pub mod matmul;
pub mod matmul_bq;
pub mod nan_replace;
pub mod pool;
pub mod reduce;
pub mod resize;
pub mod rms_norm;
pub mod routing;
pub mod scaled_masked_softmax;
pub mod sdpa;
pub mod shaders;
pub mod softmax;
pub mod subgroup;
pub mod tiled_gemm;
pub mod topk;

/// Every pipeline this crate's kernels can need. They are built when the
/// context is created so that no frame ever waits on a shader compile.
pub fn all_pipeline_keys(shader_f16: bool) -> Vec<shaders::PipelineKey> {
    let mut keys = vec![];
    keys.extend(bin_ops::all_pipeline_keys(shader_f16));
    keys.extend(cast::all_pipeline_keys(shader_f16));
    keys.extend(conv::all_pipeline_keys(shader_f16));
    keys.extend(copy::all_pipeline_keys(shader_f16));
    keys.extend(deconv::all_pipeline_keys(shader_f16));
    keys.extend(element_wise::all_pipeline_keys(shader_f16));
    keys.extend(ingest::all_pipeline_keys(shader_f16));
    keys.extend(matmul::all_pipeline_keys(shader_f16));
    keys.extend(pool::all_pipeline_keys(shader_f16));
    keys.extend(reduce::all_pipeline_keys(shader_f16));
    keys.extend(resize::all_pipeline_keys(shader_f16));
    keys.extend(softmax::all_pipeline_keys(shader_f16));
    keys
}

tract_core::declare_knob!(
    scopable TRACT_WGPU_SUBGROUPS,
    bool,
    false,
    "Experimental subgroup GEMV and subgroup-matrix GEMM. Off by default. Read when the device \
     is created, so a device opened without it cannot use the feature later, and read again at \
     dispatch. Sizes and tiles come from the adapter: Chrome, Dawn, native wgpu, and WebKit each \
     advertise a different set. f16 matrix tiles that accumulate in f16 also require \
     TRACT_WGPU_ALLOW_F16_ACCUMULATION."
);

tract_core::declare_knob!(
    scopable TRACT_WGPU_ALLOW_F16_ACCUMULATION,
    bool,
    false,
    "Let f16 convolutions and matrix products accumulate in f16 in the blocked, split-K and \
     direct kernels instead of widening to f32: faster, but sums can lose precision or overflow. \
     Read when a model is prepared."
);
