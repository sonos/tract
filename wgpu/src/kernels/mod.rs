pub mod bin_ops;
pub mod cast;
pub mod chain;
pub mod conv;
pub mod copy;
pub mod deconv;
pub mod element_wise;
pub mod ingest;
pub mod matmul;
pub mod pool;
pub mod reduce;
pub mod resize;
pub mod shaders;
pub mod softmax;

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
