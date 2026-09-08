pub mod bin_ops;
pub mod cast;
pub mod copy;
pub mod element_wise;
pub mod shaders;

/// Every pipeline this crate's kernels can need. They are built when the
/// context is created so that no frame ever waits on a shader compile.
pub fn all_pipeline_keys(shader_f16: bool) -> Vec<shaders::PipelineKey> {
    let mut keys = vec![];
    keys.extend(bin_ops::all_pipeline_keys(shader_f16));
    keys.extend(cast::all_pipeline_keys(shader_f16));
    keys.extend(copy::all_pipeline_keys(shader_f16));
    keys.extend(element_wise::all_pipeline_keys(shader_f16));
    keys
}
