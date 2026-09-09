pub mod cast;
pub mod copy;
pub mod shaders;

/// Every pipeline this crate's kernels can need. They are built when the
/// context is created so that no frame ever waits on a shader compile.
pub fn all_pipeline_keys(shader_f16: bool) -> Vec<shaders::PipelineKey> {
    let mut keys = vec![];
    keys.extend(cast::all_pipeline_keys(shader_f16));
    keys.extend(copy::all_pipeline_keys(shader_f16));
    keys
}
