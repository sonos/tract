use tract_hir::internal::*;
use tract_nnef::ops::*;

pub fn tract_nnef_onnx_registry() -> TractResult<Registry> {
    let mut primitives: Registry = Registry::new("tract_onnx");
    Ok(primitives)
}
