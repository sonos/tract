// Compile-only check: `tract_proxy::impl_ndarray_interop!` expands cleanly.
// Running requires libtract to be available for linking; that's handled by
// the `mobilenet` test's build path.

tract_proxy::impl_ndarray_interop!();

#[allow(dead_code)]
fn sanity(t: &tract_proxy::Tensor) -> anyhow::Result<()> {
    let arr = ndarray::Array4::<f32>::zeros((1, 3, 4, 5));
    let _converted: tract_proxy::Tensor = arr.tract()?;
    let _view = t.ndarray::<f32>()?;
    let _v4 = t.ndarray4::<f32>()?;
    let _v0 = t.ndarray0::<f32>()?;
    Ok(())
}
