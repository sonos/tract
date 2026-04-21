// Compile-only check: both forms of `tract_proxy::impl_ndarray_interop!`
// expand cleanly. Running requires libtract to be available for linking;
// that's handled by the `mobilenet` test's build path.

mod default_form {
    tract_proxy::impl_ndarray_interop!();

    #[allow(dead_code)]
    pub(super) fn sanity(t: &tract_proxy::Tensor) -> anyhow::Result<()> {
        let arr = ndarray::Array4::<f32>::zeros((1, 3, 4, 5));
        let _converted: tract_proxy::Tensor = arr.tract()?;
        let _view = t.ndarray::<f32>()?;
        let _v4 = t.ndarray4::<f32>()?;
        let _v0 = t.ndarray0::<f32>()?;
        Ok(())
    }
}

mod explicit_path {
    // Re-alias the ndarray crate to a different path to prove the explicit
    // form resolves against the caller-supplied root.
    use ndarray as my_nd;
    tract_proxy::impl_ndarray_interop!(my_nd);

    #[allow(dead_code)]
    pub(super) fn sanity(t: &tract_proxy::Tensor) -> anyhow::Result<()> {
        let arr = my_nd::Array4::<f32>::zeros((1, 3, 4, 5));
        let _converted: tract_proxy::Tensor = arr.tract()?;
        let _view = t.ndarray::<f32>()?;
        Ok(())
    }
}
