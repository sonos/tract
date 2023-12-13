use crate::internal::*;
use crate::ops::cnn::{PaddingSpec, PoolSpec};

#[allow(clippy::module_inception)]
mod deconv;
mod deconv_sum;

pub use deconv::Deconv;

pub fn output_shape<D: DimLike>(
    pool_spec: &PoolSpec,
    x_shape: &[D],
    adjustments: &[usize],
) -> TractResult<TVec<D>> {
    let x_shape = pool_spec.data_format.shape(x_shape)?;
    let spatial_input_shape = x_shape.hw_dims();
    let spatial_output_details = pool_spec.padding.compute_for_deconv(
        spatial_input_shape,
        &pool_spec.kernel_shape,
        &pool_spec.dilations(),
        &pool_spec.strides(),
        adjustments,
    )?;
    let deconv_shape: TVec<D> =
        spatial_output_details.iter().map(|comp| comp.deconvoluted.clone()).collect();
    let output_shape = pool_spec.data_format.from_n_c_hw(
        x_shape.n().cloned().unwrap_or_else(|| 1.into()),
        pool_spec.output_channels.into(),
        deconv_shape,
    )?;
    Ok(output_shape.shape)
}

pub fn adjustments(
    pool_spec: &PoolSpec,
    input_geo: &[usize],
    output_geo: &[usize],
) -> TractResult<TVec<usize>> {
    debug_assert_eq!(pool_spec.rank(), pool_spec.strides().len());
    debug_assert_eq!(pool_spec.rank(), pool_spec.dilations().len());
    debug_assert_eq!(pool_spec.rank(), pool_spec.kernel_shape.len());
    debug_assert_eq!(pool_spec.rank(), input_geo.len());
    debug_assert_eq!(pool_spec.rank(), output_geo.len());
    let rank = pool_spec.rank();
    let pad: TVec<usize> = match &pool_spec.padding {
        PaddingSpec::Explicit(beg, end) => (0..rank).map(|r| beg[r] + end[r]).collect(),
        PaddingSpec::Valid => tvec!(0; rank),
        pad => todo!("Unsupported padding in deconvolution arguments {pad:?}"),
    };
    tract_itertools::izip!(
        input_geo,
        &pool_spec.kernel_shape,
        output_geo,
        pool_spec.strides().as_ref(),
        pool_spec.dilations().as_ref(),
        pad,
    )
    .map(|(x, k, y, s, d, p)| {
        let adj = y.to_usize()? + p - s * (x.to_usize()? - 1) - (k.to_usize()? - 1) * d - 1;
        Ok(adj)
    })
    .collect::<TractResult<TVec<usize>>>()
}
