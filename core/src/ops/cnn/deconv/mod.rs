use crate::internal::*;
use crate::ops::cnn::PoolSpec;

mod deconv_sum;
#[cfg(test)]
mod proptest;
mod unary;

pub use unary::DeconvUnary;

pub fn output_shape<D: DimLike>(
    pool_spec: &PoolSpec,
    x_shape: &[D],
    adjustments: &[usize],
) -> TractResult<TVec<D>> {
    let x_shape = pool_spec.data_format.shape(x_shape)?;
    let spatial_input_shape = x_shape.hw_dims();
    let spatial_output_details = pool_spec.padding.compute_for_deconv(
        &spatial_input_shape,
        &pool_spec.kernel_shape,
        &pool_spec.dilations(),
        &pool_spec.strides(),
        &adjustments,
    );
    let deconv_shape: TVec<D> =
        spatial_output_details.iter().map(|comp| comp.deconvoluted.clone()).collect();
    let co = pool_spec.output_channel_override.unwrap();
    let output_shape = pool_spec.data_format.from_n_c_hw(
        x_shape.n().cloned().unwrap_or(1.into()),
        co.into(),
        deconv_shape,
    )?;
    Ok(output_shape.shape.into())
}

pub fn adjustments(
    pool_spec: &PoolSpec,
    input_geo: &[usize],
    output_geo: &[usize],
) -> TractResult<TVec<usize>> {
    tract_itertools::izip!(
        input_geo,
        &pool_spec.kernel_shape,
        output_geo,
        pool_spec.strides().as_ref(),
        pool_spec.dilations().as_ref(),
    )
    .map(|(x, k, y, s, d)| {
        let pad = y.to_usize()? - s * (x.to_usize()? - 1) - (k.to_usize()? - 1) * d - 1;
        Ok(pad)
    })
    .collect::<TractResult<TVec<usize>>>()
}
