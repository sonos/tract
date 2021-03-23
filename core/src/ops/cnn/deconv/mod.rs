use crate::internal::*;
use crate::ops::cnn::KernelFormat;
use crate::ops::cnn::PoolSpec;

mod deconv_sum;
#[cfg(test)]
mod proptest;
mod unary;

pub use unary::DeconvUnary;

pub fn output_shape<D: DimLike>(
    pool_spec: &PoolSpec,
    kernel_format: &KernelFormat,
    x_shape: &[D],
    adjustments: &[usize],
) -> TractResult<TVec<D>> {
    let x_shape = pool_spec.data_format.shape(x_shape)?;
    let spatial_input_shape = x_shape.hw_dims();
    let spatial_kernel_shape = kernel_format.spatial_shape(&pool_spec.kernel_shape);
    let spatial_output_details = pool_spec.padding.compute_for_deconv(
        &spatial_input_shape,
        &spatial_kernel_shape,
        &pool_spec.dilations(),
        &pool_spec.strides(),
        &adjustments,
    );
    let deconv_shape: TVec<D> =
        spatial_output_details.iter().map(|comp| comp.deconvoluted.clone()).collect();
    let co = match kernel_format {
        KernelFormat::HWIO => pool_spec.kernel_shape[pool_spec.kernel_shape.len() - 2],
        KernelFormat::OIHW => pool_spec.kernel_shape[1],
    };
    let output_shape = pool_spec.data_format.from_n_c_hw(
        x_shape.n().cloned().unwrap_or(1.into()),
        co.into(),
        deconv_shape,
    )?;
    Ok(output_shape.shape.into())
}
