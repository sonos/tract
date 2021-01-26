use crate::internal::*;
use crate::ops::cnn::{KernelFormat, PaddingSpec};
use crate::ops::nn::DataFormat;

mod deconv_sum;
mod unary;
#[cfg(test)]
mod proptest;

pub use unary::DeconvUnary;

pub fn output_shape<D: DimLike>(
    data_format: &DataFormat,
    kernel_format: &KernelFormat,
    padding: &PaddingSpec,
    kernel_shape: &[usize],
    x_shape: &[D],
) -> TractResult<TVec<D>> {
    let x_shape = data_format.shape(x_shape)?;
    let spatial_input_shape = x_shape.hw_dims();
    let spatial_kernel_shape = match kernel_format {
        KernelFormat::OIHW => &kernel_shape[2..],
        KernelFormat::HWIO => &kernel_shape[..kernel_shape.len() - 2],
    };
    let ones = tvec!(1; spatial_input_shape.len());
    let spatial_output_shape =
        padding.compute_for_deconv(&spatial_input_shape, &spatial_kernel_shape, &ones, &ones);
    let deconv_shape: TVec<D> =
        spatial_output_shape.iter().map(|comp| comp.deconvoluted.clone()).collect();
    let output_shape = data_format.from_n_c_hw(
        x_shape.n().cloned().unwrap_or(1.into()),
        kernel_shape[1].into(), // FIXME Kernel format
        deconv_shape,
    )?;
    Ok(output_shape.shape.into())
}
