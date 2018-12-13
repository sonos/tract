use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::{DataFormat, Patch};

#[derive(Debug, Clone, new)]
pub struct ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub patch: Patch,
    pub full_output_shape: TVec<usize>,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub kernel_is_hwio: bool,
    pub kernel: Array2<D>,
    pub bias: Option<ArrayD<D>>,
    pub group: usize,
}

impl<D> ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub(super) fn conv_gemm<'i>(&'i self, mega_matrix: &'i ArrayView2<'i, D>) -> TractResult<ArrayD<D>> {
        let mut output = unsafe { ArrayD::<D>::uninitialized(&*self.full_output_shape) };
        let input_shape = &self.patch.input_shape;
        let co_per_group = self.full_output_shape[input_shape.c_axis()] / self.group;
        for i in 0..input_shape.n_dim() {
            for g in 0..self.group {
                let mm_offset = self.n * (g + (i * self.group));
                let mut output_subview = output.view_mut();
                output_subview.slice_axis_inplace(Axis(input_shape.n_axis()), (i..(i + 1)).into());
                output_subview.slice_axis_inplace(
                    Axis(input_shape.c_axis()),
                    (g * co_per_group..(g + 1) * co_per_group).into(),
                );
                match self.patch.input_shape.fmt {
                    DataFormat::NHWC => {
                        let mut output_panel = output_subview.into_shape((self.n, self.m))?;
                        ::ndarray::linalg::general_mat_mul(
                            D::one(),
                            &self.kernel.slice_axis(
                                Axis(0),
                                (co_per_group * g..co_per_group * (g + 1)).into(),
                            ),
                            &mega_matrix
                                .slice_axis(Axis(1), (mm_offset..(mm_offset + self.n)).into()),
                            D::zero(),
                            &mut output_panel.reversed_axes(),
                        );
                    }
                    DataFormat::NCHW => {
                        let mut output_panel =
                            output_subview.into_shape((self.m / self.group, self.n))?;
                        ::ndarray::linalg::general_mat_mul(
                            D::one(),
                            &self.kernel.slice_axis(
                                Axis(0),
                                (co_per_group * g..co_per_group * (g + 1)).into(),
                            ),
                            &mega_matrix
                                .slice_axis(Axis(1), (mm_offset..(mm_offset + self.n)).into()),
                            D::zero(),
                            &mut output_panel,
                        );
                    }
                }
            }
        }

        if let Some(ref bias) = self.bias {
            output += &bias;
        }

        Ok(output)
    }
}
