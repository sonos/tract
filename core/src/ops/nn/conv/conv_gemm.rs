use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::{DataFormat, Patch};
use ops::nn::conv::KernelFormat;

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
    pub kernel_fmt: KernelFormat,
    pub kernel: Array2<D>,
    pub bias: Option<ArrayD<D>>,
    pub group: usize,
}

impl<D> ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub(super) fn conv_gemm<'i>(
        &'i self,
        mega_matrix: &'i ArrayView2<'i, D>,
    ) -> TractResult<ArrayD<D>> {
        let mut output = unsafe { ArrayD::<D>::uninitialized(&*self.full_output_shape) };
        let input_shape = &self.patch.input_shape;

        let c_panel_shape = (self.m, self.n);
        let mut c_panel = unsafe { Array2::uninitialized(c_panel_shape) };

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
                ::ndarray::linalg::general_mat_mul(
                    D::one(),
                    &self
                        .kernel
                        .slice_axis(Axis(0), (co_per_group * g..co_per_group * (g + 1)).into()),
                    &mega_matrix.slice_axis(Axis(1), (mm_offset..(mm_offset + self.n)).into()),
                    D::zero(),
                    &mut c_panel,
                );
                let shape = output_subview.shape().to_vec();
                match self.patch.input_shape.fmt {
                    DataFormat::NHWC => output_subview.assign(&c_panel.t().into_shape(shape)?),
                    DataFormat::NCHW => output_subview.assign(&c_panel.view().into_shape(shape)?),
                };
            }
        }

        if let Some(ref bias) = self.bias {
            output += &bias;
        }

        Ok(output)
    }
}

impl<D> Op for ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn name(&self) -> Cow<str> {
        "ConvGemm".into()
    }
}

impl<D> StatelessOp for ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let output = self.conv_gemm(&input.to_array_view::<D>()?.into_dimensionality()?)?;
        Ok(tvec!(output.into()))
    }
}

impl<D> InferenceRulesOp for ConvGemm<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D>,
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, D::datum_type())?;
        s.equals(&outputs[0].datum_type, D::datum_type())?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.full_output_shape))?;
        Ok(())
    }
}
