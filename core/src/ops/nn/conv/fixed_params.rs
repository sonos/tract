use ndarray::prelude::*;
use ops::prelude::*;

use ops::nn::{DataFormat, PaddingSpec, Patch};
use super::im2col::Im2Col;

use insideout::InsideOut;

#[derive(Debug, Clone)]
pub struct FixedParamsConv<D: Datum> {
    im2col: Im2Col<D>,
    kernel_is_hwio: bool,
    patch: Patch,
    kernel: Array2<D>,
    full_output_shape: TVec<usize>,
    k: usize,
    m: usize,
    n: usize,
    bias: Option<ArrayD<D>>,
    group: usize,
}

impl<D: Datum> FixedParamsConv<D> {
    pub fn new(
        data_fmt: DataFormat,
        kernel_is_hwio: bool,
        dilations: TVec<usize>,
        strides: TVec<usize>,
        padding: PaddingSpec,
        input_full_shape: &[usize],
        kernel: ArrayViewD<D>,
        bias: Option<ArrayViewD<D>>,
        group: usize,
    ) -> TractResult<FixedParamsConv<D>> {
        let output_channels = if kernel_is_hwio {
            *kernel.shape().last().unwrap()
        } else {
            kernel.shape()[0]
        };

        let kernel_spatial_shape = if kernel_is_hwio {
            &kernel.shape()[..(input_full_shape.len() - 2)]
        } else {
            &kernel.shape()[2..]
        };

        let patch = Patch::new(
            data_fmt,
            dilations,
            kernel_spatial_shape.into(),
            &padding,
            strides,
            input_full_shape.into(),
        );

        let shape: TVec<usize> = patch.output_full_shape(output_channels);

        let k = kernel.len() / output_channels;
        let m = output_channels;
        let n = patch.output_spatial_shape.iter().product();
        let kernel = kernel.to_shared();

        let kernel: Array2<D> = if kernel_is_hwio {
            let mut permutation: Vec<usize> = vec![kernel.ndim() - 1, kernel.ndim() - 2];
            permutation.extend(0..(kernel.ndim() - 2));
            let permuted = kernel.permuted_axes(permutation);
            Array2::<D>::from_shape_vec((m, k), permuted.iter().cloned().collect::<Vec<_>>())?
        } else {
            kernel.into_shape((m, k))?.to_owned()
        };

        let bias = bias
            .map(|bias| -> TractResult<_> {
                let mut bias_shape: Vec<usize> = ::std::iter::repeat(1).take(shape.len()).collect();
                bias_shape[1] = output_channels;
                Ok(bias.view().into_shape(&*bias_shape)?.to_owned())
            })
            .inside_out()?;

        let im2col = Im2Col::new(patch.clone(), m, k, n, group);

        Ok(FixedParamsConv {
            im2col,
            kernel_is_hwio,
            patch,
            kernel,
            full_output_shape: shape,
            k,
            m,
            n,
            bias,
            group,
        })
    }
}

impl<D> FixedParamsConv<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub(super) fn convolve<'i>(&'i self, input: &'i ArrayViewD<'i, D>) -> TractResult<ArrayD<D>> {
        let mega_matrix = self.im2col.im2col(input)?;
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

impl<D> Op for FixedParamsConv<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn name(&self) -> Cow<str> {
        "FixedParamsConv".into()
    }
}

impl<D> StatelessOp for FixedParamsConv<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let output = self.convolve(&inputs[0].to_array_view::<D>()?)?;
        Ok(tvec!(output.into()))
    }
}

impl<D> InferenceRulesOp for FixedParamsConv<D>
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
        s.equals(
            &inputs[0].shape,
            ShapeFact::from(&*self.patch.input_shape.shape),
        )?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.full_output_shape))?;
        Ok(())
    }
}
