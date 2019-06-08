use ndarray::prelude::*;
use internal::*;

use ops::nn::{DataFormat, PaddingSpec, Patch};
use ops::nn::conv::KernelFormat;
use super::im2col::Im2Col;
use super::conv_gemm::ConvGemm;

use insideout::InsideOut;

#[derive(Debug, Clone)]
pub struct FixedParamsConv<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    im2col: Im2Col<D>,
    conv_gemm: ConvGemm<D>,
}

impl<D: Datum> FixedParamsConv<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub fn new(
        data_fmt: DataFormat,
        kernel_fmt: KernelFormat,
        dilations: TVec<usize>,
        strides: TVec<usize>,
        padding: PaddingSpec,
        input_full_shape: &[usize],
        kernel: ArrayViewD<D>,
        bias: Option<ArrayViewD<D>>,
        group: usize,
    ) -> TractResult<FixedParamsConv<D>> {
        let output_channels = match kernel_fmt {
            KernelFormat::HWIO => *kernel.shape().last().unwrap(),
            KernelFormat::OIHW => kernel.shape()[0],
        };

        let kernel_spatial_shape = &kernel.shape()[kernel_fmt.h_axis()..][..(input_full_shape.len() - 2)];

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
        let conv_gemm = ConvGemm::new(patch, shape, m, k, n, kernel_is_hwio, kernel, bias, group);

        Ok(FixedParamsConv {
            im2col,
            conv_gemm
        })
    }
}

impl<D> FixedParamsConv<D>
where
    D: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<D> + PartialEq,
{
    pub(super) fn convolve<'i>(&'i self, input: &'i ArrayViewD<'i, D>) -> TractResult<ArrayD<D>> {
        let mega_matrix = self.im2col.im2col(input)?;
        self.conv_gemm.conv_gemm(&mega_matrix.view())
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
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, D::datum_type())?;
        s.equals(&outputs[0].datum_type, D::datum_type())?;
        s.equals(
            &inputs[0].shape,
            ShapeFact::from(&*self.im2col.patch.input_shape.shape),
        )?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.conv_gemm.full_output_shape))?;
        Ok(())
    }
}
