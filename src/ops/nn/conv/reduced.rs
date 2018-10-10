use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use super::{ Conv, FixedParamsConv };
use ops::nn::{ DataFormat, PaddingSpec };

#[derive(Debug, Clone)]
pub struct ReducedConv<D:DimLike, T:Datum> {
    data_fmt: DataFormat,
    kernel_is_hwio: bool, // default is oihw (onnx)
    padding: PaddingSpec,
    dilations: Vec<usize>,
    strides: Vec<usize>,
    kernel: ArrayD<T>,

    bias: Option<ArrayD<T>>,
    full_input_shape: Vec<D>,
    full_output_shape: Vec<D>,
}

impl<T: Datum, D:DimLike> ReducedConv<D,T> {
    pub fn new(
        conv: &Conv,
        full_input_shape: &[D],
        full_output_shape: &[D],
        kernel: ArrayD<T>,
        bias: Option<ArrayD<T>>,
    ) -> TfdResult<ReducedConv<D,T>> {
        let spatial_rank = full_input_shape.len() - 2;
        let dilations = conv.dilations.clone().unwrap_or(vec![1; spatial_rank]);
        let strides = conv.strides.clone().unwrap_or(vec![1; spatial_rank]);

        Ok(ReducedConv {
            data_fmt: conv.data_fmt,
            kernel_is_hwio: conv.kernel_is_hwio,
            padding: conv.padding.clone(),
            dilations,
            strides,
            kernel,
            bias,
            full_input_shape: full_input_shape.to_vec(),
            full_output_shape: full_output_shape.to_vec(),
        })
    }
}

impl<D, T> Op for ReducedConv<D, T>
where
    D: DimLike,
    T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + PartialEq,
{
    fn name(&self) -> &str {
        "ReducedConv"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let spatial_rank = self.full_input_shape.len() - 2;
        let kernel_spatial_shape = &self.kernel.shape()[2*(!self.kernel_is_hwio as usize)..][..spatial_rank];

        let input = args_1!(inputs);
        let convoler = FixedParamsConv::new(
            self.data_fmt,
            self.kernel_is_hwio,
            self.dilations.clone(),
            self.strides.clone(),
            kernel_spatial_shape,
            self.padding.clone(),
            input.shape(),
            self.kernel.view(),
            self.bias.as_ref().map(|b| b.view())
        )?;
        let output = convoler.convolve(&input.to_array_view::<T>()?)?;
        Ok(tvec!(output.into()))
    }
}

impl<D, T> InferenceRulesOp for ReducedConv<D,T>
where
    D: DimLike,
    T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T>,
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _solver: &mut Solver<'r>,
        _inputs: &'p TensorsProxy,
        _outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        Ok(())
    }
}
