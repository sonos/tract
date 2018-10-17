use analyser::rules::prelude::*;
use insideout::InsideOut;
use ops::prelude::*;

use super::{Conv, FixedParamsConv};
use ops::nn::{DataFormat, PaddingSpec};

#[derive(Debug, Clone)]
pub struct ConvUnary {
    pub data_fmt: DataFormat,
    pub kernel_is_hwio: bool, // default is oihw (onnx)
    pub padding: PaddingSpec,
    pub dilations: Vec<usize>,
    pub strides: Vec<usize>,
    pub kernel: Tensor,

    pub bias: Option<Tensor>,
    pub full_input_shape: Vec<TDim>,
    pub full_output_shape: Vec<TDim>,
    pub group: usize,
}

impl ConvUnary {
    pub fn new(
        conv: &Conv,
        full_input_shape: &[TDim],
        full_output_shape: &[TDim],
        kernel: Tensor,
        bias: Option<Tensor>,
        group: usize,
    ) -> TfdResult<ConvUnary> {
        let spatial_rank = full_input_shape.len() - 2;
        let dilations = conv.dilations.clone().unwrap_or(vec![1; spatial_rank]);
        let strides = conv.strides.clone().unwrap_or(vec![1; spatial_rank]);

        Ok(ConvUnary {
            data_fmt: conv.data_fmt,
            kernel_is_hwio: conv.kernel_is_hwio,
            padding: conv.padding.clone(),
            dilations,
            strides,
            kernel,
            bias,
            full_input_shape: full_input_shape.to_vec(),
            full_output_shape: full_output_shape.to_vec(),
            group,
        })
    }

    fn eval_t<T>(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + ::std::ops::AddAssign<T> + PartialEq,
    {
        let spatial_rank = self.full_input_shape.len() - 2;
        let kernel_spatial_shape =
            &self.kernel.shape()[2 * (!self.kernel_is_hwio as usize)..][..spatial_rank];

        let input = args_1!(inputs);
        let convoler = FixedParamsConv::new(
            self.data_fmt,
            self.kernel_is_hwio,
            self.dilations.clone(),
            self.strides.clone(),
            kernel_spatial_shape,
            self.padding.clone(),
            input.shape(),
            self.kernel.to_array_view::<T>()?,
            self.bias
                .as_ref()
                .map(|b| b.to_array_view::<T>())
                .inside_out()?,
            self.group,
        )?;
        let output = convoler.convolve(&input.to_array_view::<T>()?)?;
        Ok(tvec!(output.into()))
    }

    pub fn rm_dummy_axis(&self, axis: usize) -> TfdResult<Option<ConvUnary>> {
        let shape = self.data_fmt.shape(&self.full_input_shape);
        if axis < shape.h_axis() {
            return Ok(None);
        }
        let geo_axis = axis - shape.h_axis();
        if geo_axis >= shape.hw_rank() {
            return Ok(None);
        }
        if self.dilations[geo_axis] != 1
            || self.strides[geo_axis] != 1
            || !self.padding.valid_dim(geo_axis)
        {
            return Ok(None);
        }
        let kernel_spatial_shape =
            &self.kernel.shape()[2 * (!self.kernel_is_hwio as usize)..][..shape.hw_rank()];
        if kernel_spatial_shape[geo_axis] != 1 {
            return Ok(None);
        }
        fn copy_rm_nth<D: DimLike>(input: &[D], nth: usize) -> Vec<D> {
            input
                .iter()
                .enumerate()
                .filter(|&(ax, _)| ax != nth)
                .map(|(_, &d)| d)
                .collect()
        }
        let kernel_shape: Vec<usize> = copy_rm_nth(
            self.kernel.shape().clone(),
            geo_axis + 2 * (!self.kernel_is_hwio as usize),
        );
        let kernel = self.kernel.clone().into_shape(&kernel_shape)?;
        let new_op = ConvUnary {
            data_fmt: self.data_fmt,
            kernel_is_hwio: self.kernel_is_hwio,
            padding: self.padding.rm_axis(geo_axis),
            dilations: copy_rm_nth(&self.dilations, geo_axis),
            strides: copy_rm_nth(&self.strides, geo_axis),
            kernel,
            bias: self.bias.clone(),
            full_input_shape: copy_rm_nth(&self.full_input_shape, axis),
            full_output_shape: copy_rm_nth(&self.full_output_shape, axis),
            group: self.group,
        };
        Ok(Some(new_op))
    }
}

impl Op for ConvUnary {
    fn name(&self) -> &str {
        "ConvUnary"
    }

    fn reduce(
        &self,
        _inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
    ) -> TfdResult<Option<ReducedOpRewire>> {
        let spatial_rank = self.full_input_shape.len() - 2;
        let kernel_spatial_shape =
            &self.kernel.shape()[2 * (!self.kernel_is_hwio as usize)..][..spatial_rank];
        if kernel_spatial_shape.iter().product::<usize>() == 1
            && self.dilations.iter().all(|&x| x == 1)
            && self.strides.iter().all(|&x| x == 1)
            && self.group == 1
            && self.bias.is_none()
            && (0..spatial_rank).all(|ax| self.padding.valid_dim(ax))
        {
            if self.kernel_is_hwio && self.data_fmt == DataFormat::NHWC {
                use ops::math::mat_mul::MatMulUnaryA;
                let kernel_shape = &self.kernel.shape()[spatial_rank..];
                let kernel = self.kernel.clone().into_shape(&kernel_shape)?;
                return Ok(Some(ReducedOpRewire::new(
                    Box::new(MatMulUnaryA::new(kernel)),
                    tvec!(0),
                )));
            }
        }
        Ok(None)
    }
}

impl StatelessOp for ConvUnary {
    fn eval(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}


impl InferenceRulesOp for ConvUnary {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, self.full_input_shape.clone())?;
        s.equals(&outputs[0].shape, self.full_output_shape.clone())?;
        Ok(())
    }
}
