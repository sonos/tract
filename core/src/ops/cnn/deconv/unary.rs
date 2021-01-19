use crate::internal::*;
use crate::ops::cnn::{KernelFormat, PaddingSpec};
use crate::ops::nn::DataFormat;

// NCHW OIHW rank=4 valid, no-stride, no-dil, no-bias, no-group, f32

#[derive(Clone, Debug, new, Hash)]
pub struct DeconvUnary {
    pub data_format: DataFormat,
    pub kernel_format: KernelFormat,
    pub padding: PaddingSpec,
    pub kernel: Arc<Tensor>,
}

impl DeconvUnary {
    pub fn output_shape<D: DimLike>(&self, x_shape: &[D]) -> TractResult<TVec<D>> {
        let x_shape = self.data_format.shape(x_shape)?;
        let spatial_input_shape = x_shape.hw_dims();
        let spatial_kernel_shape = match self.kernel_format {
            KernelFormat::OIHW => &self.kernel.shape()[2..],
            KernelFormat::HWIO => &self.kernel.shape()[..self.kernel.rank() - 2],
        };
        let ones = tvec!(1; spatial_input_shape.len());
        let spatial_output_shape = self.padding.compute_for_deconv(
            &spatial_input_shape,
            &spatial_kernel_shape,
            &ones,
            &ones,
        );
        let deconv_shape: TVec<D> =
            spatial_output_shape.iter().map(|comp| comp.deconvoluted.clone()).collect();
        let output_shape = self.data_format.from_n_c_hw(
            x_shape.n().cloned().unwrap_or(1.into()),
            self.kernel.shape()[1].into(), // FIXME Kernel format
            deconv_shape,
        )?;
        Ok(output_shape.shape.into())
    }
}

impl_dyn_hash!(DeconvUnary);

impl Op for DeconvUnary {
    fn name(&self) -> Cow<str> {
        "DeconvUnary".into()
    }
    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for DeconvUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let output_shape = self.output_shape(input.shape())?;
        let mut tensor = Tensor::zero_dt(input.datum_type(), &output_shape)?;
        /*
        let input = input.to_array_view::<f32>()?.into_dimensionality()?;
        let mut output = tensor.to_array_view_mut::<f32>()?.into_dimensionality()?;
        let kernel = self.kernel.to_array_view::<f32>()?.into_dimensionality()?;
        for n in 0..input.shape()[0] {
            for co in 0..output.shape()[1] {
                for ci in 0..input.shape()[1] {
                    for hi in 0..input.shape()[2] {
                        for wi in 0..input.shape()[3] {
                            for hk in 0..self.kernel.shape()[2] {
                                for wk in 0..self.kernel.shape()[3] {
                                    output[(n, co, hi + hk, wi + wk)] +=
                                        input[(n, ci, hi, wi)] * kernel[(ci, co, hk, wk)];
                                }
                            }
                        }
                    }
                }
            }
        }
        */
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl TypedOp for DeconvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x_fact = inputs[0];
        let output_shape = self.output_shape(&*x_fact.shape)?;
        Ok(tvec!(TypedFact::dt_shape(x_fact.datum_type, &output_shape)))
    }

    as_op!();
}
