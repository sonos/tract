use crate::internal::*;
use crate::ops::cnn::PaddingSpec;

// NCHW OIHW rank=4 valid, no-stride, no-dil, no-bias, no-group, f32

#[derive(Clone, Debug, new, Hash)]
pub struct DeconvUnary {
    pub padding: PaddingSpec,
    pub kernel: Arc<Tensor>,
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
        let output_shape = tvec!(
            input.shape()[0],
            self.kernel.shape()[1],
            input.shape()[2] + self.kernel.shape()[2] - 1,
            input.shape()[3] + self.kernel.shape()[3] - 1,
        );
        let mut tensor = Tensor::zero_dt(input.datum_type(), &output_shape)?;
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
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl TypedOp for DeconvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x_fact = inputs[0];
        let spatial_input_shape = &x_fact.shape[2..];
        let spatial_kernel_shape = &self.kernel.shape()[2..];
        let spatial_output_shape = self.padding.compute_for_deconv(
            &spatial_input_shape,
            &spatial_kernel_shape,
            &[1, 1],
            &[1, 1],
        );
        let output_shape = tvec!(
            x_fact.shape[0].clone(),
            self.kernel.shape()[1].to_dim(),
            spatial_output_shape[0].deconvoluted.clone(),
            spatial_output_shape[1].deconvoluted.clone(),
        );
        Ok(tvec!(TypedFact::dt_shape(x_fact.datum_type, &output_shape)))
    }

    as_op!();
}
