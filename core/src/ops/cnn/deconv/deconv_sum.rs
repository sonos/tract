use crate::internal::*;
use crate::ops::cnn::{KernelFormat, PoolSpec};
use tract_ndarray::prelude::*;

/*
(N) (G) C   H   W
Reshaped Input (N) (G) C   HW
Kernel         (N) (G) OHkWk   C
Gemm           (N) (G) OHkWk   HW              (Gemm: m: OHkWk k:C n:HW)
DeconvSum      (N) (G) O   H'  W'
*/

// f32, ndarray::indices in order

#[derive(Clone, Debug, new, Hash)]
pub struct DeconvSum {
    pub pool_spec: PoolSpec,
    pub kernel_format: KernelFormat,
    /// shape of the deconvolution input
    pub input_shape: TVec<TDim>,
    pub adjustments: TVec<usize>,
    pub bias: Option<Arc<Tensor>>,
    pub group: usize,
}

impl_dyn_hash!(DeconvSum);

impl Op for DeconvSum {
    fn name(&self) -> Cow<str> {
        "DeconvSum".into()
    }

    op_core!();
    op_as_typed_op!();
}

impl EvalOp for DeconvSum {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let gemm = args_1!(inputs).into_tensor();
        debug_assert_eq!(gemm.datum_type(), f32::datum_type());
        let input_shape =
            self.input_shape.iter().map(|i| i.to_usize().unwrap()).collect::<TVec<usize>>();
        let input_shape = self.pool_spec.data_format.shape(&input_shape)?;
        let output_shape =
            super::output_shape(&self.pool_spec, &input_shape.shape, &self.adjustments)?;
        let output_shape = self.pool_spec.data_format.shape(output_shape)?;
        let spatial_output_details = self.pool_spec.padding.compute_for_deconv(
            &input_shape.hw_dims(),
            &self.pool_spec.kernel_shape,
            &self.pool_spec.dilations(),
            &self.pool_spec.strides(),
            &self.adjustments,
        )?;
        let mut tensor = Tensor::zero::<f32>(&*output_shape.shape)?;
        let mut output = tensor.to_array_view_mut::<f32>()?;
        if let Some(b) = &self.bias {
            let mut bias_shape = tvec!(1; output_shape.rank());
            bias_shape[output_shape.c_axis()] = b.len();
            let b = b.clone().into_tensor().into_shape(&bias_shape)?;
            output += &b.to_array_view::<f32>()?;
        }
        let hw = *gemm.shape().last().unwrap();
        let n = *output_shape.n().unwrap_or(&1);
        let n_o_hkwk_hw = gemm.into_shape(&[
            n,
            *output_shape.c(),
            self.pool_spec.kernel_shape.iter().product(),
            hw,
        ])?;
        let n_o_hkwk_hw: ArrayView4<f32> =
            n_o_hkwk_hw.to_array_view::<f32>()?.into_dimensionality()?;
        for n in 0..n {
            for o in 0..*output_shape.c() {
                for (kix, kcoords) in
                    tract_ndarray::indices(&*self.pool_spec.kernel_shape).into_iter().enumerate()
                {
                    for (gix, gcoords) in
                        tract_ndarray::indices(input_shape.hw_dims()).into_iter().enumerate()
                    {
                        // h' = stride * hg + dil * hk
                        let ocoord: TVec<isize> = tract_itertools::izip!(
                            kcoords.slice(),
                            gcoords.slice(),
                            self.pool_spec.strides().as_ref(),
                            self.pool_spec.dilations().as_ref(),
                            &spatial_output_details
                        )
                        .map(|(k, g, s, d, details)| {
                            (k * d + g * s) as isize - details.pad_before as isize
                        })
                        .collect();
                        if ocoord
                            .iter()
                            .zip(output_shape.hw_dims().iter())
                            .all(|(x, dim)| *x >= 0 && (*x as usize) < *dim)
                        {
                            let ocoord = ocoord.iter().map(|x| *x as usize).collect::<TVec<_>>();
                            let ocoord = self.pool_spec.data_format.from_n_c_hw(n, o, ocoord)?;
                            let value = n_o_hkwk_hw[(n, o, kix, gix)];
                            if !value.is_nan() {
                                output[&*ocoord.shape] += value
                            }
                        }
                    }
                }
            }
        }
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl TypedOp for DeconvSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = super::output_shape(&self.pool_spec, &*self.input_shape, &*self.adjustments)?;

        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*shape)))
    }

    as_op!();
}
