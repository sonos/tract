use crate::internal::*;
use crate::ops::cnn::{KernelFormat, PaddingSpec};
use crate::ops::nn::DataFormat;
use tract_ndarray::prelude::*;

/*
(N) C   H   W
Reshaped Input (N) C   HW
Kernel         (N) OHkWk   C
Gemm           (N) OHkWk   HW              (Gemm: m: OHkWk k:C n:HW)
DeconvSum      (N) O   H'  W'
*/

// f32, ndarray::indices in order, tride == 1, dilation == 1

#[derive(Clone, Debug, new, Hash)]
pub struct DeconvSum {
    pub data_format: DataFormat,
    pub kernel_format: KernelFormat,
    pub padding: PaddingSpec,
    pub kernel_shape: TVec<usize>,
    /// shape of the deconvolution input
    pub input_shape: TVec<TDim>,
    pub strides: TVec<usize>,
    pub dilations: TVec<usize>,
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
        let input_shape = self.data_format.shape(&input_shape)?;
        let output_shape = super::output_shape(
            &self.data_format,
            &self.kernel_format,
            &self.padding,
            &*self.kernel_shape,
            input_shape.shape,
            &self.strides,
            &self.dilations,
        )?;
        dbg!(&self);
        let output_shape = self.data_format.shape(output_shape)?;
        dbg!(&output_shape);
        let kernel_spatial_shape = self.kernel_format.spatial_shape(&self.kernel_shape);
        let spatial_output_details = self.padding.compute_for_deconv(
            &input_shape.hw_dims(),
            &kernel_spatial_shape,
            &self.dilations,
            &self.strides,
        );

        let mut tensor = Tensor::zero::<f32>(&*output_shape.shape)?;
        let mut output = tensor.to_array_view_mut::<f32>()?;
        let hw = *gemm.shape().last().unwrap();
        let n = *output_shape.n().unwrap_or(&1);
        let n_o_hkwk_hw = gemm.into_shape(dbg!(&[
            n,
            *output_shape.c(),
            kernel_spatial_shape.iter().product(),
            hw
        ]))?;
        let n_o_hkwk_hw: ArrayView4<f32> =
            n_o_hkwk_hw.to_array_view::<f32>()?.into_dimensionality()?;
        for n in 0..n {
            for o in 0..*output_shape.c() {
                for (kix, kcoords) in
                    tract_ndarray::indices(kernel_spatial_shape).into_iter().enumerate()
                {
                    for (gix, gcoords) in
                        tract_ndarray::indices(input_shape.hw_dims()).into_iter().enumerate()
                    {
                        // h' = stride * hg + dil * hk
                        let ocoord: TVec<isize> = tract_itertools::izip!(
                            kcoords.slice(),
                            gcoords.slice(),
                            &self.strides,
                            &self.dilations,
                            &spatial_output_details
                        )
                        .map(|(k, g, s, d, details)| (k * d + g * s) as isize - details.pad_before as isize)
                        .collect();
                        if ocoord
                            .iter()
                            .zip(output_shape.hw_dims().iter())
                            .all(|(x, dim)| *x >= 0 && (*x as usize) < *dim)
                        {
                            let ocoord = ocoord.iter().map(|x| *x as usize).collect::<TVec<_>>();
                            let ocoord = self.data_format.from_n_c_hw(n, o, ocoord)?;
                            output[&*ocoord.shape] += n_o_hkwk_hw[(n, o, kix, gix)];
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
        let shape = super::output_shape(
            &self.data_format,
            &self.kernel_format,
            &self.padding,
            &*self.kernel_shape,
            &*self.input_shape,
            &*self.strides,
            &*self.dilations,
        )?;

        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*shape)))
    }

    as_op!();
}
