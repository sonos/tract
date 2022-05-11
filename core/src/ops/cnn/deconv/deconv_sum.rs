use crate::internal::*;
use crate::ops::cnn::padding::ComputedPaddedDim;
use crate::ops::cnn::{KernelFormat, PoolSpec};
use crate::ops::nn::DataShape;
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
        let input_shape = self.pool_spec.data_format.shape(input_shape.clone())?;
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
        let mut tensor = if let Some(b) = &self.bias {
            if output_shape.shape[0..output_shape.c_axis()].iter().all(|d| *d == 1) {
                unsafe {
                    let mut tensor = Tensor::uninitialized::<f32>(&*output_shape.shape)?;
                    let values = b.as_ptr::<f32>()?;
                    let slice = tensor.as_ptr_mut::<f32>()?;
                    let stride = *output_shape.c_stride();
                    for ix in 0..b.len() {
                        let v = *values.offset(ix as isize);
                        for p in 0..stride {
                            *slice.offset((stride * ix + p) as isize) = v;
                        }
                    }
                    tensor
                }
            } else {
                let mut tensor = Tensor::zero::<f32>(&*output_shape.shape)?;
                let mut output = tensor.to_array_view_mut::<f32>()?;
                let mut bias_shape = tvec!(1; output_shape.rank());
                bias_shape[output_shape.c_axis()] = b.len();
                let b = b.clone().into_tensor().into_shape(&bias_shape)?;
                output += &b.to_array_view::<f32>()?;
                tensor
            }
        } else {
            Tensor::zero::<f32>(&*output_shape.shape)?
        };
        let mut output = tensor.to_array_view_mut::<f32>()?;
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
        if !self.pool_spec.data_format.has_n() {
            output = output.insert_axis(Axis(0));
        }
        match input_shape.hw_rank() {
            1 => self.main_loop_1d(
                &input_shape,
                &output_shape,
                &spatial_output_details,
                &n_o_hkwk_hw,
                &mut output.into_dimensionality().unwrap(),
            )?,
            2 => self.main_loop_2d(
                &input_shape,
                &output_shape,
                &spatial_output_details,
                &n_o_hkwk_hw,
                &mut output.into_dimensionality().unwrap(),
            )?,
            3 => self.main_loop_3d(
                &input_shape,
                &output_shape,
                &spatial_output_details,
                &n_o_hkwk_hw,
                &mut output.into_dimensionality().unwrap(),
            )?,
            _ => self.main_loop(
                &input_shape,
                &output_shape,
                &spatial_output_details,
                &n_o_hkwk_hw,
                &mut output.into_dimensionality().unwrap(),
            )?,
        }
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl DeconvSum {
    pub fn main_loop_1d(
        &self,
        input_shape: &DataShape,
        output_shape: &DataShape,
        spatial_output_details: &[ComputedPaddedDim<usize>],
        n_o_hkwk_hw: &ArrayView4<f32>,
        output: &mut ArrayViewMut3<f32>,
    ) -> TractResult<()> {
        let n = *output_shape.n().unwrap_or(&1);
        let kernel_len = self.pool_spec.kernel_shape[0];
        let geo_input_len = input_shape.hw_dims()[0];
        let geo_output_len = output_shape.hw_dims()[0];
        let x_stride = self.pool_spec.strides().as_ref()[0];
        let x_dil = self.pool_spec.dilations().as_ref()[0];
        let x_pad = spatial_output_details[0].pad_before as isize;
        for n in 0..n {
            for o in 0..*output_shape.c() {
                for kx in 0..kernel_len {
                    for gx in 0..geo_input_len {
                        let x = (kx * x_dil + gx * x_stride) as isize - x_pad;
                        if x < 0 || x >= geo_output_len as isize {
                            continue;
                        }
                        let coord = if self.pool_spec.data_format.c_is_last() {
                            [n, x as usize, o]
                        } else {
                            [n, o, x as usize]
                        };
                        unsafe {
                            let value = *n_o_hkwk_hw.uget((n, o, kx, gx));
                            if !value.is_nan() {
                                *output.uget_mut(coord) += value;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn main_loop_2d(
        &self,
        input_shape: &DataShape,
        output_shape: &DataShape,
        spatial_output_details: &[ComputedPaddedDim<usize>],
        n_o_hkwk_hw: &ArrayView4<f32>,
        output: &mut ArrayViewMut4<f32>,
    ) -> TractResult<()> {
        let n = *output_shape.n().unwrap_or(&1);
        let x_stride = self.pool_spec.strides().as_ref()[0];
        let y_stride = self.pool_spec.strides().as_ref()[1];
        let x_dil = self.pool_spec.dilations().as_ref()[0];
        let y_dil = self.pool_spec.dilations().as_ref()[1];
        let x_pad = spatial_output_details[0].pad_before as isize;
        let y_pad = spatial_output_details[1].pad_before as isize;
        let output_c = *output_shape.c();
        let output_c_stride = *output_shape.c_stride() as isize;
        let output_x_stride = output_shape.hw_strides()[0] as isize;
        let output_y_stride = output_shape.hw_strides()[1] as isize;
        let temp_n_stride = n_o_hkwk_hw.strides()[0];
        let temp_o_stride = n_o_hkwk_hw.strides()[1];
        let temp_k_stride = n_o_hkwk_hw.strides()[2];
        let temp_i_stride = n_o_hkwk_hw.strides()[3];
        let ox_len = output_shape.hw_dims()[0];
        let oy_len = output_shape.hw_dims()[1];
        let ix_len = input_shape.hw_dims()[0];
        let iy_len = input_shape.hw_dims()[1];
        let kx_len = self.pool_spec.kernel_shape[0];
        let ky_len = self.pool_spec.kernel_shape[1];
        unsafe {
            for n in 0..n {
                let output = output
                    .as_mut_ptr()
                    .offset((n * *output_shape.n_stride().unwrap_or(&0)) as isize);
                let temp = n_o_hkwk_hw.as_ptr().offset(n as isize * temp_n_stride);
                for kx in 0..kx_len {
                    let temp = temp.offset((kx * ky_len) as isize * temp_k_stride);
                    for ix in 0..ix_len {
                        let ox = (kx * x_dil + ix * x_stride) as isize - x_pad;
                        if ox < 0 || ox >= ox_len as isize {
                            continue;
                        }
                        let temp = temp.offset((ix * iy_len) as isize * temp_i_stride);
                        let output = output.offset(ox * output_x_stride);
                        for ky in 0..ky_len {
                            let temp = temp.offset(ky as isize * temp_k_stride);
                            let oy = (ky * y_dil) as isize - y_pad;
                            for iy in 0..iy_len {
                                let oy = oy + (iy * y_stride) as isize;
                                if oy < 0 || oy >= oy_len as isize {
                                    continue;
                                }
                                let temp = temp.offset(iy as isize * temp_i_stride);
                                let output = output.offset(oy * output_y_stride as isize);
                                Self::main_loop_2d_inner(
                                    output_c,
                                    temp,
                                    temp_o_stride,
                                    output,
                                    output_c_stride,
                                )
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    unsafe fn main_loop_2d_inner(
        output_c: usize,
        temp: *const f32,
        temp_o_stride: isize,
        output: *mut f32,
        output_c_stride: isize,
    ) {
        let mut c = 0;
        let mut right = temp;
        let mut left = output;
        while c + 8 < output_c {
            let mut left0 = *left.offset(0 * output_c_stride);
            let mut left1 = *left.offset(1 * output_c_stride);
            let mut left2 = *left.offset(2 * output_c_stride);
            let mut left3 = *left.offset(3 * output_c_stride);
            let mut left4 = *left.offset(4 * output_c_stride);
            let mut left5 = *left.offset(5 * output_c_stride);
            let mut left6 = *left.offset(6 * output_c_stride);
            let mut left7 = *left.offset(7 * output_c_stride);
            let right0 = *right.offset(0 * temp_o_stride);
            let right1 = *right.offset(1 * temp_o_stride);
            let right2 = *right.offset(2 * temp_o_stride);
            let right3 = *right.offset(3 * temp_o_stride);
            let right4 = *right.offset(4 * temp_o_stride);
            let right5 = *right.offset(5 * temp_o_stride);
            let right6 = *right.offset(6 * temp_o_stride);
            let right7 = *right.offset(7 * temp_o_stride);
            left0 += right0;
            left1 += right1;
            left2 += right2;
            left3 += right3;
            left4 += right4;
            left5 += right5;
            left6 += right6;
            left7 += right7;
            *left.offset(0 * output_c_stride) = left0;
            *left.offset(1 * output_c_stride) = left1;
            *left.offset(2 * output_c_stride) = left2;
            *left.offset(3 * output_c_stride) = left3;
            *left.offset(4 * output_c_stride) = left4;
            *left.offset(5 * output_c_stride) = left5;
            *left.offset(6 * output_c_stride) = left6;
            *left.offset(7 * output_c_stride) = left7;
            c += 8;
            left = left.offset(8 * output_c_stride);
            right = right.offset(8 * temp_o_stride);
        }
        for c in c..output_c {
            let value = *temp.offset(c as isize * temp_o_stride);
            *output.offset(c as isize * output_c_stride) += value;
        }
    }

    pub fn main_loop_3d(
        &self,
        input_shape: &DataShape,
        output_shape: &DataShape,
        spatial_output_details: &[ComputedPaddedDim<usize>],
        n_o_hkwk_hw: &ArrayView4<f32>,
        output: &mut ArrayViewMut5<f32>,
    ) -> TractResult<()> {
        let n = *output_shape.n().unwrap_or(&1);
        let kernel_shape: [usize; 3] = [
            self.pool_spec.kernel_shape[0],
            self.pool_spec.kernel_shape[1],
            self.pool_spec.kernel_shape[2],
        ];
        let geo_input_shape: [usize; 3] =
            [input_shape.hw_dims()[0], input_shape.hw_dims()[1], input_shape.hw_dims()[2]];
        let geo_output_shape: [usize; 3] =
            [output_shape.hw_dims()[0], output_shape.hw_dims()[1], output_shape.hw_dims()[2]];
        let x_stride = self.pool_spec.strides().as_ref()[0];
        let y_stride = self.pool_spec.strides().as_ref()[1];
        let z_stride = self.pool_spec.strides().as_ref()[2];
        let x_dil = self.pool_spec.dilations().as_ref()[0];
        let y_dil = self.pool_spec.dilations().as_ref()[1];
        let z_dil = self.pool_spec.dilations().as_ref()[2];
        let x_pad = spatial_output_details[0].pad_before as isize;
        let y_pad = spatial_output_details[1].pad_before as isize;
        let z_pad = spatial_output_details[2].pad_before as isize;
        for n in 0..n {
            for o in 0..*output_shape.c() {
                for (kix, (kx, ky, kz)) in
                    tract_ndarray::indices(kernel_shape).into_iter().enumerate()
                {
                    for (gix, (gx, gy, gz)) in
                        tract_ndarray::indices(geo_input_shape).into_iter().enumerate()
                    {
                        let x = (kx * x_dil + gx * x_stride) as isize - x_pad;
                        let y = (ky * y_dil + gy * y_stride) as isize - y_pad;
                        let z = (kz * z_dil + gz * z_stride) as isize - z_pad;
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= geo_output_shape[0] as isize
                            || y >= geo_output_shape[1] as isize
                            || z >= geo_output_shape[2] as isize
                        {
                            continue;
                        }
                        let coord = if self.pool_spec.data_format.c_is_last() {
                            [n, x as usize, y as usize, z as usize, o]
                        } else {
                            [n, o, x as usize, y as usize, z as usize]
                        };
                        unsafe {
                            let value = *n_o_hkwk_hw.uget((n, o, kix, gix));
                            if !value.is_nan() {
                                *output.uget_mut(coord) += value;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn main_loop(
        &self,
        input_shape: &DataShape,
        output_shape: &DataShape,
        spatial_output_details: &[ComputedPaddedDim<usize>],
        n_o_hkwk_hw: &ArrayView4<f32>,
        output: &mut ArrayViewMutD<f32>,
    ) -> TractResult<()> {
        let n = *output_shape.n().unwrap_or(&1);
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
                            spatial_output_details
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
                            let ocoord =
                                self.pool_spec.data_format.with_n().from_n_c_hw(n, o, ocoord)?;
                            let value = n_o_hkwk_hw[(n, o, kix, gix)];
                            if !value.is_nan() {
                                output[&*ocoord.shape] += value
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl TypedOp for DeconvSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = super::output_shape(&self.pool_spec, &*self.input_shape, &*self.adjustments)?;
        Ok(tvec!(inputs[0].datum_type.fact(&*shape)))
    }

    as_op!();
}
