#![allow(dead_code)]

use std::ops::AddAssign;

use crate::internal::*;
use crate::ops::cnn::padding::ComputedPaddedDim;
use crate::ops::cnn::{KernelFormat, PoolSpec};
use crate::ops::nn::DataShape;
use tract_ndarray::prelude::*;
use tract_num_traits::Float;

/*
(N) (G) C   H   W
Reshaped Input (N) (G) C   HW
Kernel         (N) (G) OHkWk   C
Gemm           (N) (G) OHkWk   HW              (Gemm: m: OHkWk k:C n:HW)
DeconvSum      (N) (G) O   H'  W'
*/

// f32, ndarray::indices in order

#[derive(Clone, Debug, new, Hash, PartialEq, Eq)]
pub struct DeconvSum {
    pub pool_spec: PoolSpec,
    pub kernel_format: KernelFormat,
    /// shape of the deconvolution input
    pub input_shape: ShapeFact,
    pub adjustments: TVec<usize>,
    pub group: usize,
}

impl Op for DeconvSum {
    fn name(&self) -> StaticName {
        "DeconvSum".into()
    }

    op_as_typed_op!();
}

impl EvalOp for DeconvSum {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.eval_with_values(inputs, ctx.symbols)
    }
}

impl DeconvSum {
    fn eval_with_values(
        &self,
        inputs: TVec<TValue>,
        values: &SymbolValues,
    ) -> TractResult<TVec<TValue>> {
        let (gemm, bias) = args_2!(inputs);
        let input_shape = self.input_shape.eval_to_usize(values)?.into_owned();
        let input_shape = self.pool_spec.data_format.shape(input_shape)?;
        let output_shape =
            super::output_shape(&self.pool_spec, &input_shape.shape, &self.adjustments)?;
        let output_shape = self.pool_spec.data_format.shape(output_shape)?;
        let spatial_output_details = self.pool_spec.padding.compute_for_deconv(
            input_shape.hw_dims(),
            &self.pool_spec.kernel_shape,
            &self.pool_spec.dilations(),
            &self.pool_spec.strides(),
            &self.adjustments,
        )?;
        let mut tensor = bias.into_tensor();
        let hw = *gemm.shape().last().unwrap();
        let n = *output_shape.n().unwrap_or(&1);
        let n_o_hkwk_hw = gemm.into_tensor().into_shape(&[
            n,
            *output_shape.c(),
            self.pool_spec.kernel_shape.iter().product(),
            hw,
        ])?;
        if !self.pool_spec.data_format.has_n() {
            tensor.insert_axis(0)?;
        }
        if !try_fast_nchw_2x2_s2_f32(
            self,
            &input_shape,
            &output_shape,
            &spatial_output_details,
            &n_o_hkwk_hw,
            &mut tensor,
        )? {
            eval(
                self,
                &input_shape,
                &output_shape,
                &spatial_output_details,
                &n_o_hkwk_hw,
                &mut tensor,
            )?;
        }
        if !self.pool_spec.data_format.has_n() {
            tensor.remove_axis(0)?;
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for DeconvSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2);
        let shape = super::output_shape(&self.pool_spec, &self.input_shape, &self.adjustments)?;
        ensure!(*inputs[1].shape == *shape);
        Ok(tvec!(inputs[0].datum_type.fact(shape)))
    }

    fn set_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        target.wire_node(
            &node.name,
            Self { input_shape: self.input_shape.substitute(subs)?.into_owned(), ..self.clone() },
            &[mapping[&node.inputs[0]], mapping[&node.inputs[1]]],
        )
    }

    as_op!();
}

/// NCHW 2×2 stride-2 unpack: write even/odd W with `vld2`/`vst2` instead of
/// the generic loop's channel-inner scatter (C stride is `H*W` on NCHW).
fn try_fast_nchw_2x2_s2_f32(
    op: &DeconvSum,
    input_shape: &DataShape,
    output_shape: &DataShape,
    spatial_output_details: &[ComputedPaddedDim<usize>],
    gemm: &Tensor,
    output: &mut Tensor,
) -> TractResult<bool> {
    if output.datum_type() != f32::datum_type() {
        return Ok(false);
    }
    if op.pool_spec.data_format != crate::ops::nn::DataFormat::NCHW {
        return Ok(false);
    }
    if op.pool_spec.kernel_shape[..] != [2, 2] {
        return Ok(false);
    }
    if op.pool_spec.strides()[..] != [2, 2] || op.pool_spec.dilations()[..] != [1, 1] {
        return Ok(false);
    }
    if spatial_output_details.len() != 2
        || spatial_output_details[0].pad_before != 0
        || spatial_output_details[1].pad_before != 0
    {
        return Ok(false);
    }
    if *output_shape.w_stride() != 1 {
        return Ok(false);
    }
    let ih = input_shape.hw_dims()[0];
    let iw = input_shape.hw_dims()[1];
    let oh = output_shape.hw_dims()[0];
    let ow = output_shape.hw_dims()[1];
    if oh != ih * 2 || ow != iw * 2 {
        return Ok(false);
    }
    unsafe {
        deconv_nchw_2x2_s2_f32(gemm, output, input_shape, output_shape, ih, iw);
    }
    Ok(true)
}

unsafe fn deconv_nchw_2x2_s2_f32(
    gemm: &Tensor,
    output: &mut Tensor,
    input_shape: &DataShape,
    output_shape: &DataShape,
    ih: usize,
    iw: usize,
) {
    unsafe {
        let gptr = gemm.as_ptr::<f32>().expect("f32 gemm");
        let optr = output.as_ptr_mut::<f32>().expect("f32 out");
        let n = *output_shape.n().unwrap_or(&1);
        let oc = *output_shape.c();
        let g_n = gemm.strides()[0];
        let g_o = gemm.strides()[1];
        let g_k = gemm.strides()[2];
        let g_i = gemm.strides()[3];
        let o_n = *output_shape.n_stride().unwrap_or(&0) as isize;
        let o_c = *output_shape.c_stride() as isize;
        let o_h = *output_shape.h_stride() as isize;
        for ni in 0..n as isize {
            for o in 0..oc as isize {
                let g = gptr.offset(ni * g_n + o * g_o);
                let dst = optr.offset(ni * o_n + o * o_c);
                let src00 = g;
                let src01 = g.offset(g_k);
                let src10 = g.offset(2 * g_k);
                let src11 = g.offset(3 * g_k);
                for ix in 0..ih {
                    let in_row = (ix * iw) as isize * g_i;
                    interleave_add_row(
                        dst.offset((2 * ix) as isize * o_h),
                        src00.offset(in_row),
                        src01.offset(in_row),
                        iw,
                        g_i,
                    );
                    interleave_add_row(
                        dst.offset((2 * ix + 1) as isize * o_h),
                        src10.offset(in_row),
                        src11.offset(in_row),
                        iw,
                        g_i,
                    );
                }
            }
        }
        let _ = input_shape;
    }
}

unsafe fn interleave_add_row(
    dst: *mut f32,
    even: *const f32,
    odd: *const f32,
    iw: usize,
    src_stride: isize,
) {
    unsafe {
        let mut i = 0usize;
        if src_stride == 1 {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                while i + 4 <= iw {
                    let e = vld1q_f32(even.add(i));
                    let o = vld1q_f32(odd.add(i));
                    let mut d = vld2q_f32(dst.add(2 * i));
                    d.0 = vaddq_f32(d.0, e);
                    d.1 = vaddq_f32(d.1, o);
                    vst2q_f32(dst.add(2 * i), d);
                    i += 4;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                // scalar tail covers x86; the add is stride-2 stores
            }
        }
        while i < iw {
            let di = dst.add(2 * i);
            *di += *even.offset(i as isize * src_stride);
            *di.add(1) += *odd.offset(i as isize * src_stride);
            i += 1;
        }
    }
}

fn eval(
    op: &DeconvSum,
    input_shape: &DataShape,
    output_shape: &DataShape,
    spatial_output_details: &[ComputedPaddedDim<usize>],
    n_o_hkwk_hw: &Tensor,
    output: &mut Tensor,
) -> TractResult<()> {
    let dt = output.datum_type();
    unsafe {
        #[cfg(target_arch = "aarch64")]
        if dt == f16::datum_type() && tract_linalg::arm64::has_fp16() {
            return eval_t_aarch64fp16::<f16>(
                op,
                input_shape,
                output_shape,
                spatial_output_details,
                n_o_hkwk_hw,
                output,
                |a, b| tract_linalg::arm64::add_f16(a, b),
            );
        }
        dispatch_floatlike!(eval_t_generic(dt)(
            op,
            input_shape,
            output_shape,
            spatial_output_details,
            n_o_hkwk_hw,
            output,
            |a, b| a + b
        ))
    }
}

macro_rules! impl_eval {
        ($(#[$meta: meta])* $suffix: ident) => {
            pastey::paste! {
                $(#[$meta])*
                    unsafe fn [<eval_t_ $suffix>]<T: Datum + Float + Copy + AddAssign<T>>(
                        op: &DeconvSum,
                        input_shape: &DataShape,
                        output_shape: &DataShape,
                        spatial_output_details: &[ComputedPaddedDim<usize>],
                        n_o_hkwk_hw: &Tensor,
                        output: &mut Tensor,
                        add: impl Fn(T, T) -> T + Copy + 'static,
                        ) -> TractResult<()> {
                        let mut output_plain = output.try_as_plain_mut()?;
                        let output = output_plain.to_array_view_mut::<T>()?;
                        let n_o_hkwk_hw: ArrayView4<T> = n_o_hkwk_hw.to_plain_array_view::<T>()?.into_dimensionality()?;
                        match input_shape.hw_rank() {
                            1 => [<main_loop_1d_ $suffix>](
                                op,
                                input_shape,
                                output_shape,
                                spatial_output_details,
                                &n_o_hkwk_hw,
                                &mut output.into_dimensionality().unwrap(),
                                add,
                                ),
                            2 => [<main_loop_2d_ $suffix>](
                                op,
                                input_shape,
                                output_shape,
                                spatial_output_details,
                                &n_o_hkwk_hw,
                                &mut output.into_dimensionality().unwrap(),
                                add,
                                ),
                            3 => [<main_loop_3d_ $suffix>](
                                op,
                                input_shape,
                                output_shape,
                                spatial_output_details,
                                &n_o_hkwk_hw,
                                &mut output.into_dimensionality().unwrap(),
                                add,
                                ),
                            _ => [<main_loop_ $suffix>](
                                op,
                                input_shape,
                                output_shape,
                                spatial_output_details,
                                &n_o_hkwk_hw,
                                &mut output.into_dimensionality().unwrap(),
                                add,
                                ),
                        }
                    }

                pub fn [<main_loop_1d_ $suffix>]<T: Datum + Float>(
                    op: &DeconvSum,
                    input_shape: &DataShape,
                    output_shape: &DataShape,
                    spatial_output_details: &[ComputedPaddedDim<usize>],
                    n_o_hkwk_hw: &ArrayView4<T>,
                    output: &mut ArrayViewMut3<T>,
                    add: impl Fn(T, T) -> T + Copy + 'static,
                    ) -> TractResult<()> {
                    let n = *output_shape.n().unwrap_or(&1);
                    let kernel_len = op.pool_spec.kernel_shape[0];
                    let geo_input_len = input_shape.hw_dims()[0];
                    let geo_output_len = output_shape.hw_dims()[0];
                    let x_stride = op.pool_spec.strides().as_ref()[0];
                    let x_dil = op.pool_spec.dilations().as_ref()[0];
                    let x_pad = spatial_output_details[0].pad_before as isize;
                    for n in 0..n {
                        for o in 0..*output_shape.c() {
                            for kx in 0..kernel_len {
                                for gx in 0..geo_input_len {
                                    let x = (kx * x_dil + gx * x_stride) as isize - x_pad;
                                    if x < 0 || x >= geo_output_len as isize {
                                        continue;
                                    }
                                    let coord = if op.pool_spec.data_format.c_is_last() {
                                        [n, x as usize, o]
                                    } else {
                                        [n, o, x as usize]
                                    };
                                    unsafe {
                                        let value = *n_o_hkwk_hw.uget((n, o, kx, gx));
                                        *output.uget_mut(coord) = add(*output.uget(coord), value);
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                }

                pub fn [<main_loop_2d_ $suffix>]<T: Datum + Float>(
                    op: &DeconvSum,
                    input_shape: &DataShape,
                    output_shape: &DataShape,
                    spatial_output_details: &[ComputedPaddedDim<usize>],
                    n_o_hkwk_hw: &ArrayView4<T>,
                    output: &mut ArrayViewMut4<T>,
                    add: impl Fn(T, T) -> T + Copy + 'static,
                    ) -> TractResult<()> {
                    let n = *output_shape.n().unwrap_or(&1);
                    let x_stride = op.pool_spec.strides().as_ref()[0];
                    let y_stride = op.pool_spec.strides().as_ref()[1];
                    let x_dil = op.pool_spec.dilations().as_ref()[0];
                    let y_dil = op.pool_spec.dilations().as_ref()[1];
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
                    let kx_len = op.pool_spec.kernel_shape[0];
                    let ky_len = op.pool_spec.kernel_shape[1];
                    unsafe {
                        for n in 0..n {
                            let output = output.as_mut_ptr().add(n * *output_shape.n_stride().unwrap_or(&0));
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
                                            let output = output.offset(oy * output_y_stride);
                                            [<main_loop_2d_inner_ $suffix>](
                                                output_c,
                                                temp,
                                                temp_o_stride,
                                                output,
                                                output_c_stride,
                                                add,
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
                #[allow(clippy::erasing_op)]
                #[allow(clippy::identity_op)]
                unsafe fn [<main_loop_2d_inner_ $suffix>]<T: Datum + Float>(
                    output_c: usize,
                    temp: *const T,
                    temp_o_stride: isize,
                    output: *mut T,
                    output_c_stride: isize,
                    add: impl Fn(T, T) -> T + Copy + 'static,
                    ) { unsafe {
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
                        left0 = add(left0, right0);
                        left1 = add(left1, right1);
                        left2 = add(left2, right2);
                        left3 = add(left3, right3);
                        left4 = add(left4, right4);
                        left5 = add(left5, right5);
                        left6 = add(left6, right6);
                        left7 = add(left7, right7);
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
                        let ptr = output.offset(c as isize * output_c_stride);
                        *ptr = add(*ptr, value);
                    }
                }}

                pub fn [<main_loop_3d_ $suffix>]<T: Datum + Float>(
                    op: &DeconvSum,
                    input_shape: &DataShape,
                    output_shape: &DataShape,
                    spatial_output_details: &[ComputedPaddedDim<usize>],
                    n_o_hkwk_hw: &ArrayView4<T>,
                    output: &mut ArrayViewMut5<T>,
                    add: impl Fn(T, T) -> T + Copy + 'static,
                    ) -> TractResult<()> {
                    let n = *output_shape.n().unwrap_or(&1);
                    let kernel_shape: [usize; 3] =
                        [op.pool_spec.kernel_shape[0], op.pool_spec.kernel_shape[1], op.pool_spec.kernel_shape[2]];
                    let geo_input_shape: [usize; 3] =
                        [input_shape.hw_dims()[0], input_shape.hw_dims()[1], input_shape.hw_dims()[2]];
                    let geo_output_shape: [usize; 3] =
                        [output_shape.hw_dims()[0], output_shape.hw_dims()[1], output_shape.hw_dims()[2]];
                    let x_stride = op.pool_spec.strides().as_ref()[0];
                    let y_stride = op.pool_spec.strides().as_ref()[1];
                    let z_stride = op.pool_spec.strides().as_ref()[2];
                    let x_dil = op.pool_spec.dilations().as_ref()[0];
                    let y_dil = op.pool_spec.dilations().as_ref()[1];
                    let z_dil = op.pool_spec.dilations().as_ref()[2];
                    let x_pad = spatial_output_details[0].pad_before as isize;
                    let y_pad = spatial_output_details[1].pad_before as isize;
                    let z_pad = spatial_output_details[2].pad_before as isize;
                    for n in 0..n {
                        for o in 0..*output_shape.c() {
                            for (kix, (kx, ky, kz)) in tract_ndarray::indices(kernel_shape).into_iter().enumerate()
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
                                        let coord = if op.pool_spec.data_format.c_is_last() {
                                            [n, x as usize, y as usize, z as usize, o]
                                        } else {
                                            [n, o, x as usize, y as usize, z as usize]
                                        };
                                        unsafe {
                                            let value = *n_o_hkwk_hw.uget((n, o, kix, gix));
                                            *output.uget_mut(coord) = add(*output.uget(coord), value);
                                        }
                                    }
                            }
                        }
                    }
                    Ok(())
                }

                pub fn [<main_loop_ $suffix>]<T: Datum + Float>(
                    op: &DeconvSum,
                    input_shape: &DataShape,
                    output_shape: &DataShape,
                    spatial_output_details: &[ComputedPaddedDim<usize>],
                    n_o_hkwk_hw: &ArrayView4<T>,
                    output: &mut ArrayViewMutD<T>,
                    add: impl Fn(T, T) -> T + Copy + 'static,
                    ) -> TractResult<()> {
                    let n = *output_shape.n().unwrap_or(&1);
                    let strides = op.pool_spec.strides();
                    let dilations = op.pool_spec.dilations();
                    for n in 0..n {
                        for o in 0..*output_shape.c() {
                            for (kix, kcoords) in
                                tract_ndarray::indices(&*op.pool_spec.kernel_shape).into_iter().enumerate()
                                {
                                    for (gix, gcoords) in
                                        tract_ndarray::indices(input_shape.hw_dims()).into_iter().enumerate()
                                        {
                                            // h' = stride * hg + dil * hk
                                            let ocoord: TVec<isize> = tract_itertools::izip!(
                                                kcoords.slice(),
                                                gcoords.slice(),
                                                strides.as_ref(),
                                                dilations.as_ref(),
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
                                                        let ocoord = op.pool_spec.data_format.with_n().from_n_c_hw(n, o, ocoord)?;
                                                        let value = n_o_hkwk_hw[(n, o, kix, gix)];
                                                        output[&*ocoord.shape] = add(output[&*ocoord.shape], value)
                                                    }
                                        }
                                }
                        }
                    }
                    Ok(())
                }
            }
        }
    }

impl_eval!(generic);
impl_eval! {
#[target_feature(enable = "fp16")]
#[cfg(target_arch = "aarch64")]
        aarch64fp16
    }

crate::declare_knob!(
    TRACT_DISABLE_DEPTHWISE_DECONV,
    bool,
    false,
    "Disable the fused depthwise-deconvolution path, falling back to einsum + DeconvSum."
);

/// Fused depthwise deconvolution: `kernel * input` accumulated straight into the
/// output, instead of `EinSum` materialising `[N, C, HkWk, HW]` for `DeconvSum` to
/// scatter.
///
/// For a depthwise ConvTranspose the einsum's contraction dim is 1, so it degenerates
/// to an outer product whose result is larger than either operand — and most of it is
/// discarded: GTCRN materialises `[1, 16, 9, 363]` (3267 products per channel) to
/// produce `[1, 16, 1, 33]` (33 per channel). Roughly 91% of those products fall
/// outside the output and are thrown away. Folding the multiply inside the bounds test
/// skips them and never allocates the intermediate.
///
/// Inputs: `kernel` `[C, HkWk, 1]`, `input` `[N, C, 1, HW]`, `bias` (output-shaped).
#[derive(Clone, Debug, new, Hash, PartialEq, Eq)]
pub struct DepthwiseDeconv {
    pub pool_spec: PoolSpec,
    pub kernel_format: KernelFormat,
    pub input_shape: ShapeFact,
    pub adjustments: TVec<usize>,
    pub group: usize,
}

impl Op for DepthwiseDeconv {
    fn name(&self) -> StaticName {
        "DepthwiseDeconv".into()
    }
    op_as_typed_op!();
}

impl EvalOp for DepthwiseDeconv {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (kernel, input, bias) = args_3!(inputs);
        let input_shape = self.input_shape.eval_to_usize(ctx.symbols)?.into_owned();
        let input_shape = self.pool_spec.data_format.shape(input_shape)?;
        let output_shape =
            super::output_shape(&self.pool_spec, &input_shape.shape, &self.adjustments)?;
        let output_shape = self.pool_spec.data_format.shape(output_shape)?;
        let spatial_output_details = self.pool_spec.padding.compute_for_deconv(
            input_shape.hw_dims(),
            &self.pool_spec.kernel_shape,
            &self.pool_spec.dilations(),
            &self.pool_spec.strides(),
            &self.adjustments,
        )?;
        let mut tensor = bias.into_tensor();
        if !self.pool_spec.data_format.has_n() {
            tensor.insert_axis(0)?;
        }
        dispatch_floatlike!(Self::eval_t(tensor.datum_type())(
            self,
            &input_shape,
            &output_shape,
            &spatial_output_details,
            &kernel,
            &input,
            &mut tensor
        ))?;
        if !self.pool_spec.data_format.has_n() {
            tensor.remove_axis(0)?;
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl DepthwiseDeconv {
    /// Rank-2 fused path, mirroring `main_loop_2d`'s pointer walk: the geometry
    /// bounds are tested once per (kx, ix, ky, iy) as there, and the innermost loop
    /// runs over channels, where kernel and input are both contiguous.
    #[allow(clippy::too_many_arguments)]
    unsafe fn eval_t_2d<T: Datum + Float + Copy + AddAssign<T>>(
        &self,
        input_shape: &DataShape,
        output_shape: &DataShape,
        spatial_output_details: &[ComputedPaddedDim<usize>],
        kernel: &[T],
        input: &[T],
        output: &mut Tensor,
    ) -> TractResult<()> {
        let n_batch = *output_shape.n().unwrap_or(&1);
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
        let ox_len = output_shape.hw_dims()[0];
        let oy_len = output_shape.hw_dims()[1];
        let ix_len = input_shape.hw_dims()[0];
        let iy_len = input_shape.hw_dims()[1];
        let kx_len = self.pool_spec.kernel_shape[0];
        let ky_len = self.pool_spec.kernel_shape[1];
        let kvol = kx_len * ky_len;
        let ihw = ix_len * iy_len;
        let mut output_view = output.to_plain_array_view_mut::<T>()?;
        let out_base = output_view.as_mut_ptr();
        unsafe {
            for n in 0..n_batch {
                let out_n = out_base.add(n * *output_shape.n_stride().unwrap_or(&0));
                let in_n = input.as_ptr().add(n * output_c * ihw);
                for kx in 0..kx_len {
                    for ix in 0..ix_len {
                        let ox = (kx * x_dil + ix * x_stride) as isize - x_pad;
                        if ox < 0 || ox >= ox_len as isize {
                            continue;
                        }
                        let out_x = out_n.offset(ox * output_x_stride);
                        for ky in 0..ky_len {
                            let kix = kx * ky_len + ky;
                            let oy0 = (ky * y_dil) as isize - y_pad;
                            // Channels outer, the contiguous axis inner. The output's
                            // channel stride is the whole spatial plane (12.8 KB on
                            // DeepFilterNet3's [1, 64, 100, 32]), so a channel-innermost
                            // loop walks past L1 on every step; oy is stride-1.
                            for c in 0..output_c {
                                let kv = *kernel.get_unchecked(c * kvol + kix);
                                let in_c = in_n.add(c * ihw + ix * iy_len);
                                let out_c = out_x.offset(c as isize * output_c_stride);
                                for iy in 0..iy_len {
                                    let oy = oy0 + (iy * y_stride) as isize;
                                    if oy < 0 || oy >= oy_len as isize {
                                        continue;
                                    }
                                    *out_c.offset(oy * output_y_stride) += kv * *in_c.add(iy);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn eval_t<T: Datum + Float + Copy + AddAssign<T>>(
        &self,
        input_shape: &DataShape,
        output_shape: &DataShape,
        spatial_output_details: &[ComputedPaddedDim<usize>],
        kernel: &TValue,
        input: &TValue,
        output: &mut Tensor,
    ) -> TractResult<()> {
        let c = *output_shape.c();
        let hw: usize = input_shape.hw_dims().iter().product();
        let kernel_vol: usize = self.pool_spec.kernel_shape.iter().product();
        let kernel = kernel.to_plain_array_view::<T>()?;
        let kernel = kernel.as_slice().context("depthwise deconv kernel must be contiguous")?;
        let input = input.to_plain_array_view::<T>()?;
        let input = input.as_slice().context("depthwise deconv input must be contiguous")?;
        ensure!(kernel.len() == c * kernel_vol);
        let n = *output_shape.n().unwrap_or(&1);
        ensure!(input.len() == n * c * hw);
        // The wiring only selects this op at rank 2 (see wire_with_deconv_sum); the
        // generic loop below stays as a reference implementation for the tests.
        if input_shape.hw_rank() == 2 {
            return unsafe {
                self.eval_t_2d::<T>(
                    input_shape,
                    output_shape,
                    spatial_output_details,
                    kernel,
                    input,
                    output,
                )
            };
        }
        let strides = self.pool_spec.strides();
        let dilations = self.pool_spec.dilations();
        let mut output_plain = output.try_as_plain_mut()?;
        let mut output = output_plain.to_array_view_mut::<T>()?;
        for n in 0..n {
            for o in 0..c {
                for (kix, kcoords) in
                    tract_ndarray::indices(&*self.pool_spec.kernel_shape).into_iter().enumerate()
                {
                    // Hoisted out of the geometry loop: one load per (channel, tap).
                    let kv = kernel[o * kernel_vol + kix];
                    for (gix, gcoords) in
                        tract_ndarray::indices(input_shape.hw_dims()).into_iter().enumerate()
                    {
                        let ocoord: TVec<isize> = tract_itertools::izip!(
                            kcoords.slice(),
                            gcoords.slice(),
                            strides.as_ref(),
                            dilations.as_ref(),
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
                            // The multiply lives here, inside the bounds test, so the
                            // ~91% of products that fall outside are never computed.
                            let value = kv * input[(n * c + o) * hw + gix];
                            let ocoord = ocoord.iter().map(|x| *x as usize).collect::<TVec<_>>();
                            let ocoord =
                                self.pool_spec.data_format.with_n().from_n_c_hw(n, o, ocoord)?;
                            output[&*ocoord.shape] += value;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl TypedOp for DepthwiseDeconv {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3);
        let shape = super::output_shape(&self.pool_spec, &self.input_shape, &self.adjustments)?;
        ensure!(*inputs[2].shape == *shape);
        Ok(tvec!(inputs[1].datum_type.fact(shape)))
    }

    fn set_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        target.wire_node(
            &node.name,
            Self { input_shape: self.input_shape.substitute(subs)?.into_owned(), ..self.clone() },
            &[mapping[&node.inputs[0]], mapping[&node.inputs[1]], mapping[&node.inputs[2]]],
        )
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cnn::PaddingSpec;
    use crate::ops::nn::DataFormat;

    #[test]
    fn nchw_2x2_s2_matches_generic() {
        let n = 1usize;
        let oc = 4usize;
        let ih = 5usize;
        let iw = 6usize;
        let hw = ih * iw;
        let gemm: Vec<f32> = (0..n * oc * 4 * hw).map(|i| (i as f32 * 0.07).sin()).collect();
        let bias: Vec<f32> =
            (0..n * oc * ih * 2 * iw * 2).map(|i| (i as f32 * 0.01) - 0.2).collect();
        let gemm_t = Tensor::from_shape(&[n, oc, 4, hw], &gemm).unwrap();
        let mut out_fast = Tensor::from_shape(&[n, oc, ih * 2, iw * 2], &bias).unwrap();
        let mut out_gen = out_fast.clone();
        let pool_spec = PoolSpec::new(
            DataFormat::NCHW,
            tvec![2, 2],
            PaddingSpec::Valid,
            Some(tvec![1, 1]),
            Some(tvec![2, 2]),
            oc,
            oc,
        );
        let op = DeconvSum {
            pool_spec,
            kernel_format: KernelFormat::OIHW,
            input_shape: [n.to_dim(), oc.to_dim(), ih.to_dim(), iw.to_dim()].into(),
            adjustments: tvec!(0, 0),
            group: 1,
        };
        let ish = DataFormat::NCHW.shape(tvec![n, oc, ih, iw]).unwrap();
        let osh = DataFormat::NCHW.shape(tvec![n, oc, ih * 2, iw * 2]).unwrap();
        let spatial = op
            .pool_spec
            .padding
            .compute_for_deconv(
                ish.hw_dims(),
                &op.pool_spec.kernel_shape,
                &op.pool_spec.dilations(),
                &op.pool_spec.strides(),
                &op.adjustments,
            )
            .unwrap();
        assert!(
            try_fast_nchw_2x2_s2_f32(&op, &ish, &osh, &spatial, &gemm_t, &mut out_fast).unwrap()
        );
        eval(&op, &ish, &osh, &spatial, &gemm_t, &mut out_gen).unwrap();
        let a = out_fast.to_plain_array_view::<f32>().unwrap();
        let b = out_gen.to_plain_array_view::<f32>().unwrap();
        let mut max_abs = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        assert!(max_abs < 1e-5, "deconv 2x2 s=2 mismatch max_abs={max_abs}");
    }
}
