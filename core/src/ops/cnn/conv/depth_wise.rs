use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::cnn::patches::{Zone, ZoneScanner};
use crate::ops::nn::DataShape;
use num_traits::Zero;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct DepthWise {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
}

impl Op for DepthWise {
    fn name(&self) -> StaticName {
        "DepthWiseConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.patch)])
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for DepthWise {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let dt = inputs[0].datum_type();
        #[cfg(target_arch = "aarch64")]
        if dt == f16::datum_type() && tract_linalg::arm64::has_fp16() {
            return unsafe {
                eval_t_aarch64fp16::<f16>(
                    self,
                    inputs,
                    |a, b| tract_linalg::arm64::add_f16(a, b),
                    |a, b| tract_linalg::arm64::mul_f16(a, b),
                )
            };
        }
        dispatch_floatlike!(Self::eval_gen(dt)(self, inputs))
    }
}

impl DepthWise {
    fn eval_gen<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar>(
        &self,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        unsafe { eval_t_generic::<T>(self, inputs, |a, b| a + b, |a, b| a * b) }
    }
}

impl TypedOp for DepthWise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 3);
        anyhow::ensure!(
            self.input_shape.c() == self.output_shape.c(),
            "DepthWiseConv must have same input and output channels"
        );
        anyhow::ensure!(
            self.input_shape.c().to_dim() == inputs[2].shape.volume(),
            "DepthWiseConv data has {} channels, bias has {}",
            self.input_shape.c(),
            inputs[2].shape.len()
        );
        Ok(tvec!(inputs[0].datum_type.fact(&self.output_shape.shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let [_input, kernel, _bias] = inputs else {
            bail!("Depthwise expects three inputs");
        };
        let n_output_points = self.patch.output_shape.iter().cloned().product::<usize>();
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            kernel.shape.volume() * self.input_shape.n().unwrap_or(&1) * n_output_points
        )))
    }

    as_op!();
}

macro_rules! impl_eval {
    ($(#[$meta: meta])* $suffix: ident ) => {
        pastey::paste! {
            $(#[$meta])*
            unsafe fn [<eval_t_ $suffix>]<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar>(
                dw: &DepthWise,
                inputs: TVec<TValue>,
                add: impl Fn(T, T) -> T + Copy + 'static,
                mul: impl Fn(T, T) -> T + Copy + 'static,
            ) -> TractResult<TVec<TValue>> {
                let (img, kernel, bias) = args_3!(inputs);
                let mut output = unsafe { Tensor::uninitialized::<T>(&dw.output_shape.shape)? };
                let iptr = img.as_ptr::<T>()?;
                let optr = output.as_ptr_mut::<T>()?;
                let k_stride_i = kernel.strides()[1];
                let n = *dw.input_shape.n().unwrap_or(&1);
                let n_stride_i = *dw.input_shape.n_stride().unwrap_or(&0) as isize;
                let n_stride_o = *dw.output_shape.n_stride().unwrap_or(&0) as isize;
                let c_stride_i = *dw.input_shape.c_stride() as isize;
                let c_stride_o = *dw.output_shape.c_stride() as isize;
                let bias = bias.as_ptr::<T>()?;
                let kptr = kernel.as_ptr::<T>()?;
                unsafe {
                    for n in 0..n as isize {
                        let iptr = iptr.offset(n_stride_i * n);
                        let optr = optr.offset(n_stride_o * n);
                        for zone in &dw.patch.zones {
                            [<process_zone_ $suffix>](
                                dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                                add, mul,
                            )
                        }
                    }
                }
                Ok(tvec!(output.into_tvalue()))
            }

            #[inline(never)]
            #[allow(clippy::too_many_arguments)]
            $(#[$meta])*
            unsafe fn [<process_zone_ $suffix>]<T: Datum + Copy + Zero>(
                dw: &DepthWise,
                zone: &Zone,
                c_stride_i: isize,
                c_stride_o: isize,
                k_stride_i: isize,
                iptr: *const T,
                kptr: *const T,
                bias: *const T,
                optr: *mut T,
                add: impl Fn(T, T) -> T + Copy + 'static,
                mul: impl Fn(T, T) -> T + Copy + 'static,
                ) { unsafe {
                /*
                   if zone.values_offsets.len() == 2 {
                   self.process_zone_n::<T, 2, 4>(
                   zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                   )
                   } else if zone.values_offsets.len() == 3 {
                   dw.process_zone_n::<T, 3, 4>(
                   zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                   )
                   } else */
                match zone.values_offsets.len() {
                    1 => [<process_zone_n_ $suffix>]::<T, 1, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul,
                    ),
                    2 => [<process_zone_n_ $suffix>]::<T, 2, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul,
                    ),
                    3 => [<process_zone_n_ $suffix>]::<T, 3, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul,
                    ),
                    4 => [<process_zone_n_ $suffix>]::<T, 4, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul,
                    ),
                    _ => zone.visit_output(&dw.patch, |visitor| {
                        for c in 0..*dw.input_shape.c() as isize {
                            let iptr = iptr.offset(c_stride_i * c);
                            let optr = optr.offset(c_stride_o * c);
                            let kptr = kptr.offset(k_stride_i * c);
                            [<inner_loop_ $suffix>]::<T>(iptr, kptr, bias, optr, c, visitor, add, mul)
                        }
                    }),
                }
            }}

            #[inline(never)]
            #[allow(clippy::too_many_arguments)]
            $(#[$meta])*
            unsafe fn [<process_zone_n_ $suffix>]<T: Datum + Copy + Zero, const N: usize, const UNROLL: usize>(
                dw: &DepthWise,
                zone: &Zone,
                c_stride_i: isize,
                c_stride_o: isize,
                k_stride_i: isize,
                iptr: *const T,
                kptr: *const T,
                bias: *const T,
                optr: *mut T,
                add: impl Fn(T, T) -> T,
                mul: impl Fn(T, T) -> T,
                ) { unsafe {
                let mut visitor = ZoneScanner::new(zone, &dw.patch);
                let mut ioffset = [0isize; N];
                for i in 0..N {
                    ioffset[i] = zone.values_offsets[i].1;
                }
                let mut k = [T::zero(); N];
                for c in 0..*dw.input_shape.c() as isize {
                    visitor.reset();
                    let iptr = iptr.offset(c_stride_i * c);
                    let optr = optr.offset(c_stride_o * c);
                    for n in 0..N {
                        k[n] = *kptr.offset(k_stride_i * c).add(zone.values_offsets[n].0);
                    }
                    let bias = *bias.offset(c);
                    while !visitor.done {
                        let iptr = iptr.offset(visitor.input_center_offset);
                        let optr = optr.offset(visitor.output_offset);
                        #[cfg(target_arch = "aarch64")]
                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                            && visitor.inner_loop_output_stride == 1
                            && visitor.inner_loop_input_full_stride >= 1
                        {
                            let k_f32 = *(&k as *const [T; N] as *const [f32; N]);
                            let bias_f32 = *(&bias as *const T as *const f32);
                            neon_depthwise_w_f32::<N>(
                                iptr as *const f32,
                                optr as *mut f32,
                                &k_f32,
                                &ioffset,
                                bias_f32,
                                visitor.inner_loop_len,
                                visitor.inner_loop_input_full_stride,
                            );
                            visitor.next_non_inner_axis();
                            continue;
                        }
                        let mut i = 0isize;
                        while i + (UNROLL as isize) < visitor.inner_loop_len as isize {
                            let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
                            let optr = optr.offset(visitor.inner_loop_output_stride * i);
                            let mut iptrs = [std::ptr::null(); UNROLL];
                            for u in 0..UNROLL {
                                iptrs[u] = iptr.offset(visitor.inner_loop_input_full_stride * u as isize);
                            }
                            let mut optrs = [std::ptr::null_mut(); UNROLL];
                            for u in 0..UNROLL {
                                optrs[u] = optr.offset(visitor.inner_loop_output_stride * u as isize);
                            }
                            let mut is = [[T::zero(); N]; UNROLL];
                            for u in 0..UNROLL {
                                for n in 0..N {
                                    is[u][n] = *iptrs[u].offset(ioffset[n]);
                                }
                            }
                            let mut ps = [[T::zero(); N]; UNROLL];
                            for u in 0..UNROLL {
                                for n in 0..N {
                                    ps[u][n] = mul(is[u][n], k[n]);
                                }
                            }
                            for u in 0..UNROLL {
                                let mut sum = bias;
                                for n in 0..N {
                                    sum = add(sum, ps[u][n]);
                                }
                                *optrs[u] = sum;
                            }
                            i += UNROLL as isize;
                        }
                        while i < visitor.inner_loop_len as isize {
                            let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
                            let optr = optr.offset(visitor.inner_loop_output_stride * i);
                            let mut is = [T::zero(); N];
                            for n in 0..N {
                                is[n] = *iptr.offset(ioffset[n]);
                            }
                            let mut p = [T::zero(); N];
                            for n in 0..N {
                                p[n] = mul(is[n], k[n]);
                            }
                            let mut sum = bias;
                            for n in 0..N {
                                sum = add(sum, p[n]);
                            }
                            *optr = sum;
                            i += 1;
                        }
                        visitor.next_non_inner_axis()
                    }
                }
            }}

            #[inline(never)]
            #[allow(clippy::too_many_arguments)]
            $(#[$meta])*
            unsafe fn [<inner_loop_ $suffix>]<T: Datum + Copy>(
                iptr: *const T,
                kptr: *const T,
                bias: *const T,
                optr: *mut T,
                c: isize,
                visitor: &ZoneScanner,
                add: impl Fn(T, T) -> T,
                mul: impl Fn(T, T) -> T,
                ) { unsafe {
                let mut sum = *bias.offset(c);
                let mut iter = visitor.valid_offsets_ker_in();
                if iter.size_hint() == (3, Some(3)) {
                    let (ix, v) = iter.next().unwrap();
                    let k0 = *kptr.add(ix);
                    let i0 = *iptr.offset(v);
                    let (ix, v) = iter.next().unwrap();
                    let k1 = *kptr.add(ix);
                    let i1 = *iptr.offset(v);
                    let (ix, v) = iter.next().unwrap();
                    let k2 = *kptr.add(ix);
                    let i2 = *iptr.offset(v);
                    sum = add(add(add(sum, mul(k0, i0)), mul(k1, i1)), mul(k2, i2));
                } else {
                    for (ix, v) in iter {
                        let k = *kptr.add(ix);
                        let i = *iptr.offset(v);
                        sum = add(sum, mul(k, i));
                    }
                }
                let optr = optr.offset(visitor.output_offset);
                *optr = sum;
            }}
        }
    }
}

impl_eval!(generic);
impl_eval! {
#[target_feature(enable = "fp16")]
#[cfg(target_arch = "aarch64")]
aarch64fp16
}

/// Depthwise along an output-contiguous spatial axis (`output_stride == 1`).
/// Vectorise over consecutive output points. Input may be contiguous (stride 1)
/// or strided: DPDFNet 48 kHz encoder DW is NCHW `kw=3` with W-stride 2 or 3,
/// which `vld2q`/`vld3q` de-interleave. Scalar `process_zone_n` stays for
/// padded / non-unit output-stride zones. Does not touch `BlockedConv`.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_depthwise_w_f32<const N: usize>(
    iptr: *const f32,
    optr: *mut f32,
    k: &[f32; N],
    ioffset: &[isize; N],
    bias: f32,
    len: usize,
    in_stride: isize,
) {
    unsafe {
        use std::arch::aarch64::*;
        let biasv = vdupq_n_f32(bias);
        let mut i = 0usize;
        if in_stride == 1 {
            while i + 8 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let p = iptr.offset(ioffset[n]).add(i);
                    acc0 = vfmaq_f32(acc0, vld1q_f32(p), kn);
                    acc1 = vfmaq_f32(acc1, vld1q_f32(p.add(4)), kn);
                }
                vst1q_f32(optr.add(i), acc0);
                vst1q_f32(optr.add(i + 4), acc1);
                i += 8;
            }
            while i + 4 <= len {
                let mut acc = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    acc = vfmaq_f32(acc, vld1q_f32(iptr.offset(ioffset[n]).add(i)), kn);
                }
                vst1q_f32(optr.add(i), acc);
                i += 4;
            }
        } else if in_stride == 2 {
            while i + 8 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let p = iptr.offset(ioffset[n]).offset(i as isize * 2);
                    // vld2 de-interleaves even/odd; even lanes are stride-2 samples.
                    let a = vld2q_f32(p);
                    let b = vld2q_f32(p.add(8));
                    acc0 = vfmaq_f32(acc0, a.0, kn);
                    acc1 = vfmaq_f32(acc1, b.0, kn);
                }
                vst1q_f32(optr.add(i), acc0);
                vst1q_f32(optr.add(i + 4), acc1);
                i += 8;
            }
            while i + 4 <= len {
                let mut acc = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let a = vld2q_f32(iptr.offset(ioffset[n]).offset(i as isize * 2));
                    acc = vfmaq_f32(acc, a.0, kn);
                }
                vst1q_f32(optr.add(i), acc);
                i += 4;
            }
        } else if in_stride == 3 {
            while i + 8 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let p = iptr.offset(ioffset[n]).offset(i as isize * 3);
                    let a = vld3q_f32(p);
                    let b = vld3q_f32(p.add(12));
                    acc0 = vfmaq_f32(acc0, a.0, kn);
                    acc1 = vfmaq_f32(acc1, b.0, kn);
                }
                vst1q_f32(optr.add(i), acc0);
                vst1q_f32(optr.add(i + 4), acc1);
                i += 8;
            }
            while i + 4 <= len {
                let mut acc = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let a = vld3q_f32(iptr.offset(ioffset[n]).offset(i as isize * 3));
                    acc = vfmaq_f32(acc, a.0, kn);
                }
                vst1q_f32(optr.add(i), acc);
                i += 4;
            }
        }
        while i < len {
            let mut sum = bias;
            for n in 0..N {
                sum += k[n] * *iptr.offset(ioffset[n]).offset(i as isize * in_stride);
            }
            *optr.add(i) = sum;
            i += 1;
        }
    }
}
//#[target_feature(enable = "fp16")] impl_eval!(aarch64fp16);

/* partial alternative impl that may be relevant when simd gets better */

/*
#[inline(never)]
unsafe fn process_zone_4_f32(
&self,
zone: &Zone,
c_stride_i: isize,
c_stride_o: isize,
k_stride_i: isize,
iptr: *const f32,
kptr: *const f32,
bias: *const f32,
optr: *mut f32,
) {
use std::simd::*;
let mut visitor = ZoneScanner::new(zone, &self.patch);
let ioffset0 = zone.values_offsets[0].1;
let ioffset1 = zone.values_offsets[1].1;
let ioffset2 = zone.values_offsets[2].1;
let ioffset3 = zone.values_offsets[3].1;
for c in 0..*self.input_shape.c() as isize {
visitor.reset();
let kptr = kptr.offset(k_stride_i * c);
let iptr = iptr.offset(c_stride_i * c);
let optr = optr.offset(c_stride_o * c);
let k0 = *kptr.offset(zone.values_offsets[0].0 as isize);
let k1 = *kptr.offset(zone.values_offsets[1].0 as isize);
let k2 = *kptr.offset(zone.values_offsets[2].0 as isize);
let k3 = *kptr.offset(zone.values_offsets[3].0 as isize);
let k0 = f32x4::splat(k0);
let k1 = f32x4::splat(k1);
let k2 = f32x4::splat(k2);
let k3 = f32x4::splat(k3);
let bias = f32x4::splat(*bias.offset(c));
while !visitor.done {
let iptr = iptr.offset(visitor.input_center_offset);
let optr = optr.offset(visitor.output_offset);
let mut i  = 0;
while i + 4 <
for i in 0..visitor.inner_loop_len as isize {
let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
let optr = optr.offset(visitor.inner_loop_output_stride * i);
let i0 = *iptr.offset(ioffset0);
let i1 = *iptr.offset(ioffset1);
let i2 = *iptr.offset(ioffset2);
let i3 = *iptr.offset(ioffset3);
let i = f32x4::from_array([i0, i1, i2, i3]);
let p = (i * k).reduce_sum();
let sum = bias + p;
     *optr = sum
     }
     visitor.next_non_inner_axis()
     }
     }
     }
     */

/*
#[inline(never)]
unsafe fn process_zone_4_f32(
&self,
zone: &Zone,
c_stride_i: isize,
c_stride_o: isize,
k_stride_i: isize,
iptr: *const f32,
kptr: *const f32,
bias: *const f32,
optr: *mut f32,
) {
use std::simd::*;
let mut visitor = ZoneScanner::new(zone, &self.patch);
let ioffset0 = zone.values_offsets[0].1;
let ioffset1 = zone.values_offsets[1].1;
let ioffset2 = zone.values_offsets[2].1;
let ioffset3 = zone.values_offsets[3].1;
for c in 0..*self.input_shape.c() as isize {
visitor.reset();
let kptr = kptr.offset(k_stride_i * c);
let iptr = iptr.offset(c_stride_i * c);
let optr = optr.offset(c_stride_o * c);
let k0 = *kptr.offset(zone.values_offsets[0].0 as isize);
let k1 = *kptr.offset(zone.values_offsets[1].0 as isize);
let k2 = *kptr.offset(zone.values_offsets[2].0 as isize);
let k3 = *kptr.offset(zone.values_offsets[3].0 as isize);
let k = f32x4::from_array([k0, k1, k2, k3]);
let bias = *bias.offset(c);
while !visitor.done {
let iptr = iptr.offset(visitor.input_center_offset);
let optr = optr.offset(visitor.output_offset);
for i in 0..visitor.inner_loop_len as isize {
let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
let optr = optr.offset(visitor.inner_loop_output_stride * i);
let i0 = *iptr.offset(ioffset0);
let i1 = *iptr.offset(ioffset1);
let i2 = *iptr.offset(ioffset2);
let i3 = *iptr.offset(ioffset3);
let i = f32x4::from_array([i0, i1, i2, i3]);
let p = (i * k).reduce_sum();
let sum = bias + p;
     *optr = sum
     }
     visitor.next_non_inner_axis()
     }
     }
     }
     */

/*
#[inline(never)]
unsafe fn process_zone_4<T: Datum + Copy + ndarray::LinalgScalar>(
&self,
zone: &Zone,
c_stride_i: isize,
c_stride_o: isize,
k_stride_i: isize,
iptr: *const T,
kptr: *const T,
bias: *const T,
optr: *mut T,
) {
let mut visitor = ZoneScanner::new(zone, &self.patch);
let ioffset0 = zone.values_offsets[0].1;
let ioffset1 = zone.values_offsets[1].1;
let ioffset2 = zone.values_offsets[2].1;
let ioffset3 = zone.values_offsets[3].1;
for c in 0..*self.input_shape.c() as isize {
visitor.reset();
let kptr = kptr.offset(k_stride_i * c);
let iptr = iptr.offset(c_stride_i * c);
let optr = optr.offset(c_stride_o * c);
let k0 = *kptr.offset(zone.values_offsets[0].0 as isize);
let k1 = *kptr.offset(zone.values_offsets[1].0 as isize);
let k2 = *kptr.offset(zone.values_offsets[2].0 as isize);
let k3 = *kptr.offset(zone.values_offsets[3].0 as isize);
let bias = *bias.offset(c);
while !visitor.done {
let iptr = iptr.offset(visitor.input_center_offset);
let optr = optr.offset(visitor.output_offset);
for i in 0..visitor.inner_loop_len as isize {
let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
let optr = optr.offset(visitor.inner_loop_output_stride * i);
let i0 = *iptr.offset(ioffset0);
let i1 = *iptr.offset(ioffset1);
let i2 = *iptr.offset(ioffset2);
let i3 = *iptr.offset(ioffset3);
let p0 = i0 * k0;
let p1 = i1 * k1;
let p2 = i2 * k2;
let p3 = i3 * k3;
let sum = bias + p0 + p1 + p2 + p3;
     *optr = sum
     }
     visitor.next_non_inner_axis()
     }
     }
     }
     */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cnn::conv::{Conv, KernelFormat};
    use crate::ops::cnn::{PaddingSpec, PoolSpec};
    use crate::ops::nn::DataFormat;

    fn run_dw(
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        pad: PaddingSpec,
        stride: (usize, usize),
    ) {
        let n = 1usize;
        let x: Vec<f32> = (0..n * c * h * w).map(|i| ((i as f32 * 0.137).sin()) * 0.7).collect();
        let kernel: Vec<f32> = (0..c * kh * kw).map(|i| ((i as f32 * 0.091).cos()) * 0.3).collect();
        let bias: Vec<f32> = (0..c).map(|i| (i as f32 * 0.05) - 0.1).collect();

        let mut model = TypedModel::default();
        let xv = model.add_source("x", f32::fact([n, c, h, w])).unwrap();
        let kv =
            model.add_const("k", Tensor::from_shape(&[c, 1, kh, kw], &kernel).unwrap()).unwrap();
        let bv = model.add_const("b", Tensor::from_shape(&[c], &bias).unwrap()).unwrap();
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(kh, kw),
                padding: pad.clone(),
                dilations: None,
                strides: Some(tvec!(stride.0, stride.1)),
                input_channels: c,
                output_channels: c,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: c,
            q_params: None,
        };
        let out = model.wire_node("dw", conv, &[xv, kv, bv]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let model = model.into_decluttered().unwrap().into_optimized().unwrap();
        assert!(
            model.nodes.iter().any(|node| node.op_as::<DepthWise>().is_some()),
            "expected DepthWiseConv, got {}",
            model.nodes.iter().map(|node| node.op().name()).collect::<Vec<_>>().join(",")
        );
        let runnable = model.into_runnable().unwrap();
        let got = runnable
            .run(tvec![Tensor::from_shape(&[n, c, h, w], &x).unwrap().into_tvalue()])
            .unwrap();
        let got = got[0].to_plain_array_view::<f32>().unwrap();
        let oshape = got.shape();
        let oh = oshape[2];
        let ow = oshape[3];
        let (ph, pw) = match pad {
            PaddingSpec::Valid => (0isize, 0isize),
            _ => (((kh - 1) / 2) as isize, ((kw - 1) / 2) as isize),
        };
        let mut max_abs = 0f32;
        for oc in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias[oc];
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = oy as isize * stride.0 as isize + ky as isize - ph;
                            let ix = ox as isize * stride.1 as isize + kx as isize - pw;
                            if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                continue;
                            }
                            let xv = x[((oc * h + iy as usize) * w) + ix as usize];
                            let kv = kernel[((oc * kh + ky) * kw) + kx];
                            acc += xv * kv;
                        }
                    }
                    let g = got[[0, oc, oy, ox]];
                    max_abs = max_abs.max((g - acc).abs());
                }
            }
        }
        assert!(
            max_abs < 1e-5,
            "DepthWise mismatch c={c} {h}x{w} k={kh}x{kw} stride={stride:?} pad={pad:?}: max_abs={max_abs}"
        );
    }

    #[test]
    fn depthwise_contig_w_matches_reference() {
        // 48 kHz-like: NCHW, H=1, long W, kw=3. Inner loop is W.
        run_dw(16, 1, 64, 1, 3, PaddingSpec::Valid, (1, 1));
        run_dw(32, 1, 481, 1, 3, PaddingSpec::Valid, (1, 1));
        run_dw(8, 1, 17, 1, 3, PaddingSpec::SameUpper, (1, 1));
        run_dw(8, 12, 20, 3, 1, PaddingSpec::Valid, (1, 1));
        run_dw(4, 9, 9, 3, 3, PaddingSpec::Valid, (1, 1));
        // Encoder DW: stride 2 / 3 along W (vld2 / vld3 path).
        run_dw(64, 1, 481, 1, 3, PaddingSpec::SameUpper, (1, 3));
        run_dw(64, 1, 161, 1, 3, PaddingSpec::SameUpper, (1, 2));
        run_dw(16, 1, 64, 1, 3, PaddingSpec::Valid, (1, 2));
    }
}
