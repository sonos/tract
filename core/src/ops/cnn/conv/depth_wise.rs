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
    relu: bool,
    scale: bool,
}

impl Op for DepthWise {
    fn name(&self) -> StaticName {
        "DepthWiseConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("relu={} scale={} {:?}", self.relu, self.scale, self.patch)])
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for DepthWise {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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
        anyhow::ensure!(inputs.len() == 3 || (self.scale && inputs.len() == 4));
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

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        if model.outlet_fact(node.id.into())?.datum_type != f32::datum_type() {
            return Ok(None);
        }
        if !self.relu && super::direct_spatial::successor_is_relu0(model, node)? {
            return Ok(Some(TypedModelPatch::fuse_with_next(
                model,
                node,
                Self { relu: true, ..self.clone() },
            )?));
        }
        if !self.scale {
            let c = *self.input_shape.c();
            if let Some(other) = super::direct_spatial::successor_is_channel_mul(model, node, c)? {
                let mut patch = TypedModelPatch::new("fuse channel scale into DepthWiseConv");
                let mut taps = patch.taps(model, &node.inputs)?;
                taps.push(patch.tap_model(model, other)?);
                let succ = model.node(node.outputs[0].successors[0].node);
                let out =
                    patch.wire_node(&node.name, Self { scale: true, ..self.clone() }, &taps)?;
                patch.shunt_outside(model, succ.id.into(), out[0])?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

    as_op!();
}

#[inline(always)]
fn maybe_relu_generic<T: Copy + 'static>(v: T, relu: bool) -> T {
    if relu && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let f = unsafe { *(&v as *const T as *const f32) };
        let r = f.max(0.0);
        return unsafe { std::ptr::read(&r as *const f32 as *const T) };
    }
    v
}

#[inline(always)]
fn apply_scale_f32<T: Copy + 'static>(optr: *mut T, scale_ptr: *const f32, c: isize) {
    if scale_ptr.is_null() || std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return;
    }
    unsafe {
        let sc = *scale_ptr.offset(c);
        *(optr as *mut f32) *= sc;
    }
}

macro_rules! impl_eval {
    ($(#[$meta: meta])* $suffix: ident ) => {
        pastey::paste! {
            $(#[$meta])*
            unsafe fn [<eval_t_ $suffix>]<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar + 'static>(
                dw: &DepthWise,
                inputs: TVec<TValue>,
                add: impl Fn(T, T) -> T + Copy + 'static,
                mul: impl Fn(T, T) -> T + Copy + 'static,
            ) -> TractResult<TVec<TValue>> {
                let img = &inputs[0];
                let kernel = &inputs[1];
                let bias = &inputs[2];
                let scale_ptr: *const f32 = if dw.scale
                    && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                {
                    inputs[3].as_ptr::<f32>()?
                } else {
                    std::ptr::null()
                };
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
                                add, mul, scale_ptr,
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
                scale_ptr: *const f32,
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
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul, scale_ptr,
                    ),
                    2 => [<process_zone_n_ $suffix>]::<T, 2, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul, scale_ptr,
                    ),
                    3 => [<process_zone_n_ $suffix>]::<T, 3, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul, scale_ptr,
                    ),
                    4 => [<process_zone_n_ $suffix>]::<T, 4, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul, scale_ptr,
                    ),
                    9 => [<process_zone_n_ $suffix>]::<T, 9, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul, scale_ptr,
                    ),
                    25 => [<process_zone_n_ $suffix>]::<T, 25, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul, scale_ptr,
                    ),
                    _ => {
                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                            process_zone_along_w_f32(
                                dw,
                                zone,
                                c_stride_i,
                                c_stride_o,
                                k_stride_i,
                                iptr as *const f32,
                                kptr as *const f32,
                                bias as *const f32,
                                optr as *mut f32,
                                scale_ptr,
                            );
                        } else {
                            zone.visit_output(&dw.patch, |visitor| {
                                for c in 0..*dw.input_shape.c() as isize {
                                    let iptr = iptr.offset(c_stride_i * c);
                                    let optr = optr.offset(c_stride_o * c);
                                    let kptr = kptr.offset(k_stride_i * c);
                                    [<inner_loop_ $suffix>]::<T>(
                                        iptr, kptr, bias, optr, c, visitor, add, mul, dw.relu,
                                        scale_ptr,
                                    )
                                }
                            })
                        }
                    }
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
                scale_ptr: *const f32,
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
                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                            && visitor.inner_loop_output_stride == 1
                            && visitor.inner_loop_input_full_stride >= 1
                        {
                            let k_f32 = &*(&k as *const [T; N] as *const [f32; N]);
                            let bias_f32 = *(&bias as *const T as *const f32);
                            super::along_w::conv_along_w_f32(
                                iptr as *const f32,
                                optr as *mut f32,
                                k_f32,
                                &ioffset,
                                bias_f32,
                                visitor.inner_loop_len,
                                visitor.inner_loop_input_full_stride,
                                dw.relu,
                            );
                            if !scale_ptr.is_null() {
                                let sc = *scale_ptr.offset(c);
                                let op = optr as *mut f32;
                                for i in 0..visitor.inner_loop_len {
                                    *op.add(i) *= sc;
                                }
                            }
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
                                *optrs[u] = maybe_relu_generic(sum, dw.relu);
                                apply_scale_f32(optrs[u], scale_ptr, c);
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
                            *optr = maybe_relu_generic(sum, dw.relu);
                            apply_scale_f32(optr, scale_ptr, c);
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
                relu: bool,
                scale_ptr: *const f32,
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
                *optr = maybe_relu_generic(sum, relu);
                apply_scale_f32(optr, scale_ptr, c);
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

unsafe fn process_zone_along_w_f32(
    dw: &DepthWise,
    zone: &Zone,
    c_stride_i: isize,
    c_stride_o: isize,
    k_stride_i: isize,
    iptr: *const f32,
    kptr: *const f32,
    bias: *const f32,
    optr: *mut f32,
    scale_ptr: *const f32,
) {
    unsafe {
        let n_taps = zone.values_offsets.len();
        let mut ioffset = vec![0isize; n_taps];
        let mut k = vec![0f32; n_taps];
        for i in 0..n_taps {
            ioffset[i] = zone.values_offsets[i].1;
        }
        let mut visitor = ZoneScanner::new(zone, &dw.patch);
        for c in 0..*dw.input_shape.c() as isize {
            visitor.reset();
            let iptr_c = iptr.offset(c_stride_i * c);
            let optr_c = optr.offset(c_stride_o * c);
            let kptr_c = kptr.offset(k_stride_i * c);
            for n in 0..n_taps {
                k[n] = *kptr_c.add(zone.values_offsets[n].0);
            }
            let b = *bias.offset(c);
            while !visitor.done {
                let ip = iptr_c.offset(visitor.input_center_offset);
                let op = optr_c.offset(visitor.output_offset);
                if visitor.inner_loop_output_stride == 1
                    && visitor.inner_loop_input_full_stride >= 1
                {
                    super::along_w::conv_along_w_f32(
                        ip,
                        op,
                        &k,
                        &ioffset,
                        b,
                        visitor.inner_loop_len,
                        visitor.inner_loop_input_full_stride,
                        dw.relu,
                    );
                    if !scale_ptr.is_null() {
                        let sc = *scale_ptr.offset(c);
                        for i in 0..visitor.inner_loop_len {
                            *op.add(i) *= sc;
                        }
                    }
                } else {
                    for i in 0..visitor.inner_loop_len {
                        let mut sum = b;
                        let ipi = ip.offset(visitor.inner_loop_input_full_stride * i as isize);
                        for n in 0..n_taps {
                            sum += k[n] * *ipi.offset(ioffset[n]);
                        }
                        if dw.relu {
                            sum = sum.max(0.0);
                        }
                        if !scale_ptr.is_null() {
                            sum *= *scale_ptr.offset(c);
                        }
                        *op.offset(visitor.inner_loop_output_stride * i as isize) = sum;
                    }
                }
                visitor.next_non_inner_axis();
            }
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
        // 2-D 3×3 / 5×5 (PP-OCR / MobileNet): interior zone is 9 or 25 taps.
        run_dw(8, 32, 32, 3, 3, PaddingSpec::SameUpper, (1, 1));
        run_dw(8, 32, 32, 3, 3, PaddingSpec::Valid, (1, 1));
        run_dw(4, 40, 40, 5, 5, PaddingSpec::SameUpper, (1, 1));
        run_dw(8, 32, 32, 3, 3, PaddingSpec::Valid, (2, 2));
    }

    #[test]
    fn depthwise_channel_scale_matches_reference() {
        // Conv fuses [1,C,1,1] Mul into the DW kernel before DepthWise exists.
        run_dw_scale(8, 9, 11, 3, 3, PaddingSpec::SameUpper, (1, 1), false);
        run_dw_scale(6, 1, 32, 1, 3, PaddingSpec::Valid, (1, 1), false);
        run_dw_scale(4, 8, 8, 3, 3, PaddingSpec::Valid, (1, 1), false);
    }

    #[test]
    fn depthwise_fuses_channel_scale() {
        run_dw_scale(8, 9, 11, 3, 3, PaddingSpec::SameUpper, (1, 1), true);
        run_dw_scale(6, 1, 32, 1, 3, PaddingSpec::Valid, (1, 1), true);
        // tiny_rec Conv.35: 1×5 along W, pad 2, C=160, H=1, W=40.
        run_dw_scale(
            160,
            1,
            40,
            1,
            5,
            PaddingSpec::Explicit(tvec!(0, 2), tvec!(0, 2)),
            (1, 1),
            true,
        );
    }

    fn run_dw_scale(
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        pad: PaddingSpec,
        stride: (usize, usize),
        after_lowering: bool,
    ) {
        let n = 1usize;
        let x: Vec<f32> = (0..n * c * h * w).map(|i| ((i as f32 * 0.137).sin()) * 0.7).collect();
        let kernel: Vec<f32> = (0..c * kh * kw).map(|i| ((i as f32 * 0.091).cos()) * 0.3).collect();
        let bias: Vec<f32> = (0..c).map(|i| (i as f32 * 0.05) - 0.1).collect();
        let scale: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.1).collect();

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
        let y = model.wire_node("dw", conv, &[xv, kv, bv]).unwrap();
        model.select_output_outlets(&y).unwrap();
        let mut model = if after_lowering {
            model.into_decluttered().unwrap().into_optimized().unwrap()
        } else {
            model
        };
        let conv_out = model.output_outlets().unwrap()[0];
        let sv =
            model.add_const("scale", Tensor::from_shape(&[1, c, 1, 1], &scale).unwrap()).unwrap();
        let out = model.wire_node("se", crate::ops::math::mul(), &[conv_out, sv]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let model = model.into_decluttered().unwrap().into_optimized().unwrap();
        let dw = model
            .nodes
            .iter()
            .find_map(|node| node.op_as::<DepthWise>())
            .expect("expected DepthWiseConv");
        if after_lowering {
            assert!(dw.scale, "channel-scale Mul should fuse into DepthWiseConv");
        }
        let runnable = model.into_runnable().unwrap();
        let got = runnable
            .run(tvec![Tensor::from_shape(&[n, c, h, w], &x).unwrap().into_tvalue()])
            .unwrap();
        let got = got[0].to_plain_array_view::<f32>().unwrap();
        let oh = got.shape()[2];
        let ow = got.shape()[3];
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
                    acc *= scale[oc];
                    max_abs = max_abs.max((got[[0, oc, oy, ox]] - acc).abs());
                }
            }
        }
        assert!(
            max_abs < 1e-4,
            "DepthWise scale mismatch c={c} {h}x{w} k={kh}x{kw} stride={stride:?} pad={pad:?} after_lowering={after_lowering}: max_abs={max_abs}"
        );
    }
}
