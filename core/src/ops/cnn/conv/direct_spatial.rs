//! Direct NCHW spatial convolution (no im2col), vectorised along W.
//!
//! Same lowering ORT keeps as `Conv`/`FusedConv` for small-`k` stems
//! (`k = Cin·kh·kw`). Depthwise goes through `DepthWise`; large-`k` 3×3 stays
//! on im2col + AMX.

use super::along_w::{conv_along_w_f32, conv_along_w_oc4_f32};
use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::cnn::patches::ZoneScanner;
use crate::ops::nn::DataShape;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct DirectSpatialConv {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    ic: usize,
    oc: usize,
    spatial: usize,
    relu: bool,
    scale: bool,
}

impl Op for DirectSpatialConv {
    fn name(&self) -> StaticName {
        "DirectSpatialConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "ic={} oc={} spatial={} relu={} scale={} {:?}",
            self.ic, self.oc, self.spatial, self.relu, self.scale, self.patch
        )])
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for DirectSpatialConv {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let dt = inputs[0].datum_type();
        dispatch_floatlike!(Self::eval_t(dt)(self, inputs))
    }
}

impl DirectSpatialConv {
    fn eval_t<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar>(
        &self,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        if T::datum_type() == f32::datum_type() {
            return unsafe { self.eval_f32(inputs) };
        }
        bail!("DirectSpatialConv is f32-only, got {:?}", T::datum_type())
    }

    unsafe fn eval_f32(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let img = &inputs[0];
        let kernel = &inputs[1];
        let bias = &inputs[2];
        let scale =
            if self.scale { Some(inputs[3].try_as_plain()?.as_slice::<f32>()?) } else { None };
        unsafe {
            let mut output = Tensor::uninitialized::<f32>(&self.output_shape.shape)?;
            let iptr = img.as_ptr::<f32>()?;
            let kptr = kernel.as_ptr::<f32>()?;
            let bptr = bias.as_ptr::<f32>()?;
            let optr = output.as_ptr_mut::<f32>()?;
            let n = *self.input_shape.n().unwrap_or(&1);
            let n_stride_i = *self.input_shape.n_stride().unwrap_or(&0) as isize;
            let n_stride_o = *self.output_shape.n_stride().unwrap_or(&0) as isize;
            let c_stride_i = *self.input_shape.c_stride() as isize;
            let c_stride_o = *self.output_shape.c_stride() as isize;
            let k_oc_stride = (self.ic * self.spatial) as isize;
            // `can_direct_spatial` keeps k = ic·kh·kw ≤ 64.
            const MAX_TAPS: usize = 64;
            for ni in 0..n as isize {
                let iptr_n = iptr.offset(n_stride_i * ni);
                let optr_n = optr.offset(n_stride_o * ni);
                for zone in &self.patch.zones {
                    let n_spatial = zone.values_offsets.len();
                    let n_taps = self.ic * n_spatial;
                    debug_assert!(n_taps <= MAX_TAPS);
                    let mut ioffset = [0isize; MAX_TAPS];
                    let mut t = 0usize;
                    for ic in 0..self.ic {
                        for (_ker_ix, in_off) in zone.values_offsets.iter() {
                            ioffset[t] = ic as isize * c_stride_i + *in_off;
                            t += 1;
                        }
                    }
                    let mut oc = 0usize;
                    while oc + 4 <= self.oc {
                        let mut kpack = [0f32; MAX_TAPS * 4];
                        let mut bias4 = [0f32; 4];
                        for o in 0..4 {
                            let k_oc = kptr.offset((oc + o) as isize * k_oc_stride);
                            bias4[o] = *bptr.add(oc + o);
                            let mut t = 0usize;
                            for ic in 0..self.ic {
                                for (ker_ix, _in_off) in zone.values_offsets.iter() {
                                    kpack[o * n_taps + t] = *k_oc.add(ic * self.spatial + *ker_ix);
                                    t += 1;
                                }
                            }
                        }
                        let optr_oc = optr_n.offset(c_stride_o * oc as isize);
                        let valid_rows = zone.valid
                            && zone.output_ranges.len() == 2
                            && *self.output_shape.w_stride() == 1;
                        if valid_rows {
                            let y0 = zone.output_ranges[0].start;
                            let y1 = zone.output_ranges[0].end;
                            let x0 = zone.output_ranges[1].start;
                            let x1 = zone.output_ranges[1].end;
                            let len = x1 - x0;
                            let sh = self.patch.spec.strides[0] as isize;
                            let sw = self.patch.spec.strides[1] as isize;
                            let in_h = *self.input_shape.h_stride() as isize;
                            let in_w = *self.input_shape.w_stride() as isize;
                            let out_h = *self.output_shape.h_stride() as isize;
                            let in_step = sw * in_w;
                            for oy in y0..y1 {
                                let ip =
                                    iptr_n.offset(oy as isize * sh * in_h + x0 as isize * in_step);
                                let op = optr_oc.offset(oy as isize * out_h + x0 as isize);
                                conv_along_w_oc4_f32(
                                    ip,
                                    op,
                                    &kpack[..4 * n_taps],
                                    &ioffset[..n_taps],
                                    &bias4,
                                    len,
                                    in_step,
                                    c_stride_o,
                                    self.relu,
                                );
                                scale_oc4_row(
                                    op,
                                    len,
                                    c_stride_o,
                                    scale,
                                    n,
                                    self.oc,
                                    ni as usize,
                                    oc,
                                );
                            }
                        } else {
                            let mut visitor = ZoneScanner::new(zone, &self.patch);
                            while !visitor.done {
                                let ip = iptr_n.offset(visitor.input_center_offset);
                                let op = optr_oc.offset(visitor.output_offset);
                                if visitor.inner_loop_output_stride == 1
                                    && visitor.inner_loop_input_full_stride >= 1
                                {
                                    conv_along_w_oc4_f32(
                                        ip,
                                        op,
                                        &kpack[..4 * n_taps],
                                        &ioffset[..n_taps],
                                        &bias4,
                                        visitor.inner_loop_len,
                                        visitor.inner_loop_input_full_stride,
                                        c_stride_o,
                                        self.relu,
                                    );
                                    scale_oc4_row(
                                        op,
                                        visitor.inner_loop_len,
                                        c_stride_o,
                                        scale,
                                        n,
                                        self.oc,
                                        ni as usize,
                                        oc,
                                    );
                                } else {
                                    for o in 0..4 {
                                        let b = bias4[o];
                                        let ko = o * n_taps;
                                        let opo = op.offset(c_stride_o * o as isize);
                                        for i in 0..visitor.inner_loop_len {
                                            let mut sum = b;
                                            let ipi = ip.offset(
                                                visitor.inner_loop_input_full_stride * i as isize,
                                            );
                                            for t in 0..n_taps {
                                                sum += kpack[ko + t] * *ipi.offset(ioffset[t]);
                                            }
                                            if self.relu {
                                                sum = sum.max(0.0);
                                            }
                                            *opo.offset(
                                                visitor.inner_loop_output_stride * i as isize,
                                            ) = sum;
                                        }
                                    }
                                }
                                visitor.next_non_inner_axis();
                            }
                        } // valid_rows else
                        oc += 4;
                    }
                    while oc < self.oc {
                        let k_oc = kptr.offset(oc as isize * k_oc_stride);
                        let b = *bptr.add(oc);
                        let optr_oc = optr_n.offset(c_stride_o * oc as isize);
                        let mut k = [0f32; MAX_TAPS];
                        let mut t = 0usize;
                        for ic in 0..self.ic {
                            for (ker_ix, _in_off) in zone.values_offsets.iter() {
                                k[t] = *k_oc.add(ic * self.spatial + *ker_ix);
                                t += 1;
                            }
                        }
                        let mut visitor = ZoneScanner::new(zone, &self.patch);
                        while !visitor.done {
                            let ip = iptr_n.offset(visitor.input_center_offset);
                            let op = optr_oc.offset(visitor.output_offset);
                            if visitor.inner_loop_output_stride == 1
                                && visitor.inner_loop_input_full_stride >= 1
                            {
                                conv_along_w_f32(
                                    ip,
                                    op,
                                    &k[..n_taps],
                                    &ioffset[..n_taps],
                                    b,
                                    visitor.inner_loop_len,
                                    visitor.inner_loop_input_full_stride,
                                    self.relu,
                                );
                                scale_row(
                                    op,
                                    visitor.inner_loop_len,
                                    scale_at(scale, n, self.oc, ni as usize, oc),
                                );
                            } else {
                                for i in 0..visitor.inner_loop_len {
                                    let mut sum = b;
                                    let ipi = ip
                                        .offset(visitor.inner_loop_input_full_stride * i as isize);
                                    for n in 0..n_taps {
                                        sum += k[n] * *ipi.offset(ioffset[n]);
                                    }
                                    if self.relu {
                                        sum = sum.max(0.0);
                                    }
                                    *op.offset(visitor.inner_loop_output_stride * i as isize) = sum;
                                }
                            }
                            visitor.next_non_inner_axis();
                        }
                        oc += 1;
                    }
                }
            }
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl TypedOp for DirectSpatialConv {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 3 || (self.scale && inputs.len() == 4));
        Ok(tvec!(inputs[0].datum_type.fact(&self.output_shape.shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let n_out = self.patch.output_shape.iter().cloned().product::<usize>();
        let n = *self.input_shape.n().unwrap_or(&1);
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            (self.oc * self.ic * self.spatial * n * n_out).to_dim()
        )))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        if !self.relu && successor_is_relu0(model, node)? {
            return Ok(Some(TypedModelPatch::fuse_with_next(
                model,
                node,
                Self { relu: true, ..self.clone() },
            )?));
        }
        if !self.scale
            && let Some(other) = successor_is_channel_mul(model, node, self.oc)?
        {
            let mut patch = TypedModelPatch::new("fuse channel scale into DirectSpatialConv");
            let mut taps = patch.taps(model, &node.inputs)?;
            taps.push(patch.tap_model(model, other)?);
            let succ = model.node(node.outputs[0].successors[0].node);
            let out = patch.wire_node(&node.name, Self { scale: true, ..self.clone() }, &taps)?;
            patch.shunt_outside(model, succ.id.into(), out[0])?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    as_op!();
}

pub(crate) fn successor_is_relu0(model: &TypedModel, node: &TypedNode) -> TractResult<bool> {
    if node.outputs.len() != 1 || node.outputs[0].successors.len() != 1 {
        return Ok(false);
    }
    if model.output_outlets()?.contains(&node.id.into()) {
        return Ok(false);
    }
    let succ_inlet = node.outputs[0].successors[0];
    let succ = model.node(succ_inlet.node);
    let is_max = succ
        .op_as::<crate::ops::binary::TypedBinOp>()
        .is_some_and(|op| op.0.is::<crate::ops::math::Max>())
        || succ
            .op_as::<crate::ops::binary::OptBinByScalar>()
            .is_some_and(|op| op.binop.is::<crate::ops::math::Max>());
    if !is_max || succ.inputs.len() != 2 {
        return Ok(false);
    }
    let other = succ.inputs[1 - succ_inlet.slot];
    let fact = model.outlet_fact(other)?;
    Ok(fact
        .uniform
        .as_ref()
        .is_some_and(|u| u.cast_to_scalar::<f32>().ok().is_some_and(|v| v == 0.0)))
}

pub(crate) fn successor_is_channel_mul(
    model: &TypedModel,
    node: &TypedNode,
    c: usize,
) -> TractResult<Option<OutletId>> {
    if node.outputs.len() != 1 || node.outputs[0].successors.len() != 1 {
        return Ok(None);
    }
    if model.output_outlets()?.contains(&node.id.into()) {
        return Ok(None);
    }
    let succ_inlet = node.outputs[0].successors[0];
    let succ = model.node(succ_inlet.node);
    let is_mul = succ
        .op_as::<crate::ops::binary::TypedBinOp>()
        .is_some_and(|op| op.0.is::<crate::ops::math::Mul>())
        || succ
            .op_as::<crate::ops::binary::OptBinByScalar>()
            .is_some_and(|op| op.binop.is::<crate::ops::math::Mul>());
    if !is_mul || succ.inputs.len() != 2 {
        return Ok(None);
    }
    let other = succ.inputs[1 - succ_inlet.slot];
    let other_fact = model.outlet_fact(other)?;
    let out_fact = model.outlet_fact(node.id.into())?;
    if other_fact.shape == out_fact.shape {
        return Ok(None);
    }
    let Ok(vol) = other_fact.shape.volume().to_usize() else {
        return Ok(None);
    };
    let n = out_fact.shape.get(0).and_then(|d| d.to_usize().ok()).unwrap_or(1);
    if vol != c && vol != c * n {
        return Ok(None);
    }
    Ok(Some(other))
}

fn scale_at(scale: Option<&[f32]>, n: usize, oc: usize, ni: usize, o: usize) -> f32 {
    match scale {
        None => 1.0,
        Some(s) if s.len() == oc => s[o],
        Some(s) if s.len() == n * oc => s[ni * oc + o],
        Some(s) if !s.is_empty() => s[o % s.len()],
        Some(_) => 1.0,
    }
}

unsafe fn scale_row(row: *mut f32, len: usize, sc: f32) {
    if sc == 1.0 {
        return;
    }
    unsafe {
        for i in 0..len {
            *row.add(i) *= sc;
        }
    }
}

unsafe fn scale_oc4_row(
    op: *mut f32,
    len: usize,
    c_stride: isize,
    scale: Option<&[f32]>,
    n: usize,
    oc: usize,
    ni: usize,
    oc0: usize,
) {
    let Some(_) = scale else { return };
    unsafe {
        for o in 0..4 {
            scale_row(op.offset(c_stride * o as isize), len, scale_at(scale, n, oc, ni, oc0 + o));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cnn::conv::{Conv, KernelFormat};
    use crate::ops::cnn::{PaddingSpec, PoolSpec};
    use crate::ops::nn::DataFormat;

    #[test]
    fn stem_3x3_s2_matches_im2col() {
        let n = 1usize;
        let ic = 3usize;
        let oc = 16usize;
        let h = 32usize;
        let w = 32usize;
        let kh = 3usize;
        let kw = 3usize;
        let x: Vec<f32> = (0..n * ic * h * w).map(|i| (i as f32 * 0.11).sin()).collect();
        let kernel: Vec<f32> =
            (0..oc * ic * kh * kw).map(|i| (i as f32 * 0.07).cos() * 0.2).collect();
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01).collect();

        let mut model = TypedModel::default();
        let xv = model.add_source("x", f32::fact([n, ic, h, w])).unwrap();
        let kv =
            model.add_const("k", Tensor::from_shape(&[oc, ic, kh, kw], &kernel).unwrap()).unwrap();
        let bv = model.add_const("b", Tensor::from_shape(&[oc], &bias).unwrap()).unwrap();
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(kh, kw),
                padding: PaddingSpec::Explicit(tvec!(1, 1), tvec!(1, 1)),
                dilations: None,
                strides: Some(tvec!(2, 2)),
                input_channels: ic,
                output_channels: oc,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        let out = model.wire_node("conv", conv, &[xv, kv, bv]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let model = model.into_decluttered().unwrap().into_optimized().unwrap();
        assert!(
            model.nodes.iter().any(|node| node.op_as::<DirectSpatialConv>().is_some()),
            "expected DirectSpatialConv, got {}",
            model.nodes.iter().map(|node| node.op().name()).collect::<Vec<_>>().join(",")
        );
        let runnable = model.into_runnable().unwrap();
        let got = runnable
            .run(tvec![Tensor::from_shape(&[n, ic, h, w], &x).unwrap().into_tvalue()])
            .unwrap();
        let y = got[0].to_plain_array_view::<f32>().unwrap();
        let oh = y.shape()[2];
        let ow = y.shape()[3];
        let mut max_abs = 0f32;
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias[o];
                    for c in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy as isize * 2 + ky as isize - 1;
                                let ix = ox as isize * 2 + kx as isize - 1;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let xv = x[(c * h + iy as usize) * w + ix as usize];
                                let kv = kernel[((o * ic + c) * kh + ky) * kw + kx];
                                acc += xv * kv;
                            }
                        }
                    }
                    max_abs = max_abs.max((y[[0, o, oy, ox]] - acc).abs());
                }
            }
        }
        assert!(max_abs < 1e-4, "stem mismatch max_abs={max_abs}");
    }

    #[test]
    fn stem_3x3_s2_fuses_relu() {
        let n = 1usize;
        let ic = 3usize;
        let oc = 8usize;
        let h = 16usize;
        let w = 16usize;
        let kh = 3usize;
        let kw = 3usize;
        let x: Vec<f32> = (0..n * ic * h * w).map(|i| (i as f32 * 0.11).sin() - 0.3).collect();
        let kernel: Vec<f32> =
            (0..oc * ic * kh * kw).map(|i| (i as f32 * 0.07).cos() * 0.2).collect();
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01 - 0.2).collect();

        let mut model = TypedModel::default();
        let xv = model.add_source("x", f32::fact([n, ic, h, w])).unwrap();
        let kv =
            model.add_const("k", Tensor::from_shape(&[oc, ic, kh, kw], &kernel).unwrap()).unwrap();
        let bv = model.add_const("b", Tensor::from_shape(&[oc], &bias).unwrap()).unwrap();
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(kh, kw),
                padding: PaddingSpec::Explicit(tvec!(1, 1), tvec!(1, 1)),
                dilations: None,
                strides: Some(tvec!(2, 2)),
                input_channels: ic,
                output_channels: oc,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        let y = model.wire_node("conv", conv, &[xv, kv, bv]).unwrap()[0];
        let zero = model.add_const("zero", tensor0(0f32).broadcast_into_rank(4).unwrap()).unwrap();
        let out = model.wire_node("relu", crate::ops::math::max(), &[y, zero]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let model = model.into_decluttered().unwrap().into_optimized().unwrap();
        let dsc = model
            .nodes
            .iter()
            .find_map(|node| node.op_as::<DirectSpatialConv>())
            .expect("expected DirectSpatialConv");
        assert!(dsc.relu, "Relu should fuse into DirectSpatialConv");
        assert!(
            !model.nodes.iter().any(|node| {
                node.op_as::<crate::ops::binary::TypedBinOp>()
                    .is_some_and(|op| op.0.is::<crate::ops::math::Max>())
                    || node
                        .op_as::<crate::ops::binary::OptBinByScalar>()
                        .is_some_and(|op| op.binop.is::<crate::ops::math::Max>())
            }),
            "fused Relu should not remain as a Max"
        );
        let runnable = model.into_runnable().unwrap();
        let got = runnable
            .run(tvec![Tensor::from_shape(&[n, ic, h, w], &x).unwrap().into_tvalue()])
            .unwrap();
        let y = got[0].to_plain_array_view::<f32>().unwrap();
        let oh = y.shape()[2];
        let ow = y.shape()[3];
        let mut max_abs = 0f32;
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias[o];
                    for c in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy as isize * 2 + ky as isize - 1;
                                let ix = ox as isize * 2 + kx as isize - 1;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let xv = x[(c * h + iy as usize) * w + ix as usize];
                                let kv = kernel[((o * ic + c) * kh + ky) * kw + kx];
                                acc += xv * kv;
                            }
                        }
                    }
                    acc = acc.max(0.0);
                    max_abs = max_abs.max((y[[0, o, oy, ox]] - acc).abs());
                }
            }
        }
        assert!(max_abs < 1e-4, "fused-relu stem mismatch max_abs={max_abs}");
    }

    #[test]
    fn conv_2x2_s1_oc16_matches_reference() {
        let n = 1usize;
        let ic = 8usize;
        let oc = 16usize;
        let h = 16usize;
        let w = 16usize;
        let kh = 2usize;
        let kw = 2usize;
        let x: Vec<f32> = (0..n * ic * h * w).map(|i| (i as f32 * 0.13).sin() - 0.2).collect();
        let kernel: Vec<f32> =
            (0..oc * ic * kh * kw).map(|i| (i as f32 * 0.09).cos() * 0.15).collect();
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.02 - 0.1).collect();

        let mut model = TypedModel::default();
        let xv = model.add_source("x", f32::fact([n, ic, h, w])).unwrap();
        let kv =
            model.add_const("k", Tensor::from_shape(&[oc, ic, kh, kw], &kernel).unwrap()).unwrap();
        let bv = model.add_const("b", Tensor::from_shape(&[oc], &bias).unwrap()).unwrap();
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(kh, kw),
                padding: PaddingSpec::SameUpper,
                dilations: None,
                strides: Some(tvec!(1, 1)),
                input_channels: ic,
                output_channels: oc,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        let y = model.wire_node("conv", conv, &[xv, kv, bv]).unwrap()[0];
        let zero = model.add_const("zero", tensor0(0f32).broadcast_into_rank(4).unwrap()).unwrap();
        let out = model.wire_node("relu", crate::ops::math::max(), &[y, zero]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let model = model.into_decluttered().unwrap().into_optimized().unwrap();
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.op_as::<DirectSpatialConv>().is_some_and(|d| d.relu)),
            "expected fused DirectSpatialConv"
        );
        let runnable = model.into_runnable().unwrap();
        let got = runnable
            .run(tvec![Tensor::from_shape(&[n, ic, h, w], &x).unwrap().into_tvalue()])
            .unwrap();
        let y = got[0].to_plain_array_view::<f32>().unwrap();
        let oh = y.shape()[2];
        let ow = y.shape()[3];
        let mut max_abs = 0f32;
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias[o];
                    for c in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy as isize + ky as isize;
                                let ix = ox as isize + kx as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let xv = x[(c * h + iy as usize) * w + ix as usize];
                                let kv = kernel[((o * ic + c) * kh + ky) * kw + kx];
                                acc += xv * kv;
                            }
                        }
                    }
                    acc = acc.max(0.0);
                    max_abs = max_abs.max((y[[0, o, oy, ox]] - acc).abs());
                }
            }
        }
        assert!(max_abs < 1e-4, "2x2 s=1 mismatch max_abs={max_abs}");
    }

    #[test]
    fn conv_2x2_16to8_matches_reference() {
        let n = 1usize;
        let ic = 16usize;
        let oc = 8usize;
        let h = 12usize;
        let w = 12usize;
        let kh = 2usize;
        let kw = 2usize;
        let x: Vec<f32> = (0..n * ic * h * w).map(|i| (i as f32 * 0.13).sin() - 0.2).collect();
        let kernel: Vec<f32> =
            (0..oc * ic * kh * kw).map(|i| (i as f32 * 0.09).cos() * 0.15).collect();
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.02 - 0.1).collect();

        let mut model = TypedModel::default();
        let xv = model.add_source("x", f32::fact([n, ic, h, w])).unwrap();
        let kv =
            model.add_const("k", Tensor::from_shape(&[oc, ic, kh, kw], &kernel).unwrap()).unwrap();
        let bv = model.add_const("b", Tensor::from_shape(&[oc], &bias).unwrap()).unwrap();
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec!(kh, kw),
                padding: PaddingSpec::SameUpper,
                dilations: None,
                strides: Some(tvec!(1, 1)),
                input_channels: ic,
                output_channels: oc,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        let out = model.wire_node("conv", conv, &[xv, kv, bv]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let model = model.into_decluttered().unwrap().into_optimized().unwrap();
        assert!(
            model.nodes.iter().any(|node| node.op_as::<DirectSpatialConv>().is_some()),
            "expected DirectSpatialConv for 2x2 16→8 (k=64)"
        );
        let runnable = model.into_runnable().unwrap();
        let got = runnable
            .run(tvec![Tensor::from_shape(&[n, ic, h, w], &x).unwrap().into_tvalue()])
            .unwrap();
        let y = got[0].to_plain_array_view::<f32>().unwrap();
        let oh = y.shape()[2];
        let ow = y.shape()[3];
        let mut max_abs = 0f32;
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias[o];
                    for c in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy as isize + ky as isize;
                                let ix = ox as isize + kx as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let xv = x[(c * h + iy as usize) * w + ix as usize];
                                let kv = kernel[((o * ic + c) * kh + ky) * kw + kx];
                                acc += xv * kv;
                            }
                        }
                    }
                    max_abs = max_abs.max((y[[0, o, oy, ox]] - acc).abs());
                }
            }
        }
        assert!(max_abs < 1e-4, "2x2 16→8 mismatch max_abs={max_abs}");
    }
}
