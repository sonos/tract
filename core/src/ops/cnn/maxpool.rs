use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct MaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Option<DatumType>,
}

impl Op for MaxPool {
    fn name(&self) -> StaticName {
        "MaxPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalOp for MaxPool {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let shape: TVec<TDim> = inputs[0].shape().iter().map(|d| d.to_dim()).collect();
        self.to_optimized(&shape)?.eval(_ctx, inputs)
    }
}

impl TypedOp for MaxPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut facts = self.pool_spec.output_facts(inputs)?;
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.with_index_outputs.is_some()
            && node.outputs[1].successors.len() == 0
            && !model.output_outlets()?.contains(&OutletId::new(node.id, 1))
        {
            let op = Self { with_index_outputs: None, ..self.clone() };
            let mut patch = TypedModelPatch::default();
            let mut wire = patch.tap_model(model, node.inputs[0])?;
            wire = patch.wire_node(&node.name, op, &[wire])?[0];
            patch.shunt_outside(model, node.id.into(), wire)?;
            return Ok(Some(patch));
        }
        let fact = model.outlet_fact(node.inputs[0])?;
        if let Some(pool_spec) = self.pool_spec.declutter(&fact.shape)? {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                Self { pool_spec, ..self.clone() },
            )?));
        }
        Ok(None)
    }

    /// Lower to `OptMaxPool` with the geometry pre-resolved to `Concrete` when the
    /// input shape is fixed, so the `Patch` is built once here rather than per eval.
    /// Symbolic shapes are left as `MaxPool`.
    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let fact = model.outlet_fact(node.inputs[0])?;
        if fact.shape.as_concrete().is_none() {
            return Ok(None);
        }
        let mut op = self.to_optimized(&fact.shape.to_tvec())?;
        op.geometry = op.geometry.optimize_if(fact.shape.as_concrete())?;
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, op)?))
    }

    as_op!();
}

impl MaxPool {
    fn to_optimized(&self, input_shape: &[TDim]) -> TractResult<OptMaxPool> {
        Ok(OptMaxPool {
            pool_spec: self.pool_spec.clone(),
            with_index_outputs: self.with_index_outputs,
            geometry: self.pool_spec.compute_geo(input_shape)?,
        })
    }
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct OptMaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Option<DatumType>,
    pub geometry: PoolGeometry,
}

impl Op for OptMaxPool {
    fn name(&self) -> StaticName {
        "OptMaxPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalOp for OptMaxPool {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let geo = self.geometry.to_concrete(input.shape())?;
        dispatch_numbers!(Self::eval_t(input.datum_type())(self, &*input, geo.as_ref()))
    }
}

impl TypedOp for OptMaxPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut facts = self.pool_spec.output_facts(inputs)?;
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }

    as_op!();
}

impl OptMaxPool {
    fn eval_t<T: Datum + Copy + num_traits::Bounded + PartialOrd>(
        &self,
        input: &Tensor,
        geo: &ConcretePoolGeometry,
    ) -> TractResult<TVec<TValue>> {
        let input_dt = input.datum_type();

        if self.with_index_outputs.is_none()
            && let Some(values) = self.try_nchw_2d::<T>(input, geo)?
        {
            return Ok(tvec!(values.into_tvalue()));
        }

        let input_plain = input.try_as_plain()?;
        let input: ArrayViewD<T> = input_plain.to_array_view()?;
        let input_ptr = input.as_ptr();

        let mut values = unsafe { ArrayD::<T>::uninit(&*geo.output_shape.shape).assume_init() };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::<i32>::uninit(&*geo.output_shape.shape).assume_init() })
        } else {
            None
        };
        let n = *geo.input_shape.n().unwrap_or(&1);
        let n_stride_i = geo.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = geo.output_shape.n_stride().unwrap_or(&0);
        unsafe {
            geo.patch.visit_output(|visitor| {
                for n in 0..n {
                    let input_offset = n * n_stride_i;
                    let output_offset = n * n_stride_o;
                    for c in 0..*geo.input_shape.c() {
                        let input_offset = input_offset + geo.input_shape.c_stride() * c;
                        let output_offset = output_offset + geo.output_shape.c_stride() * c;
                        let max = visitor
                            .valid_offsets()
                            .map(|v| (v, *input_ptr.offset(v + input_offset as isize)))
                            .fold((0, T::min_value()), |acc, v| if acc.1 < v.1 { v } else { acc });
                        *values
                            .as_mut_ptr()
                            .offset(output_offset as isize + visitor.output_offset) = max.1;
                        if let Some(ref mut indices) = indices {
                            *indices
                                .as_mut_ptr()
                                .offset(output_offset as isize + visitor.output_offset) =
                                max.0 as i32 / geo.patch.spec.output_inner_stride as i32;
                        }
                    }
                }
            });
        }
        let mut values = values.into_tensor();
        unsafe {
            values.set_datum_type(input_dt);
        }
        if let Some(dt) = self.with_index_outputs {
            Ok(tvec!(
                values.into_tvalue(),
                indices.unwrap().into_tensor().cast_to_dt(dt)?.into_owned().into_tvalue()
            ))
        } else {
            Ok(tvec!(values.into_tvalue()))
        }
    }

    /// NCHW spatial loop (n, c, y, x) instead of `visit_output`'s inverted
    /// (spatial, n, c). W is contiguous, so 2×2 stride-1 vectorises along x.
    fn try_nchw_2d<T: Datum + Copy + num_traits::Bounded + PartialOrd>(
        &self,
        input: &Tensor,
        geo: &ConcretePoolGeometry,
    ) -> TractResult<Option<Tensor>> {
        if self.pool_spec.data_format != crate::ops::nn::DataFormat::NCHW {
            return Ok(None);
        }
        let patch = &geo.patch;
        if patch.rank() != 2 || *geo.input_shape.w_stride() != 1 {
            return Ok(None);
        }
        let mut values =
            unsafe { Tensor::uninitialized_dt(input.datum_type(), &geo.output_shape.shape)? };
        if T::datum_type() == f32::datum_type()
            && patch.spec.kernel_shape[..] == [2, 2]
            && patch.spec.dilations[..] == [1, 1]
        {
            unsafe {
                maxpool_2x2_f32(input.as_ptr::<f32>()?, values.as_ptr_mut::<f32>()?, geo);
            }
            return Ok(Some(values));
        }
        unsafe {
            maxpool_nchw_2d::<T>(input.as_ptr::<T>()?, values.as_ptr_mut::<T>()?, geo);
        }
        Ok(Some(values))
    }
}

unsafe fn maxpool_nchw_2d<T: Copy + num_traits::Bounded + PartialOrd>(
    iptr: *const T,
    optr: *mut T,
    geo: &ConcretePoolGeometry,
) {
    unsafe {
        let ish = &geo.input_shape;
        let osh = &geo.output_shape;
        let (h, w) = (ish.hw_dims()[0] as isize, ish.hw_dims()[1] as isize);
        let (oh, ow) = (geo.patch.output_shape[0], geo.patch.output_shape[1]);
        let (kh, kw) =
            (geo.patch.spec.kernel_shape[0] as isize, geo.patch.spec.kernel_shape[1] as isize);
        let sh = geo.patch.spec.strides[0] as isize;
        let sw = geo.patch.spec.strides[1] as isize;
        let dh = geo.patch.spec.dilations[0] as isize;
        let dw = geo.patch.spec.dilations[1] as isize;
        let pt = geo.patch.pad_before[0] as isize;
        let pl = geo.patch.pad_before[1] as isize;
        let ih_stride = *ish.h_stride() as isize;
        let oh_stride = *osh.h_stride() as isize;
        let n = *ish.n().unwrap_or(&1) as isize;
        let in_stride = *ish.n_stride().unwrap_or(&0) as isize;
        let on_stride = *osh.n_stride().unwrap_or(&0) as isize;
        let c = *ish.c() as isize;
        let ic_stride = *ish.c_stride() as isize;
        let oc_stride = *osh.c_stride() as isize;
        for nn in 0..n {
            for cc in 0..c {
                let in_base = nn * in_stride + cc * ic_stride;
                let out_base = nn * on_stride + cc * oc_stride;
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut m = T::min_value();
                        for ky in 0..kh {
                            let iy = oy as isize * sh + ky * dh - pt;
                            if iy < 0 || iy >= h {
                                continue;
                            }
                            let row = iptr.offset(in_base + iy * ih_stride);
                            for kx in 0..kw {
                                let ix = ox as isize * sw + kx * dw - pl;
                                if ix < 0 || ix >= w {
                                    continue;
                                }
                                let v = *row.offset(ix);
                                if m < v {
                                    m = v;
                                }
                            }
                        }
                        *optr.offset(out_base + oy as isize * oh_stride + ox as isize) = m;
                    }
                }
            }
        }
    }
}

unsafe fn maxpool_2x2_f32(iptr: *const f32, optr: *mut f32, geo: &ConcretePoolGeometry) {
    unsafe {
        let ish = &geo.input_shape;
        let osh = &geo.output_shape;
        let (h, w) = (ish.hw_dims()[0] as isize, ish.hw_dims()[1] as isize);
        let (oh, ow) = (geo.patch.output_shape[0], geo.patch.output_shape[1]);
        let sh = geo.patch.spec.strides[0] as isize;
        let sw = geo.patch.spec.strides[1] as isize;
        let pt = geo.patch.pad_before[0] as isize;
        let pl = geo.patch.pad_before[1] as isize;
        let ih_stride = *ish.h_stride() as isize;
        let oh_stride = *osh.h_stride() as isize;
        let n = *ish.n().unwrap_or(&1) as isize;
        let in_stride = *ish.n_stride().unwrap_or(&0) as isize;
        let on_stride = *osh.n_stride().unwrap_or(&0) as isize;
        let c = *ish.c() as isize;
        let ic_stride = *ish.c_stride() as isize;
        let oc_stride = *osh.c_stride() as isize;
        // Fully-valid 2×2 windows (both taps in-bounds). SameUpper 2×2 s=1
        // keeps H/W and pads after, so the last row/col are partial.
        let simd_s1 = sh == 1 && sw == 1;
        let y0 = pt.max(0) as usize;
        let y1 = ((h - 1 + pt).max(0) as usize).min(oh);
        let x0 = pl.max(0) as usize;
        let x1 = ((w - 1 + pl).max(0) as usize).min(ow);
        for nn in 0..n {
            for cc in 0..c {
                let in_base = nn * in_stride + cc * ic_stride;
                let out_base = nn * on_stride + cc * oc_stride;
                if simd_s1 && y1 > y0 && x1 > x0 {
                    maxpool_2x2_s1_valid_f32(
                        iptr.offset(in_base + (y0 as isize - pt) * ih_stride + (x0 as isize - pl)),
                        optr.offset(out_base + y0 as isize * oh_stride + x0 as isize),
                        y1 - y0,
                        x1 - x0,
                        ih_stride,
                        oh_stride,
                    );
                }
                for oy in 0..oh {
                    let interior_y = simd_s1 && oy >= y0 && oy < y1;
                    for ox in 0..ow {
                        if interior_y && ox >= x0 && ox < x1 {
                            continue;
                        }
                        let mut m = f32::NEG_INFINITY;
                        for ky in 0..2 {
                            let iy = oy as isize * sh + ky - pt;
                            if iy < 0 || iy >= h {
                                continue;
                            }
                            let row = iptr.offset(in_base + iy * ih_stride);
                            for kx in 0..2 {
                                let ix = ox as isize * sw + kx - pl;
                                if ix < 0 || ix >= w {
                                    continue;
                                }
                                m = m.max(*row.offset(ix));
                            }
                        }
                        *optr.offset(out_base + oy as isize * oh_stride + ox as isize) = m;
                    }
                }
            }
        }
    }
}

unsafe fn maxpool_2x2_s1_valid_f32(
    iptr: *const f32,
    optr: *mut f32,
    oh: usize,
    ow: usize,
    ih_stride: isize,
    oh_stride: isize,
) {
    unsafe {
        for oy in 0..oh {
            let row0 = iptr.offset(oy as isize * ih_stride);
            let row1 = iptr.offset((oy as isize + 1) * ih_stride);
            let dst = optr.offset(oy as isize * oh_stride);
            let mut ox = 0usize;
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                while ox + 4 <= ow {
                    let a = vld1q_f32(row0.add(ox));
                    let b = vld1q_f32(row0.add(ox + 1));
                    let c = vld1q_f32(row1.add(ox));
                    let d = vld1q_f32(row1.add(ox + 1));
                    vst1q_f32(dst.add(ox), vmaxq_f32(vmaxq_f32(a, b), vmaxq_f32(c, d)));
                    ox += 4;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx") {
                    use std::arch::x86_64::*;
                    while ox + 8 <= ow {
                        let a = _mm256_loadu_ps(row0.add(ox));
                        let b = _mm256_loadu_ps(row0.add(ox + 1));
                        let c = _mm256_loadu_ps(row1.add(ox));
                        let d = _mm256_loadu_ps(row1.add(ox + 1));
                        _mm256_storeu_ps(
                            dst.add(ox),
                            _mm256_max_ps(_mm256_max_ps(a, b), _mm256_max_ps(c, d)),
                        );
                        ox += 8;
                    }
                }
            }
            while ox < ow {
                let m = (*row0.add(ox))
                    .max(*row0.add(ox + 1))
                    .max(*row1.add(ox))
                    .max(*row1.add(ox + 1));
                *dst.add(ox) = m;
                ox += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cnn::PaddingSpec;
    use crate::ops::nn::DataFormat;

    fn test_case() -> (TypedModel, TVec<TValue>) {
        let mut model = TypedModel::default();
        let source = model.add_source("data", f32::fact([1, 3, 8, 8])).unwrap();
        let pool_spec = PoolSpec::new(
            DataFormat::NCHW,
            tvec![2, 2],
            PaddingSpec::Valid,
            None,
            Some(tvec![2, 2]),
            3,
            3,
        );
        let op = MaxPool { pool_spec, with_index_outputs: None };
        let out = model.wire_node("pool", op, &[source]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let input = ndarray::Array4::from_shape_fn((1, 3, 8, 8), |(_, c, y, x)| {
            (c * 64 + y * 8 + x) as f32
        })
        .into_tensor()
        .into_tvalue();
        (model, tvec!(input))
    }

    #[test]
    fn optimized_maxpool_has_concrete_geometry() {
        let (model, input) = test_case();
        let plain = model.clone().into_runnable().unwrap().run(input.clone()).unwrap();

        let optimized = model.into_optimized().unwrap();
        let pool = optimized
            .nodes
            .iter()
            .find_map(|n| n.op_as::<OptMaxPool>())
            .expect("optimized model should contain an OptMaxPool");
        assert!(
            pool.geometry.is_concrete(),
            "OptMaxPool geometry should be concrete after optimization"
        );

        let opt = optimized.into_runnable().unwrap().run(input).unwrap();
        assert_eq!(*opt[0], *plain[0]);
    }

    fn nchw_maxpool(
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        pad: PaddingSpec,
    ) {
        let mut model = TypedModel::default();
        let source = model.add_source("data", f32::fact([n, c, h, w])).unwrap();
        let pool_spec = PoolSpec::new(
            DataFormat::NCHW,
            tvec![kh, kw],
            pad.clone(),
            None,
            Some(tvec![sh, sw]),
            c,
            c,
        );
        let op = MaxPool { pool_spec, with_index_outputs: None };
        let out = model.wire_node("pool", op, &[source]).unwrap();
        model.select_output_outlets(&out).unwrap();
        let x: Vec<f32> = (0..n * c * h * w).map(|i| (i as f32 * 0.17).sin() * 3.0 - 0.5).collect();
        let input = Tensor::from_shape(&[n, c, h, w], &x).unwrap().into_tvalue();
        let optimized = model.into_optimized().unwrap();
        assert!(optimized.nodes.iter().any(|node| node.op_as::<OptMaxPool>().is_some()));
        let opt = optimized.into_runnable().unwrap().run(tvec!(input)).unwrap();
        let got = opt[0].to_plain_array_view::<f32>().unwrap();
        let oh = got.shape()[2];
        let ow = got.shape()[3];
        let (pt, pl) = match pad {
            PaddingSpec::Valid => (0isize, 0isize),
            PaddingSpec::SameUpper | PaddingSpec::SameLower => {
                let ph = kh.saturating_sub(sh) as isize;
                let pw = kw.saturating_sub(sw) as isize;
                // SameUpper puts the extra pad after for even remainders; for
                // 2×2 s=1 that is pad_before=0, pad_after=1.
                if matches!(pad, PaddingSpec::SameUpper) {
                    (ph / 2, pw / 2)
                } else {
                    ((ph + 1) / 2, (pw + 1) / 2)
                }
            }
            PaddingSpec::Explicit(ref b, _) => {
                (b.first().copied().unwrap_or(0) as isize, b.get(1).copied().unwrap_or(0) as isize)
            }
            _ => (0, 0),
        };
        let mut max_abs = 0f32;
        for nn in 0..n {
            for cc in 0..c {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut m = f32::NEG_INFINITY;
                        for ky in 0..kh as isize {
                            let iy = oy as isize * sh as isize + ky - pt;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            for kx in 0..kw as isize {
                                let ix = ox as isize * sw as isize + kx - pl;
                                if ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                m = m.max(x[((nn * c + cc) * h + iy as usize) * w + ix as usize]);
                            }
                        }
                        max_abs = max_abs.max((got[[nn, cc, oy, ox]] - m).abs());
                    }
                }
            }
        }
        assert!(
            max_abs < 1e-6,
            "maxpool mismatch n={n} c={c} {h}x{w} k={kh}x{kw} s={sh}x{sw} pad={pad:?} max_abs={max_abs}"
        );
    }

    #[test]
    fn nchw_2x2_s1_matches_generic() {
        nchw_maxpool(1, 3, 8, 8, 2, 2, 1, 1, PaddingSpec::Valid);
        nchw_maxpool(1, 16, 17, 19, 2, 2, 1, 1, PaddingSpec::Valid);
        nchw_maxpool(2, 4, 9, 9, 2, 2, 1, 1, PaddingSpec::SameUpper);
        nchw_maxpool(1, 8, 16, 16, 2, 2, 2, 2, PaddingSpec::Valid);
        nchw_maxpool(1, 4, 11, 13, 3, 3, 1, 1, PaddingSpec::Valid);
    }
}
