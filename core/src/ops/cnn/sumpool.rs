use crate::internal::*;
use num_traits::AsPrimitive;
use std::iter::Sum;

use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};

crate::declare_knob!(
    TRACT_AVGPOOL_SEPARABLE,
    bool,
    false,
    "Use the separable average-pool kernel for stride-1 NCHW/NHWC pools. Not bit-identical: \
     it reassociates the sum, permitted by SumPool's Validation::Rounding contract."
);

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct SumPool {
    pub pool_spec: PoolSpec,
    pub count_include_pad: bool,
    pub normalize: bool,
}

impl Op for SumPool {
    fn name(&self) -> StaticName {
        "SumPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for SumPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let shape: TVec<TDim> = inputs[0].shape().iter().map(|d| d.to_dim()).collect();
        self.to_optimized(&shape)?.eval(inputs)
    }
}

impl TypedOp for SumPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.pool_spec.output_facts(inputs)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
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

    /// Lower to `OptSumPool` with the geometry pre-resolved to `Concrete` when the
    /// input shape is fixed, so the `Patch` is built once here rather than per eval.
    /// Symbolic shapes are left as `SumPool`.
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

impl SumPool {
    fn to_optimized(&self, input_shape: &[TDim]) -> TractResult<OptSumPool> {
        Ok(OptSumPool {
            pool_spec: self.pool_spec.clone(),
            count_include_pad: self.count_include_pad,
            normalize: self.normalize,
            geometry: self.pool_spec.compute_geo(input_shape)?,
        })
    }
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct OptSumPool {
    pub pool_spec: PoolSpec,
    pub count_include_pad: bool,
    pub normalize: bool,
    pub geometry: PoolGeometry,
}

impl Op for OptSumPool {
    fn name(&self) -> StaticName {
        "OptSumPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for OptSumPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let geo = self.geometry.to_concrete(input.shape())?;
        let values = if input.datum_type().is_float() {
            let mut values =
                unsafe { Tensor::uninitialized_dt(input.datum_type(), &geo.output_shape.shape)? };
            dispatch_floatlike!(Self::eval_t(input.datum_type())(
                self,
                &*input,
                values.as_ptr_mut()?,
                geo.as_ref()
            ))?;
            values
        } else {
            let mut values =
                unsafe { Tensor::uninitialized_dt(DatumType::F32, &geo.output_shape.shape)? };
            let input_f32 = input.cast_to_dt(DatumType::F32)?;
            self.eval_t::<f32>(input_f32.as_ref(), values.as_ptr_mut()?, geo.as_ref())?;
            values.cast_to_dt(input.datum_type())?.into_owned()
        };

        Ok(tvec!(values.into_tvalue()))
    }
}

impl TypedOp for OptSumPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.pool_spec.output_facts(inputs)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
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

    as_op!();
}

impl OptSumPool {
    fn eval_t<T: Copy + Datum + Sum + num_traits::Float>(
        &self,
        input: &Tensor,
        values_ptr: *mut T,
        geo: &ConcretePoolGeometry,
    ) -> TractResult<()>
    where
        usize: AsPrimitive<T>,
    {
        if self.try_fast_2d::<T>(input, values_ptr, geo)? {
            return Ok(());
        }
        let input_ptr = input.as_ptr::<T>()?;

        let n = *geo.input_shape.n().unwrap_or(&1);
        let n_stride_i = geo.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = geo.output_shape.n_stride().unwrap_or(&0);
        unsafe {
            geo.patch.visit_output(|visitor| {
                let div: Option<T> = if self.normalize {
                    Some(
                        if self.count_include_pad {
                            geo.patch.standard_layout_data_field.len().as_()
                        } else {
                            visitor.valid_count().as_()
                        }
                        .recip(),
                    )
                } else {
                    None
                };
                for n in 0..n {
                    let input_offset = n * n_stride_i;
                    let output_offset = n * n_stride_o;
                    for c in 0..*geo.input_shape.c() {
                        let input_offset = input_offset + geo.input_shape.c_stride() * c;
                        let output_offset = output_offset + geo.output_shape.c_stride() * c;
                        let sum = visitor
                            .valid_offsets()
                            .map(|v| *input_ptr.offset(v + input_offset as isize))
                            .sum::<T>();

                        if let Some(div) = div {
                            *values_ptr.offset(output_offset as isize + visitor.output_offset) =
                                sum * div;
                        }
                    }
                }
            });
        }
        Ok(())
    }

    /// Opt-in separable average-pool fast path, gated by `TRACT_AVGPOOL_SEPARABLE`.
    /// Returns `true` when it handled the eval, `false` to fall back to the generic
    /// kernel. Restricted to rank-2, stride-1, dilation-1, normalize pools; the NCHW
    /// and NHWC layouts have dedicated kernels, any other layout falls back.
    fn try_fast_2d<T: Copy + Datum + num_traits::Float>(
        &self,
        input: &Tensor,
        values_ptr: *mut T,
        geo: &ConcretePoolGeometry,
    ) -> TractResult<bool>
    where
        usize: AsPrimitive<T>,
    {
        let patch = &geo.patch;
        if !TRACT_AVGPOOL_SEPARABLE.get()
            || !self.normalize
            || patch.rank() != 2
            || *patch.spec.strides != [1, 1]
            || *patch.spec.dilations != [1, 1]
        {
            return Ok(false);
        }
        let input_ptr = input.as_ptr::<T>()?;
        let ish = &geo.input_shape;
        if *ish.w_stride() == 1 {
            unsafe {
                self.fast_2d_separable::<T>(input_ptr, values_ptr, geo);
            }
            Ok(true)
        } else if *ish.c_stride() == 1 && *ish.w_stride() == *ish.c() {
            unsafe {
                self.fast_2d_separable_nhwc::<T>(input_ptr, values_ptr, geo);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Separable running-sum average pool. Out-of-bounds taps contribute 0, so the
    /// box sum over the padded window equals the sum of in-bounds values; the divisor
    /// is the per-cell valid count, itself separable into `kx_valid * ky_valid`.
    /// Reassociates the sum, so it is not bit-identical to the generic kernel.
    unsafe fn fast_2d_separable<T: Copy + Datum + num_traits::Float>(
        &self,
        input_ptr: *const T,
        values_ptr: *mut T,
        geo: &ConcretePoolGeometry,
    ) where
        usize: AsPrimitive<T>,
    {
        let ish = &geo.input_shape;
        let osh = &geo.output_shape;
        let (h, w) = (ish.hw_dims()[0] as isize, ish.hw_dims()[1] as isize);
        let (ho, wo) = (geo.patch.output_shape[0], geo.patch.output_shape[1]);
        let (kh, kw) =
            (geo.patch.spec.kernel_shape[0] as isize, geo.patch.spec.kernel_shape[1] as isize);
        let (pt, pl) = (geo.patch.pad_before[0] as isize, geo.patch.pad_before[1] as isize);
        let ih_stride = *ish.h_stride() as isize;
        let oh_stride = *osh.h_stride() as isize;
        let ow_stride = *osh.w_stride() as isize;
        let n = *ish.n().unwrap_or(&1);
        let in_stride = *ish.n_stride().unwrap_or(&0) as isize;
        let on_stride = *osh.n_stride().unwrap_or(&0) as isize;
        let c = *ish.c();
        let ic_stride = *ish.c_stride() as isize;
        let oc_stride = *osh.c_stride() as isize;

        let axis_valid = |out: usize, k: isize, pad: isize, lim: isize| -> Vec<usize> {
            (0..out)
                .map(|o| {
                    let lo = o as isize - pad;
                    let start = (-lo).max(0);
                    let end = (lim - lo).min(k);
                    (end - start).max(0) as usize
                })
                .collect()
        };
        let kx_valid = axis_valid(wo, kw, pl, w);
        let ky_valid = axis_valid(ho, kh, pt, h);
        let full_recip: T = ((kh * kw) as usize).as_().recip();

        let mut htmp = vec![T::zero(); h as usize * wo];
        unsafe {
            for nn in 0..n as isize {
                for cc in 0..c as isize {
                    let in_base = nn * in_stride + cc * ic_stride;
                    let out_base = nn * on_stride + cc * oc_stride;
                    for y in 0..h {
                        let row = in_base + y * ih_stride;
                        let dst = y as usize * wo;
                        let mut acc = T::zero();
                        for kx in 0..kw {
                            let ix = -pl + kx;
                            if ix >= 0 && ix < w {
                                acc = acc + *input_ptr.offset(row + ix);
                            }
                        }
                        *htmp.get_unchecked_mut(dst) = acc;
                        for ox in 1..wo as isize {
                            let entering = ox - pl + kw - 1;
                            let leaving = ox - pl - 1;
                            if entering >= 0 && entering < w {
                                acc = acc + *input_ptr.offset(row + entering);
                            }
                            if leaving >= 0 && leaving < w {
                                acc = acc - *input_ptr.offset(row + leaving);
                            }
                            *htmp.get_unchecked_mut(dst + ox as usize) = acc;
                        }
                    }
                    for ox in 0..wo {
                        let mut acc = T::zero();
                        for ky in 0..kh {
                            let iy = -pt + ky;
                            if iy >= 0 && iy < h {
                                acc = acc + *htmp.get_unchecked(iy as usize * wo + ox);
                            }
                        }
                        let store = |oy: usize, acc: T| {
                            let div = if self.count_include_pad {
                                full_recip
                            } else {
                                (kx_valid[ox] * ky_valid[oy]).as_().recip()
                            };
                            *values_ptr.offset(
                                out_base + oy as isize * oh_stride + ox as isize * ow_stride,
                            ) = acc * div;
                        };
                        store(0, acc);
                        for oy in 1..ho as isize {
                            let entering = oy - pt + kh - 1;
                            let leaving = oy - pt - 1;
                            if entering >= 0 && entering < h {
                                acc = acc + *htmp.get_unchecked(entering as usize * wo + ox);
                            }
                            if leaving >= 0 && leaving < h {
                                acc = acc - *htmp.get_unchecked(leaving as usize * wo + ox);
                            }
                            store(oy as usize, acc);
                        }
                    }
                }
            }
        }
    }

    /// NHWC counterpart of `fast_2d_separable`. Channels are the innermost
    /// (contiguous) axis, so both separable passes accumulate `C`-wide running
    /// sums, keeping the inner channel loops contiguous. Same reassociation
    /// caveat as the NCHW path: not bit-identical to the generic kernel.
    unsafe fn fast_2d_separable_nhwc<T: Copy + Datum + num_traits::Float>(
        &self,
        input_ptr: *const T,
        values_ptr: *mut T,
        geo: &ConcretePoolGeometry,
    ) where
        usize: AsPrimitive<T>,
    {
        let ish = &geo.input_shape;
        let osh = &geo.output_shape;
        let (h, w) = (ish.hw_dims()[0] as isize, ish.hw_dims()[1] as isize);
        let (ho, wo) = (geo.patch.output_shape[0], geo.patch.output_shape[1]);
        let (kh, kw) =
            (geo.patch.spec.kernel_shape[0] as isize, geo.patch.spec.kernel_shape[1] as isize);
        let (pt, pl) = (geo.patch.pad_before[0] as isize, geo.patch.pad_before[1] as isize);
        let ih_stride = *ish.h_stride() as isize;
        let iw_stride = *ish.w_stride() as isize;
        let oh_stride = *osh.h_stride() as isize;
        let ow_stride = *osh.w_stride() as isize;
        let n = *ish.n().unwrap_or(&1);
        let in_stride = *ish.n_stride().unwrap_or(&0) as isize;
        let on_stride = *osh.n_stride().unwrap_or(&0) as isize;
        let c = *ish.c();

        let axis_valid = |out: usize, k: isize, pad: isize, lim: isize| -> Vec<usize> {
            (0..out)
                .map(|o| {
                    let lo = o as isize - pad;
                    let start = (-lo).max(0);
                    let end = (lim - lo).min(k);
                    (end - start).max(0) as usize
                })
                .collect()
        };
        let kx_valid = axis_valid(wo, kw, pl, w);
        let ky_valid = axis_valid(ho, kh, pt, h);
        let full_recip: T = ((kh * kw) as usize).as_().recip();

        let mut htmp = vec![T::zero(); h as usize * wo * c];
        let mut acc = vec![T::zero(); c];
        unsafe {
            for nn in 0..n as isize {
                let in_base = nn * in_stride;
                let out_base = nn * on_stride;
                for y in 0..h {
                    let row = in_base + y * ih_stride;
                    let hrow = y as usize * wo * c;
                    acc.iter_mut().for_each(|a| *a = T::zero());
                    for kx in 0..kw {
                        let ix = -pl + kx;
                        if ix >= 0 && ix < w {
                            let p = row + ix * iw_stride;
                            for (ch, a) in acc.iter_mut().enumerate() {
                                *a = *a + *input_ptr.offset(p + ch as isize);
                            }
                        }
                    }
                    htmp[hrow..hrow + c].copy_from_slice(&acc);
                    for ox in 1..wo as isize {
                        let entering = ox - pl + kw - 1;
                        let leaving = ox - pl - 1;
                        if entering >= 0 && entering < w {
                            let p = row + entering * iw_stride;
                            for (ch, a) in acc.iter_mut().enumerate() {
                                *a = *a + *input_ptr.offset(p + ch as isize);
                            }
                        }
                        if leaving >= 0 && leaving < w {
                            let p = row + leaving * iw_stride;
                            for (ch, a) in acc.iter_mut().enumerate() {
                                *a = *a - *input_ptr.offset(p + ch as isize);
                            }
                        }
                        let dst = hrow + ox as usize * c;
                        htmp[dst..dst + c].copy_from_slice(&acc);
                    }
                }
                for ox in 0..wo {
                    acc.iter_mut().for_each(|a| *a = T::zero());
                    for ky in 0..kh {
                        let iy = -pt + ky;
                        if iy >= 0 && iy < h {
                            let src = iy as usize * wo * c + ox * c;
                            for (ch, a) in acc.iter_mut().enumerate() {
                                *a = *a + *htmp.get_unchecked(src + ch);
                            }
                        }
                    }
                    let store = |oy: usize, acc: &[T]| {
                        let div = if self.count_include_pad {
                            full_recip
                        } else {
                            (kx_valid[ox] * ky_valid[oy]).as_().recip()
                        };
                        let o = out_base + oy as isize * oh_stride + ox as isize * ow_stride;
                        for (ch, &a) in acc.iter().enumerate() {
                            *values_ptr.offset(o + ch as isize) = a * div;
                        }
                    };
                    store(0, &acc);
                    for oy in 1..ho as isize {
                        let entering = oy - pt + kh - 1;
                        let leaving = oy - pt - 1;
                        if entering >= 0 && entering < h {
                            let src = entering as usize * wo * c + ox * c;
                            for (ch, a) in acc.iter_mut().enumerate() {
                                *a = *a + *htmp.get_unchecked(src + ch);
                            }
                        }
                        if leaving >= 0 && leaving < h {
                            let src = leaving as usize * wo * c + ox * c;
                            for (ch, a) in acc.iter_mut().enumerate() {
                                *a = *a - *htmp.get_unchecked(src + ch);
                            }
                        }
                        store(oy as usize, &acc);
                    }
                }
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
        let op = SumPool { pool_spec, count_include_pad: false, normalize: true };
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
    fn optimized_sumpool_has_concrete_geometry() {
        let (model, input) = test_case();
        let plain = model.clone().into_runnable().unwrap().run(input.clone()).unwrap();

        let optimized = model.into_optimized().unwrap();
        let pool = optimized
            .nodes
            .iter()
            .find_map(|n| n.op_as::<OptSumPool>())
            .expect("optimized model should contain an OptSumPool");
        assert!(
            pool.geometry.is_concrete(),
            "OptSumPool geometry should be concrete after optimization"
        );

        let opt = optimized.into_runnable().unwrap().run(input).unwrap();
        assert_eq!(*opt[0], *plain[0]);
    }

    #[test]
    fn separable_matches_generic_kernel() {
        let (c, h, w) = (5usize, 7usize, 9usize);
        let pool_spec = PoolSpec::new(
            DataFormat::NCHW,
            tvec![3, 3],
            PaddingSpec::SameUpper,
            None,
            Some(tvec![1, 1]),
            c,
            c,
        );
        let op = OptSumPool {
            pool_spec: pool_spec.clone(),
            count_include_pad: false,
            normalize: true,
            geometry: pool_spec
                .compute_geo(&[1.to_dim(), c.to_dim(), h.to_dim(), w.to_dim()])
                .unwrap(),
        };
        let input: Tensor = ndarray::Array4::from_shape_fn((1, c, h, w), |(_, cc, y, x)| {
            ((cc * 17 + y * 3 + x) % 13) as f32 - 6.0
        })
        .into_tensor();

        // generic zoned kernel (knob off by default)
        let generic = op.eval(tvec![input.clone().into_tvalue()]).unwrap();
        let generic = generic[0].try_as_plain().unwrap().as_slice::<f32>().unwrap().to_vec();

        // separable kernel, called directly
        let geo = op.geometry.to_concrete(input.shape()).unwrap();
        let mut out = Tensor::zero::<f32>(&geo.output_shape.shape).unwrap();
        unsafe {
            op.fast_2d_separable::<f32>(
                input.as_ptr::<f32>().unwrap(),
                out.as_ptr_mut::<f32>().unwrap(),
                geo.as_ref(),
            );
        }
        let sep = out.try_as_plain().unwrap().as_slice::<f32>().unwrap();

        let max_abs = generic.iter().zip(sep).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_abs < 1e-4, "separable vs generic max abs diff {max_abs}");
    }

    #[test]
    fn separable_nhwc_matches_generic_kernel() {
        let (c, h, w) = (5usize, 7usize, 9usize);
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec![3, 3],
            PaddingSpec::SameUpper,
            None,
            Some(tvec![1, 1]),
            c,
            c,
        );
        let op = OptSumPool {
            pool_spec: pool_spec.clone(),
            count_include_pad: false,
            normalize: true,
            geometry: pool_spec
                .compute_geo(&[1.to_dim(), h.to_dim(), w.to_dim(), c.to_dim()])
                .unwrap(),
        };
        let input: Tensor = ndarray::Array4::from_shape_fn((1, h, w, c), |(_, y, x, cc)| {
            ((cc * 17 + y * 3 + x) % 13) as f32 - 6.0
        })
        .into_tensor();

        // generic zoned kernel (knob off by default)
        let generic = op.eval(tvec![input.clone().into_tvalue()]).unwrap();
        let generic = generic[0].try_as_plain().unwrap().as_slice::<f32>().unwrap().to_vec();

        // separable NHWC kernel, called directly
        let geo = op.geometry.to_concrete(input.shape()).unwrap();
        let mut out = Tensor::zero::<f32>(&geo.output_shape.shape).unwrap();
        unsafe {
            op.fast_2d_separable_nhwc::<f32>(
                input.as_ptr::<f32>().unwrap(),
                out.as_ptr_mut::<f32>().unwrap(),
                geo.as_ref(),
            );
        }
        let sep = out.try_as_plain().unwrap().as_slice::<f32>().unwrap();

        let max_abs = generic.iter().zip(sep).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_abs < 1e-4, "separable NHWC vs generic max abs diff {max_abs}");
    }
}
