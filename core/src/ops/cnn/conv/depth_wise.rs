use crate::internal::*;
use crate::ops::cnn::patches::{Scanner, Zone, ZoneScanner};
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;

#[derive(Debug, Clone, new, Hash)]
pub struct DepthWise {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    kernel_chw: Arc<Tensor>,
    bias: Arc<Tensor>,
}

impl_dyn_hash!(DepthWise);

impl Op for DepthWise {
    fn name(&self) -> Cow<str> {
        "DepthWiseConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.patch)])
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_core_lir!();
    op_as_typed_op!();
}

impl EvalOp for DepthWise {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl DepthWise {
    fn eval_t<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar>(
        &self,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let img = args_1!(inputs);
        let mut output = unsafe { Tensor::uninitialized::<T>(&*self.output_shape.shape)? };
        let iptr = img.as_ptr::<T>()?;
        let optr = output.as_ptr_mut::<T>()?;
        let k_stride_i = self.kernel_chw.strides()[1];
        let n = *self.input_shape.n().unwrap_or(&1);
        let n_stride_i = *self.input_shape.n_stride().unwrap_or(&0) as isize;
        let n_stride_o = *self.output_shape.n_stride().unwrap_or(&0) as isize;
        let c_stride_i = *self.input_shape.c_stride() as isize;
        let c_stride_o = *self.output_shape.c_stride() as isize;
        let bias = self.bias.as_ptr::<T>()?;
        let kptr = self.kernel_chw.as_ptr::<T>()?;
        unsafe {
            for n in 0..n as isize {
                let iptr = iptr.offset(n_stride_i * n);
                let optr = optr.offset(n_stride_o * n);
                for zone in &self.patch.zones {
                    self.process_zone(
                        zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                    )
                }
            }
        }
        Ok(tvec!(output.into_arc_tensor()))
    }

    #[inline(never)]
    unsafe fn process_zone<T: Datum + Copy + ndarray::LinalgScalar>(
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
        if zone.values_offsets.len() == 4 {
            self.process_zone_4(zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr)
        } else {
            zone.visit_output(&self.patch, |visitor| {
                for c in 0..*self.input_shape.c() as isize {
                    let iptr = iptr.offset(c_stride_i * c);
                    let optr = optr.offset(c_stride_o * c);
                    let kptr = kptr.offset(k_stride_i * c);
                    Self::inner_loop::<T>(iptr, kptr, bias, optr, c, visitor)
                }
            })
        }
    }

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
        for c in 0..*self.input_shape.c() as isize {
            let kptr = kptr.offset(k_stride_i * c);
            let iptr = iptr.offset(c_stride_i * c);
            let optr = optr.offset(c_stride_o * c);
            let k0 = *kptr.offset(zone.values_offsets[0].0 as isize);
            let k1 = *kptr.offset(zone.values_offsets[1].0 as isize);
            let k2 = *kptr.offset(zone.values_offsets[2].0 as isize);
            let k3 = *kptr.offset(zone.values_offsets[3].0 as isize);
            let ioffset0 = zone.values_offsets[0].1;
            let ioffset1 = zone.values_offsets[1].1;
            let ioffset2 = zone.values_offsets[2].1;
            let ioffset3 = zone.values_offsets[3].1;
            let bias = *bias.offset(c);
            zone.visit_output(&self.patch, |visitor| {
                let iptr = iptr.offset(visitor.input_center_offset);
                let mut sum = bias;
                let i0 = *iptr.offset(ioffset0);
                let i1 = *iptr.offset(ioffset1);
                let i2 = *iptr.offset(ioffset2);
                let i3 = *iptr.offset(ioffset3);
                let p0 = i0 * k0;
                let p1 = i1 * k1;
                let p2 = i2 * k2;
                let p3 = i3 * k3;
                sum = sum + p0 + p1 + p2 + p3;
                *optr.offset(visitor.output_offset) = sum;
            })
        }
    }

    #[inline(never)]
    unsafe fn inner_loop<T: Datum + Copy + ndarray::LinalgScalar>(
        iptr: *const T,
        kptr: *const T,
        bias: *const T,
        optr: *mut T,
        c: isize,
        visitor: &ZoneScanner,
    ) {
        let mut sum = *bias.offset(c);
        let mut iter = visitor.valid_offsets_ker_in();
        if iter.size_hint() == (4, Some(4)) {
            let (ix, v) = iter.next().unwrap();
            let k0 = *kptr.offset(ix as isize);
            let i0 = *iptr.offset(v as isize);
            let (ix, v) = iter.next().unwrap();
            let k1 = *kptr.offset(ix as isize);
            let i1 = *iptr.offset(v as isize);
            let (ix, v) = iter.next().unwrap();
            let k2 = *kptr.offset(ix as isize);
            let i2 = *iptr.offset(v as isize);
            let (ix, v) = iter.next().unwrap();
            let k3 = *kptr.offset(ix as isize);
            let i3 = *iptr.offset(v as isize);
            sum = sum + k0 * i0 + k1 * i1 + k2 * i2 + k3 * i3;
        } else if iter.size_hint() == (3, Some(3)) {
            let (ix, v) = iter.next().unwrap();
            let k0 = *kptr.offset(ix as isize);
            let i0 = *iptr.offset(v as isize);
            let (ix, v) = iter.next().unwrap();
            let k1 = *kptr.offset(ix as isize);
            let i1 = *iptr.offset(v as isize);
            let (ix, v) = iter.next().unwrap();
            let k2 = *kptr.offset(ix as isize);
            let i2 = *iptr.offset(v as isize);
            sum = sum + k0 * i0 + k1 * i1 + k2 * i2;
        } else {
            for (ix, v) in iter {
                let k = *kptr.offset(ix as isize);
                let i = *iptr.offset(v as isize);
                sum = sum + k * i;
            }
        }
        let optr = optr.offset(visitor.output_offset);
        *optr = sum;
    }
}

impl TypedOp for DepthWise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(
            self.input_shape.c() == self.output_shape.c(),
            "DepthWiseConv must have same input and output channels"
        );
        anyhow::ensure!(
            *self.input_shape.c() == self.bias.len(),
            "DepthWiseConv data has {} channels, bias has {}",
            self.input_shape.c(),
            self.bias.len()
        );
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &self.output_shape.shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let n_output_points = self.patch.output_shape.iter().cloned().product::<usize>();
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            (self.input_shape.n().unwrap_or(&1) * n_output_points * self.kernel_chw.len()).to_dim()
        )))
    }

    as_op!();
}
