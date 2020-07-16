use num_traits::Zero;
use std::fmt;
use std::ops::{Add, Mul};

use crate::internal::*;
use crate::ops;
use ndarray::*;

use super::MMMWrapper;
use tract_linalg::mmm::FusedSpec;

use tract_linalg::frame::PackB;

pub trait FusableOps: std::fmt::Debug + Copy {
    fn fusable_ops(op: &dyn Op) -> Option<TVec<FusedSpec<Self>>>;
}

impl FusableOps for i32 {
    fn fusable_ops(op: &dyn Op) -> Option<TVec<FusedSpec<Self>>> {
        if let Some(op) = op.downcast_ref::<ops::element_wise::ElementWiseOp>() {
            if let Some(q) = op.0.downcast_ref::<ops::quant::QScaleInt32>() {
                let pair = i32_optimized_scale_factor(q.scale);
                Some(tvec!(FusedSpec::QTowardsPlusInf(pair.0, pair.1)))
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl FusableOps for f32 {
    fn fusable_ops(_op: &dyn Op) -> Option<TVec<FusedSpec<Self>>> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Educe)]
#[educe(Hash)]
pub struct MatMatMulPackB<T>
where
    T: Copy + Datum + Zero,
{
    pub(crate) pack_b: PackB<T>,
    pub(crate) row_stride: isize,
    pub(crate) col_stride: isize,
    pub(crate) output_shape: TVec<usize>,
}

impl<T> DynHash for MatMatMulPackB<T>
where
    T: Copy + Datum + Zero,
{
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        tract_linalg::hash::dyn_hash(&self, hasher)
    }
}

impl<T> Op for MatMatMulPackB<T>
where
    T: Copy + Datum + Zero,
{
    fn name(&self) -> Cow<str> {
        "MatMatMulPackB".into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<Self>().map(|other| other == self).unwrap_or(false)
    }

    op_core_lir!();
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl<T> StatelessOp for MatMatMulPackB<T>
where
    T: Copy + Datum + Zero,
{
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let b = args_1!(inputs);
        let mut packed = unsafe {
            Tensor::uninitialized_aligned::<T>(&*self.output_shape, self.pack_b.alignment())
                .unwrap()
        };
        if b.shape()[..b.shape().len() - 2].iter().any(|d| *d > 1) {
            let b = b.to_array_view::<T>()?;
            let b_prefix = &b.shape()[..b.shape().len() - 2];
            for prefix in indices(b_prefix).into_iter() {
                let mut b = b.view();
                let mut p = packed.to_array_view_mut()?;
                for &dim in prefix.slice() {
                    b.index_axis_inplace(Axis(0), dim);
                    p.index_axis_inplace(Axis(0), dim);
                }
                self.pack_b.pack(p.as_mut_ptr(), b.as_ptr(), self.row_stride, self.col_stride)
            }
        } else {
            self.pack_b.pack(packed.as_ptr_mut()?, b.as_ptr()?, self.row_stride, self.col_stride)
        }
        Ok(tvec!(packed.into_arc_tensor()))
    }
}

impl<T> TypedOp for MatMatMulPackB<T>
where
    T: Copy + Datum + Zero,
{
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.output_shape)?))
    }

    as_op!();
}

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub(crate) struct MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    pub(crate) c_trans: bool,
    pub(crate) bc_c_shape: TVec<usize>,
    pub(crate) c_fact: TypedFact,
    pub(crate) c_prefix_dim_and_stride: Option<(TVec<usize>, TVec<isize>)>,
    pub(crate) packed_as: ArrayD<Arc<Tensor>>,
    pub(crate) fused_ops: Option<ArrayD<Vec<FusedSpec<TI>>>>,
    pub(crate) mmm: MMMWrapper<TA, TB, TC, TI>,
}

impl<TA, TB, TC, TI> DynHash for MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        tract_linalg::hash::dyn_hash(&self, hasher)
    }
}

impl<TA, TB, TC, TI> Op for MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug + FusableOps,
{
    fn name(&self) -> Cow<str> {
        if self.mmm.as_mmm().n() == 1 { "MatVecMul" } else { "MatMatMul" }.into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut infos = vec![format!(
            "c_prefix: {:?} m:{} k:{} n:{} c_trans:{:?}",
            self.c_prefix_dim_and_stride,
            self.mmm.as_mmm().m(),
            self.mmm.as_mmm().k(),
            self.mmm.as_mmm().n(),
            self.c_trans
        )];
        infos.push(format!("Mult: {}", self.mmm));
        if let Some(f) = &self.fused_ops {
            infos.push(format!("{:?}", f));
        }
        Ok(infos)
    }

    op_core_lir!();
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl<TA, TB, TC, TI> StatelessOp for MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug + FusableOps,
{
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unsafe {
            let b = args_1!(inputs);
            let mut c = Tensor::uninitialized::<TC>(&*self.c_fact.shape.as_finite().unwrap())?;
            if let Some((prefix_dim, prefix_strides)) = &self.c_prefix_dim_and_stride {
                let b = b.to_array_view::<TB>()?;
                let mut c = c.to_array_view_mut::<TC>()?;
                for prefix in indices(&**prefix_dim).into_iter() {
                    let mut a = self.packed_as.view();
                    let mut b = b.view();
                    let mut c: *mut TC = c.as_mut_ptr();
                    for (ix, &dim) in prefix.slice().iter().enumerate() {
                        let d = dim.min(a.shape()[0] - 1);
                        a.index_axis_inplace(Axis(0), d);
                        let d = dim.min(b.shape()[0] - 1);
                        b.index_axis_inplace(Axis(0), d);
                        c = c.offset(prefix_strides[ix] * dim as isize);
                    }
                    let pa: &Tensor = a.iter().next().unwrap();
                    if let Some(fused) = &self.fused_ops {
                        let mut fused = fused.view();
                        for &dim in prefix.slice() {
                            let d = dim.min(fused.shape()[0] - 1);
                            fused.index_axis_inplace(Axis(0), d);
                        }
                        self.mmm.run(pa.as_ptr()?, b.as_ptr(), c, &fused.as_slice().unwrap()[0]);
                    } else {
                        self.mmm.run(pa.as_ptr()?, b.as_ptr(), c, &[]);
                    }
                }
            } else {
                if let Some(fused) = &self.fused_ops {
                    self.mmm.run(
                        self.packed_as.as_slice().unwrap()[0].as_ptr()?,
                        b.as_ptr()?,
                        c.as_ptr_mut()?,
                        &fused.as_slice().unwrap()[0],
                    );
                } else {
                    self.mmm.run(
                        self.packed_as.as_slice().unwrap()[0].as_ptr()?,
                        b.as_ptr()?,
                        c.as_ptr_mut()?,
                        &[],
                    );
                }
            }
            Ok(tvec!(c.into_arc_tensor()))
        }
    }
}

fn i32_optimized_scale_factor(factor: f32) -> (i32, usize) {
    // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/util/gemmlowp_common.h#L16
    let factor_bits = factor.to_bits();
    let current_exponent = factor_bits >> 23;
    let bumped_multi = f32::from_bits(factor_bits & 0x007fffff | 0x3f000000);
    let int_multi = (bumped_multi * (1i64 << 31) as f32).round() as i32;
    let shift = 126 - current_exponent;
    (int_multi, shift as usize)
}

impl<TA, TB, TC, TI> TypedOp for MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug + FusableOps,
{
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.c_fact.clone()))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let mmm = self.mmm.as_mmm();
        let mul = self.c_prefix_dim_and_stride.as_ref().map(|c| c.0.iter().product()).unwrap_or(1);
        Ok(tvec!(
            (Cost::FMA(TI::datum_type()), (mul * mmm.m() * mmm.n() * mmm.k()).to_dim()),
            (
                Cost::Params(TA::datum_type()),
                self.packed_as.iter().fold(0.to_dim(), |sum, a| sum + a.len())
            )
        ))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        if let Some(succ) = model.single_succ(node.id)? {
            if let Some(op) = succ.op_as::<ops::AxisOp>() {
                if op.only_shape() {
                    return Ok(Some(TypedModelPatch::fuse_with_next(
                        model,
                        &node,
                        Self { c_fact: succ.outputs[0].fact.clone(), ..self.clone() },
                    )?));
                }
            }
            let fused_micro_op = if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
                let m = self.mmm.as_mmm().m();
                if op.a.len() == m
                    && op.a.shape()[op.a.rank() - 1 - ((!self.c_trans) as usize)] == m
                {
                    if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::PerRowMul(op.a.as_slice::<TI>()?.to_vec(),)))
                    } else if op.mini_op.is::<ops::math::Add>() {
                        Some(tvec!(FusedSpec::PerRowAdd(op.a.as_slice::<TI>()?.to_vec(),)))
                    } else {
                        None
                    }
                } else if op.a.len() == 1 {
                    if op.mini_op.is::<ops::math::Max>() {
                        Some(tvec!(FusedSpec::Max(op.a.cast_to_scalar()?)))
                    } else if op.mini_op.is::<ops::math::Min>() {
                        Some(tvec!(FusedSpec::Min(op.a.cast_to_scalar()?)))
                    } else if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::ScalarMul(op.a.cast_to_scalar()?)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                TI::fusable_ops(succ.op.as_ref())
            };
            if let Some(op) = fused_micro_op {
                let mut new_op = self.clone();
                new_op
                    .fused_ops
                    .get_or_insert_with(|| {
                        let shape = vec![
                            1;
                            self.c_prefix_dim_and_stride
                                .as_ref()
                                .map(|c| c.0.len())
                                .unwrap_or(0)
                        ];
                        ArrayD::from_shape_fn(shape, |_| vec![])
                    })
                    .map_inplace(|v| v.extend(op.iter().cloned()));
                return Ok(Some(TypedModelPatch::fuse_with_next(model, &node, new_op)?));
            }
        }
        Ok(None)
    }

    as_op!();
}
