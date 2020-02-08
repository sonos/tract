use num_traits::Zero;
use std::fmt;
use std::ops::{Add, Mul};

use crate::internal::*;
use ndarray::*;

use super::MMMWrapper;
use tract_linalg::mmm::FusedSpec;

use tract_linalg::frame::PackB;

#[derive(Debug, Clone, PartialEq)]
pub struct MatMatMulPackB<T>
where
    T: Copy + Datum + Zero,
{
    pub(crate) pack_b: PackB<T>,
    pub(crate) row_stride: isize,
    pub(crate) col_stride: isize,
    pub(crate) output_shape: TVec<usize>,
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

#[derive(Debug, Clone)]
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

impl<TA, TB, TC, TI> Op for MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    fn name(&self) -> Cow<str> {
        "MatMatMulUnaryFinite".into()
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
        infos.push(format!("{}", self.mmm));
        if let Some(f) = &self.fused_ops {
            infos.push(format!("{:?}", f));
        }
        Ok(infos)
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl<TA, TB, TC, TI> StatelessOp for MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
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

impl<TA, TB, TC, TI> TypedOp for MatMatMulUnaryFinite<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.c_fact.clone()))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        if let Some(succ) = model.single_succ(node.id)? {
            if let Some(op) = succ.op_as::<ops::array::FiniteReshape>() {
                let shape = op.shape.clone();
                return Ok(Some(TypedModelPatch::fuse_with_next(
                    model,
                    &node,
                    Self {
                        c_fact: TypedFact::dt_shape(self.c_fact.datum_type, &*shape)?,
                        ..self.clone()
                    },
                )?));
            }
            let fused_micro_op = (|| -> TractResult<Option<TVec<FusedSpec<TI>>>> {
                if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
                    let l =
                        if self.c_trans { self.mmm.as_mmm().m() } else { self.mmm.as_mmm().n() };
                    if op.a.len() == l && op.a.shape()[op.a.rank() - 1] == l {
                        if op.mini_op.is::<ops::math::Mul>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowMul(
                                op.a.as_slice::<TI>()?.to_vec(),
                            ))));
                        } else if op.mini_op.is::<ops::math::Add>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowAdd(
                                op.a.as_slice::<TI>()?.to_vec(),
                            ))));
                        }
                    }
                } else if let Some(op) = succ.op_as::<ops::element_wise::ElementWiseOp>() {
                    if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMax>() {
                        return Ok(Some(tvec!(FusedSpec::Max(op.max.cast_to_scalar()?))));
                    } else if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMin>() {
                        return Ok(Some(tvec!(FusedSpec::Min(op.min.cast_to_scalar()?))));
                    } else if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMinMax>() {
                        return Ok(Some(tvec!(
                            FusedSpec::Min(op.min.cast_to_scalar()?),
                            FusedSpec::Max(op.max.cast_to_scalar()?),
                        )));
                    }
                }
                Ok(None)
            })()?;
            if let Some(op) = fused_micro_op {
                let mut new_op = self.clone();
                new_op
                    .fused_ops
                    .get_or_insert_with(|| arr0(vec![]).into_dyn())
                    .map_inplace(|v| v.extend(op.iter().cloned()));
                return Ok(Some(TypedModelPatch::fuse_with_next(model, &node, new_op)?));
            }
        }
        Ok(None)
    }

    as_op!();
}
