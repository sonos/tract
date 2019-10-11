use num_traits::{AsPrimitive, Zero};
use std::fmt;
use std::ops::{Add, Mul};

use crate::internal::*;
use ndarray::*;

use tract_linalg::mmm::{FusedSpec, MatMatMul};

use tract_linalg::frame::PackB;

#[derive(Debug, Clone)]
pub struct MatMatMulPackB<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
{
    pub(crate) pack_b: PackB<T>,
    pub(crate) row_stride: isize,
    pub(crate) col_stride: isize,
    pub(crate) output_shape: TVec<usize>,
}

impl<T> Op for MatMatMulPackB<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
{
    fn name(&self) -> Cow<str> {
        "MatMatMulPackB".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl<T> StatelessOp for MatMatMulPackB<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
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
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
{
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.output_shape)?))
    }

    typed_op_as_op!();
}

#[derive(Debug, Clone)]
pub(crate) struct MatMatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: ::num_traits::AsPrimitive<T>,
{
    pub(crate) c_shape: TVec<usize>,
    pub(crate) c_prefix_dim_and_stride: Option<(TVec<usize>, TVec<isize>)>,
    pub(crate) packed_as: ArrayD<Tensor>,
    pub(crate) mmm: Box<dyn MatMatMul<T>>,
    pub(crate) non_linear: Vec<FusedSpec<T>>,
}

impl<T> Op for MatMatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike + fmt::Display,
    f32: AsPrimitive<T>,
{
    fn name(&self) -> Cow<str> {
        "MatMatMul".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut infos = vec![format!(
            "c_prefix: {:?} m:{} k:{} n:{}",
            self.c_prefix_dim_and_stride,
            self.mmm.m(),
            self.mmm.k(),
            self.mmm.n(),
        )];
        infos.push(format!("{}", self.mmm));
        if self.non_linear.len() > 0 {
            infos.push(format!("{:?}", self.non_linear))
        }
        Ok(infos)
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        if let Some(succ) = model.single_succ(node.id)? {
            let fused_micro_op = (|| -> TractResult<Option<TVec<FusedSpec<T>>>> {
                if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
                    if op.a.shape() == &[self.mmm.m()] {
                        if op.mini_op.is::<ops::math::Mul>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowMul(
                                op.a.as_slice::<T>()?.to_vec(),
                            ))));
                        } else if op.mini_op.is::<ops::math::Add>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowAdd(
                                op.a.as_slice::<T>()?.to_vec(),
                            ))));
                        }
                    }
                } else if let Some(op) = succ.op_as::<ops::element_wise::ElementWiseOp>() {
                    if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMax>() {
                        return Ok(Some(tvec!(FusedSpec::Max(op.max.as_()))));
                    } else if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMin>() {
                        return Ok(Some(tvec!(FusedSpec::Min(op.min.as_()))));
                    } else if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMinMax>() {
                        return Ok(Some(tvec!(
                            FusedSpec::Min(op.min.as_()),
                            FusedSpec::Max(op.max.as_()),
                        )));
                    }
                }
                Ok(None)
            })()?;
            if let Some(op) = fused_micro_op {
                let mut ops = self.non_linear.clone();
                ops.extend(op.into_iter());
                return Ok(Some(TypedModelPatch::fuse_with_next(
                    model,
                    &node,
                    Self { non_linear: ops, ..self.clone() },
                )?));
            }
        }
        Ok(None)
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl<T> StatelessOp for MatMatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: ::num_traits::AsPrimitive<T>,
{
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unsafe {
            let b = args_1!(inputs);
            let mut c = Tensor::uninitialized::<T>(&*self.c_shape)?;
            if let Some((prefix_dim, prefix_strides)) = &self.c_prefix_dim_and_stride {
                let b = b.to_array_view::<T>()?;
                let mut c = c.to_array_view_mut::<T>()?;
                for prefix in indices(&**prefix_dim).into_iter() {
                    let mut a = self.packed_as.view();
                    let mut b = b.view();
                    let mut c: *mut T = c.as_mut_ptr();
                    for (ix, &dim) in prefix.slice().iter().enumerate() {
                        let d = dim.min(a.shape()[0] - 1);
                        a.index_axis_inplace(Axis(0), d);
                        let d = dim.min(b.shape()[0] - 1);
                        b.index_axis_inplace(Axis(0), d);
                        c = c.offset(prefix_strides[ix] * dim as isize);
                    }
                    let pa: &Tensor = a.iter().next().unwrap();
                    self.mmm.run(pa.as_ptr()?, b.as_ptr(), c, &self.non_linear);
                }
            } else {
                self.mmm.run(
                    self.packed_as.as_slice().unwrap()[0].as_ptr()?,
                    b.as_ptr()?,
                    c.as_ptr_mut()?,
                    &self.non_linear,
                );
            }
            Ok(tvec!(c.into_arc_tensor()))
        }
    }
}

impl<T> TypedOp for MatMatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: ::num_traits::AsPrimitive<T>,
{
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.c_shape)?))
    }

    /*
    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let g = &self.geo;
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            (g.c_shape_prefix.iter().product::<usize>() * g.m * g.k * g.n).into()
        )))
    }
    */

    typed_op_as_op!();
}
