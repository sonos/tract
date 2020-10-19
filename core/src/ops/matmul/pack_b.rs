use num_traits::Zero;

use crate::internal::*;
use ndarray::*;

use tract_linalg::frame::PackB;

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
        dyn_hash(&self, hasher)
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
}

impl<T> EvalOp for MatMatMulPackB<T>
where
    T: Copy + Datum + Zero,
{
    fn is_stateless(&self) -> bool {
        true
    }

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
