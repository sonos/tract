use crate::internal::*;
use ndarray::*;

use tract_linalg::frame::PackB;

#[derive(Debug, Clone, PartialEq, Educe)]
#[educe(Hash)]
pub struct MatMatMulPackB {
    pub(crate) pack_b: PackB,
    pub(crate) row_stride: isize,
    pub(crate) col_stride: isize,
    pub(crate) output_shape: TVec<usize>,
}

impl DynHash for MatMatMulPackB {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(&self, hasher)
    }
}

impl Op for MatMatMulPackB {
    fn name(&self) -> Cow<str> {
        "MatMatMulPackB".into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<Self>().map(|other| other == self).unwrap_or(false)
    }

    op_core_lir!();
    op_as_typed_op!();
}

impl MatMatMulPackB {
    unsafe fn pack_t<T: Datum + Copy>(&self, b: &Tensor, packed: &mut Tensor) {
        let b_prefix = &b.shape()[..b.shape().len() - 2];
        let b = b.to_array_view_unchecked::<T>();
        for prefix in indices(b_prefix).into_iter() {
            let mut b = b.view();
            let mut p = packed.to_array_view_mut_unchecked();
            for &dim in prefix.slice() {
                b.index_axis_inplace(Axis(0), dim);
                p.index_axis_inplace(Axis(0), dim);
            }
            self.pack_b.pack(p.as_mut_ptr(), b.as_ptr(), self.row_stride, self.col_stride)
        }
    }
}

impl EvalOp for MatMatMulPackB {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let b = args_1!(inputs);
        let dt = b.datum_type();
        let mut packed = unsafe {
            Tensor::uninitialized_aligned_dt(dt, &*self.output_shape, self.pack_b.alignment())
                .unwrap()
        };
        unsafe {
            dispatch_copy_by_size!(Self::pack_t(dt)(self, &b, &mut packed));
        }
        Ok(tvec!(packed.into_arc_tensor()))
    }
}

impl TypedOp for MatMatMulPackB {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.output_shape)?))
    }

    as_op!();
}
