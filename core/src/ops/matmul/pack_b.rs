use crate::internal::*;
use ndarray::*;

use tract_linalg::frame::PackB;

#[derive(Debug, Clone, PartialEq, Educe)]
#[educe(Hash)]
pub struct MatMatMulPackB {
    pub(crate) pack_b: PackB,
    pub(crate) trans: bool,
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

impl EvalOp for MatMatMulPackB {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let b = args_1!(inputs);
        let dt = b.datum_type();
        unsafe {
            let mut packed =
                Tensor::uninitialized_aligned_dt(dt, &*self.output_shape, self.pack_b.alignment())
                    .unwrap();
            for prefix in indices(&b.shape()[..b.rank() - 2]) {
                self.pack_b.pack(
                    &mut packed.view_at_prefix_mut(prefix.slice())?,
                    &b.view_at_prefix(prefix.slice())?,
                    self.trans,
                )
            }
            Ok(tvec!(packed.into_arc_tensor()))
        }
    }
}

impl TypedOp for MatMatMulPackB {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.output_shape)?))
    }

    as_op!();
}
