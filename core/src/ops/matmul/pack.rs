use crate::internal::*;
use ndarray::*;

use tract_linalg::frame::Packer;

#[derive(Debug, Clone, PartialEq, Educe)]
#[educe(Hash)]
pub struct MatMatMulPack {
    pub(crate) packer: Packer,
    pub(crate) trans: bool,
    pub(crate) output_shape: TVec<usize>,
}

impl DynHash for MatMatMulPack {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(&self, hasher)
    }
}

impl Op for MatMatMulPack {
    fn name(&self) -> Cow<str> {
        "MatMatMulPack".into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<Self>().map(|other| other == self).unwrap_or(false)
    }

    op_core_lir!();
    op_as_typed_op!();
}

impl EvalOp for MatMatMulPack {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TensorVar>) -> TractResult<TVec<Tensor>> {
        let b = args_1!(inputs);
        let dt = b.datum_type();
        unsafe {
            let mut packed =
                Tensor::uninitialized_aligned_dt(dt, &*self.output_shape, self.packer.alignment())
                    .unwrap();
            for prefix in indices(&b.shape()[..b.rank() - 2]) {
                self.packer.pack(
                    &mut packed.view_at_prefix_mut(prefix.slice())?,
                    &b.view_at_prefix(prefix.slice())?,
                    self.trans as usize,
                    !self.trans as usize,
                )
            }
            Ok(tvec!(packed))
        }
    }
}

impl TypedOp for MatMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &self.output_shape)))
    }

    as_op!();
}
