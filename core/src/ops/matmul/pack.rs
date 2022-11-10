use crate::internal::*;
use ndarray::*;

use tract_linalg::frame::Packer;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatMatMulPack {
    pub(crate) packer: Packer,
    pub(crate) output_shape: TVec<usize>,
    pub(crate) k_axis: usize,
    pub(crate) mn_axis: usize,
}

impl DynHash for MatMatMulPack {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(self, hasher)
    }
}

impl Op for MatMatMulPack {
    fn name(&self) -> Cow<str> {
        "MatMatMulPack".into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<Self>().map(|other| other == self).unwrap_or(false)
    }

    op_as_typed_op!();
}

impl EvalOp for MatMatMulPack {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let b = args_1!(inputs);
        let dt = b.datum_type();
        unsafe {
            let mut packed =
                Tensor::uninitialized_aligned_dt(dt, &self.output_shape, self.packer.alignment())
                    .unwrap();
            let mut bc_shape: TVec<usize> = b.shape().into();
            bc_shape[self.k_axis] = 1;
            bc_shape[self.mn_axis] = 1;
            for coord in indices(&*bc_shape) {
                let offset = coord
                    .as_array_view()
                    .iter()
                    .zip(b.strides())
                    .map(|(x, s)| *x as isize * s)
                    .sum::<isize>()
                    * b.datum_type().size_of() as isize;
                let mut prefix:TVec<usize> = coord.slice().into();
                prefix.remove(self.k_axis.max(self.mn_axis));
                prefix.remove(self.k_axis.min(self.mn_axis));
                self.packer.pack(
                    &mut packed.view_at_prefix_mut(&prefix)?,
                    TensorView::from_bytes(&b, offset, b.shape(), b.strides()),
                    self.k_axis,
                    self.mn_axis,
                )
            }
            Ok(tvec!(packed.into_tvalue()))
        }
    }
}

impl TypedOp for MatMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(&self.output_shape)))
    }

    as_op!();
}
