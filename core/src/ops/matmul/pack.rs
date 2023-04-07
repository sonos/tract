use crate::axes::Axis;
use crate::internal::*;
use ndarray::*;

use tract_linalg::frame::Packer;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatMatMulPack {
    pub(crate) packer: Packer,
    pub(crate) k_axis: usize,
    pub(crate) mn_axis: usize,
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
            let output_shape = self.output_shape(b.shape());
            let mut packed =
                Tensor::uninitialized_aligned_dt(dt, &output_shape, self.packer.alignment())
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
                let mut prefix: TVec<usize> = coord.slice().into();
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
        Ok(tvec!(inputs[0].datum_type.fact(self.output_shape(&inputs[0].shape))))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes: Vec<Axis> = (0..inputs[0].rank())
            .filter(|&ix| ix != self.k_axis && ix != self.mn_axis)
            .enumerate()
            .zip('a'..)
            .map(|((o, i), repr)| Axis::new(repr, 1, 1).input(0, i).output(0, o))
            .collect();
        axes.push(Axis::new('K', 1, 1).input(0, self.k_axis));
        axes.push(Axis::new('M', 1, 1).input(0, self.mn_axis));
        axes.push(Axis::new('P', 1, 1).output(0, outputs[0].rank() - 1));
        AxesMapping::new(1, 1, axes)
    }

    as_op!();
}

impl MatMatMulPack {
    fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut packed_shape: TVec<D> = input.into();
        packed_shape.remove(self.mn_axis.max(self.k_axis));
        packed_shape.remove(self.mn_axis.min(self.k_axis));
        packed_shape.push(self.packer.len(input[self.k_axis].clone(), input[self.mn_axis].clone()));
        packed_shape
    }
}
