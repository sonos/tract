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

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.do_eval(&inputs[0])
    }
}

impl TypedOp for MatMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(Opaque::datum_type().fact(self.output_shape(&inputs[0].shape))))
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
        axes.push(Axis::new('P', 1, 1).output(0, outputs[0].rank()));
        AxesMapping::new(1, 1, axes)
    }

    as_op!();
}

impl MatMatMulPack {
    fn do_eval(&self, input: &Tensor) -> TractResult<TVec<TValue>> {
        unsafe {
            let output_shape: TVec<usize> = self.output_shape(input.shape());
            let stores = if output_shape.iter().all(|d| *d == 1) {
                tensor0::<Opaque>(self.packer.pack_tensor(input, self.k_axis, self.mn_axis)?.into())
                    .into_shape(output_shape)?
            } else {
                let mut stores = Tensor::uninitialized_dt(Opaque::datum_type(), output_shape)?;
                let mut stores_view = stores.to_array_view_mut::<Opaque>()?;
                let mut bc_shape: TVec<usize> = input.shape().into();
                bc_shape[self.k_axis] = 1;
                bc_shape[self.mn_axis] = 1;

                for coord in indices(&*bc_shape) {
                    let offset = coord
                        .as_array_view()
                        .iter()
                        .zip(input.strides())
                        .map(|(x, s)| *x as isize * s)
                        .sum::<isize>()
                        * input.datum_type().size_of() as isize;
                    let mut pack_coords: TVec<usize> = coord.slice().into();
                    pack_coords.remove(self.k_axis.max(self.mn_axis));
                    pack_coords.remove(self.k_axis.min(self.mn_axis));
                    stores_view[&*pack_coords] = self
                        .packer
                        .pack_tensor_view(
                            &TensorView::from_bytes(input, offset, input.shape(), input.strides()),
                            self.k_axis,
                            self.mn_axis,
                        )?
                        .into();
                }
                stores
            };
            Ok(tvec!(stores.into_tvalue()))
        }
    }

    pub fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut packed_shape: TVec<D> = input.into();
        packed_shape.remove(self.mn_axis.max(self.k_axis));
        packed_shape.remove(self.mn_axis.min(self.k_axis));
        packed_shape
    }
}
