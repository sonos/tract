use crate::axes::Axis;
use crate::internal::*;
use ndarray::*;

use tract_linalg::frame::Packer;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatMatMulPack {
    pub(crate) packer: Packer,
    pub(crate) k_axis: usize,
    pub(crate) mn_axis: usize,
    pub(crate) output_shape_fact: ShapeFact,
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

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let output_shape = self.output_shape_fact.eval_to_usize(&session.resolved_symbols)?;
        self.do_eval(&inputs[0], &output_shape)
    }
}

impl TypedOp for MatMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(self.output_shape_fact.iter())))
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

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let output_shape_fact = self.output_shape_fact.eval(values)?.into_owned();
        let inputs: TVec<OutletId> = node.inputs.iter().map(|o| mapping[o]).collect();
        target.wire_node(&node.name, MatMatMulPack { output_shape_fact, ..self.clone() }, &inputs)
    }

    as_op!();
}

impl MatMatMulPack {
    fn do_eval(&self, input: &Tensor, output_shape: &[usize]) -> TractResult<TVec<TValue>> {
        let dt = input.datum_type();
        unsafe {
            let mut packed =
                Tensor::uninitialized_aligned_dt(dt, output_shape, self.packer.alignment())
                    .unwrap();
            if input.rank() == 2 {
                self.packer.pack(&mut packed.view(), input.view(), self.k_axis, self.mn_axis)
            } else {
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
                    let mut prefix: TVec<usize> = coord.slice().into();
                    prefix.remove(self.k_axis.max(self.mn_axis));
                    prefix.remove(self.k_axis.min(self.mn_axis));
                    self.packer.pack(
                        &mut packed.view_at_prefix_mut(&prefix)?,
                        TensorView::from_bytes(input, offset, input.shape(), input.strides()),
                        self.k_axis,
                        self.mn_axis,
                    )
                }
            }
            Ok(tvec!(packed.into_tvalue()))
        }
    }

    pub fn output_shape<D: DimLike>(
        input: &[D],
        packer: &Packer,
        mn_axis: usize,
        k_axis: usize,
    ) -> ShapeFact {
        let mut packed_shape: TVec<D> = input.into();
        packed_shape.remove(mn_axis.max(k_axis));
        packed_shape.remove(mn_axis.min(k_axis));
        packed_shape.push(packer.len(input[k_axis].clone(), input[mn_axis].clone()));
        packed_shape.into()
    }
}
