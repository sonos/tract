use tract_data::itertools::Itertools;

use crate::internal::*;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct OneHot {
    pub axis: usize,
    pub dim: usize,
    pub off: Arc<Tensor>,
    pub on: Arc<Tensor>,
}

impl Op for OneHot {
    fn name(&self) -> Cow<str> {
        "Onehot".into()
    }

    op_as_typed_op!();
}

impl TypedOp for OneHot {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.to_tvec();
        shape.insert(self.axis, self.dim.to_dim());
        Ok(tvec!(self.off.datum_type().fact(&*shape)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let axes = (0..inputs[0].rank())
            .zip('a'..)
            .map(|(i, repr)| {
                Axis::new(repr, inputs.len(), outputs.len())
                    .input(0, i)
                    .output(0, i + (i >= self.axis) as usize)
            })
            .chain(std::iter::once(
                Axis::new('Z', inputs.len(), outputs.len()).output(0, self.axis),
            ))
            .collect_vec();
        AxesMapping::new(inputs.len(), outputs.len(), axes)
    }

    as_op!();
}

impl EvalOp for OneHot {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        unsafe {
            let mut output = self.off.broadcast_scalar_to_shape(&shape)?;
            dispatch_datum_by_size!(Self::eval_t(self.off.datum_type())(
                self,
                &input,
                &mut output
            ))?;
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl OneHot {
    unsafe fn eval_t<T: Datum + Clone>(
        &self,
        input: &Tensor,
        output: &mut Tensor,
    ) -> TractResult<()> {
        let on = self.on.to_scalar_unchecked::<T>();
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        let mut array = output.to_array_view_mut_unchecked::<T>();
        let input = input.cast_to::<i32>()?;
        let input = input.to_array_view::<i32>()?;
        for icoord in tract_ndarray::indices_of(&input) {
            use tract_ndarray::Dimension;
            let mut ocoord: Vec<usize> = icoord.slice().into();
            let coord = input[&icoord];
            let coord = if coord < 0 { coord + self.dim as i32 } else { coord } as usize;
            ocoord.insert(self.axis, coord);
            array[&*ocoord] = on.clone();
        }
        Ok(())
    }
}
