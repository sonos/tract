use tract_ndarray::prelude::*;
use tract_nnef::internal::*;

#[derive(Debug, PartialEq, Clone, Hash)]
pub struct OneHot {
    pub axis: usize,
    pub dim: usize,
    pub off: Tensor,
    pub on: Tensor,
}

tract_linalg::impl_dyn_hash!(OneHot);

impl Op for OneHot {
    fn name(&self) -> Cow<str> {
        "MirOnehot".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl TypedOp for OneHot {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.to_tvec();
        shape.insert(self.axis, self.dim.to_dim());
        Ok(tvec!(TypedFact::dt_shape(self.off.datum_type(), &*shape)?))
    }

    as_op!();
}

impl EvalOp for OneHot {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let output = dispatch_datum!(Self::eval_t(self.off.datum_type())(self, &input))?;
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl OneHot {
    fn eval_t<T: Datum + Clone>(&self, input: &Tensor) -> TractResult<Tensor> {
        let off = self.off.to_scalar::<T>()?;
        let on = self.on.to_scalar::<T>()?;
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        let mut array = tract_ndarray::ArrayD::<T>::from_elem(&*shape, off.to_owned());
        let input = input.cast_to::<i32>()?;
        let input = input.to_array_view::<i32>()?;
        dbg!(&input);
        for icoord in tract_ndarray::indices_of(&input) {
            let mut ocoord:Vec<usize> = icoord.slice().into();
            let coord = input[&icoord];
            let coord = if coord < 0 { coord + self.dim as i32 } else { coord } as usize;
            ocoord.insert(self.axis, coord);
            array[&*ocoord] = on.clone();
        }
        Ok(array.into_tensor())
    }
}
