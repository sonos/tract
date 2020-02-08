use crate::internal::*;

#[derive(Debug, Clone, new, Default)]
pub struct MultiBroadcastTo {
    shape: TVec<TDim>,
}

impl MultiBroadcastTo {
    pub fn eval_t<T: Datum>(input: &Tensor, shape: &[usize]) -> TractResult<TVec<Arc<Tensor>>> {
        let input = input.to_array_view::<T>()?;
        let output = input.broadcast(&*shape).ok_or("incompatible shapes")?;
        Ok(tvec![output.to_owned().into_arc_tensor()])
    }
}

impl Op for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }

    canonic!();
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for MultiBroadcastTo {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let dims: Vec<usize> =
            self.shape.iter().map(|d| Ok(d.to_integer()? as usize)).collect::<TractResult<_>>()?;
        dispatch_datum!(Self::eval_t(input.datum_type())(&*input, &*dims))
    }
}

impl TypedOp for MultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.shape)?))
    }

    as_op!();
}
