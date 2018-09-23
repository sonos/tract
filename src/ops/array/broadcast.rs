use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct MultiBroadcastTo;

impl MultiBroadcastTo {
    fn eval_t<T: Datum>(input: &Tensor, shape: &[usize]) -> TfdResult<TVec<Value>> {
        let input = input.to_array_view::<T>()?;
        let output = input.broadcast(&*shape).ok_or("incompatible shapes")?;
        Ok(tvec![output.to_owned().into()])
    }
}

impl Op for MultiBroadcastTo {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (input, dims) = args_2!(inputs);
        let dims: Vec<usize> = dims
            .to_array_view::<i64>()?
            .iter()
            .map(|i| *i as usize)
            .collect();
        let dims = ::broadcast::multi_broadcast(&[&*dims, &*input.shape()])
            .ok_or("incompatible shapes")?;
        match input.datum_type() {
            DatumType::Bool => Self::eval_t::<bool>(input.as_tensor(), &*dims),
            DatumType::U8 => Self::eval_t::<u8>(input.as_tensor(), &*dims),
            DatumType::U16 => Self::eval_t::<u16>(input.as_tensor(), &*dims),
            DatumType::I8 => Self::eval_t::<i8>(input.as_tensor(), &*dims),
            DatumType::I16 => Self::eval_t::<i16>(input.as_tensor(), &*dims),
            DatumType::I32 => Self::eval_t::<i32>(input.as_tensor(), &*dims),
            DatumType::I64 => Self::eval_t::<i64>(input.as_tensor(), &*dims),
            DatumType::F32 => Self::eval_t::<f32>(input.as_tensor(), &*dims),
            DatumType::F64 => Self::eval_t::<f64>(input.as_tensor(), &*dims),
            DatumType::TDim => Self::eval_t::<TDim>(input.as_tensor(), &*dims),
            DatumType::String => bail!("String is not a Datum")
        }
    }
}

impl InferenceRulesOp for MultiBroadcastTo {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 2)
            .equals(&outputs.len, 1)
            .equals(&inputs[1].datum_type, DatumType::I64)
            .equals(&outputs[0].datum_type, &inputs[0].datum_type)
            .equals(&inputs[1].rank, 1)
            .given(&inputs[0].shape, move |solver, shape| {
                solver.given(&inputs[1].value, move |solver, dims| {
                    let dims: Vec<TDim> = dims
                        .to_array_view::<i64>()
                        .unwrap()
                        .iter()
                        .map(|i| TDim::from(*i))
                        .collect();
                    let dims = ::broadcast::multi_broadcast(&[&*dims, &*shape])
                        .ok_or("incompatible shapes").unwrap();
                    solver.equals(&outputs[0].shape, ShapeFact::from(dims));
                });
            });
    }
}
