use analyser::rules::prelude::*;
use ops::prelude::*;
use tensor::Datum;
use TfdResult;

#[derive(Debug, Clone, new)]
pub struct AddN {
    datum: DatumType,
    n: usize,
}

impl AddN {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T:Datum>(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let mut result = inputs.pop().unwrap().into_array::<T>()?; // checked, non empty
        for input in &inputs[0..] {
            result += &input.to_array_view()?;
        }
        Ok(tvec![result.into()])
    }
}

impl Op for AddN {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        if inputs.len() != self.n || self.n == 0 {
            bail!("Expected {} inputs", self.n);
        }
        let dt = inputs[0].datum_type();
        match dt {
            DatumType::F32 => self.eval_t::<f32>(inputs),
            DatumType::F64 => self.eval_t::<f64>(inputs),
            DatumType::I8 => self.eval_t::<i8>(inputs),
            DatumType::I32 => self.eval_t::<i32>(inputs),
            DatumType::TDim => self.eval_t::<TDim>(inputs),
            DatumType::U8 => self.eval_t::<u8>(inputs),
            DatumType::String => bail!("AddN do not support Strings")
        }
    }

    /// Returns a new streaming buffer for the operation.
    fn new_buffer(&self) -> Box<OpBuffer> {
        Box::new(QueuesBuffer::new(self.n))
    }

    fn step(
        &self,
        inputs: TVec<StepValue>,
        buffer: &mut Box<OpBuffer>,
    ) -> TfdResult<Option<TVec<Value>>> {
        let buffer = buffer
            .downcast_mut::<QueuesBuffer>()
            .ok_or("The buffer can't be downcasted to QueuesBuffer.")?;

        buffer.append(inputs)?;

        if buffer.iter().any(|q| q.is_empty()) {
            Ok(None)
        } else {
            let chunks = buffer
                .iter_mut()
                .map(|b| b.pop_front().unwrap())
                .collect::<TVec<_>>();

            Ok(Some(self.eval(chunks)?))
        }
    }
}

impl InferenceRulesOp for AddN {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        let n = self.n as isize;
        solver
            .equals(&inputs.len, n)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datum_type, &outputs[0].datum_type)
            .equals_all((0..self.n).map(|i| (&inputs[i].datum_type).bex()).collect())
            .equals(&inputs[0].rank, &outputs[0].rank)
            .equals_all((0..self.n).map(|i| inputs[i].rank.bex()).collect())
            .given(&inputs[0].rank, move |solver, rank: isize| {
                for dim in 0..(rank as usize) {
                    solver.equals(&inputs[0].shape[dim], &outputs[0].shape[dim]);
                    solver.equals_all(
                        (0..n as usize)
                            .map(|i| inputs[i].shape[dim].bex())
                            .collect(),
                    );
                }
            });
    }
}
