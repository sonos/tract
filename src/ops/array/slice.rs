use ndarray::*;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Slice {
    prune: Vec<(usize, usize)>,
}

impl Slice {
    fn eval_t<T: Datum>(&self, input: Value) -> TfdResult<Value> {
        let input = input.to_array_view::<T>()?;
        let slice_spec: Vec<SliceOrIndex> = self
            .prune
            .iter()
            .map(|&(a, b)| SliceOrIndex::Slice {
                start: a as isize,
                end: if b != 0 { Some(-(b as isize)) } else { None },
                step: 1,
            }).collect();
        let slice_info = SliceInfo::<_, IxDyn>::new(slice_spec).unwrap();
        let slice = input.slice(&slice_info.as_ref());
        Ok(slice.to_owned().into())
    }
}

impl Op for Slice {
    fn name(&self) -> &str {
        "Slice"
    }

    fn pulsify(
        &self,
        mut inputs: TVec<&PulsedTensorFact>,
    ) -> TfdResult<::pulse::PulsifiedOp> {
        let input = args_1!(inputs);
        let axis = input.axis()?;
        println!("Pulsify: {:?}", self);
        if self.prune.iter().enumerate().all(|(ax, &(a,b))| ax == axis || (a == 0 && b == 0)) {
            let delay = self.prune[axis].0;
            let mut fact = input.clone();
            fact.delay += delay;
            return Ok(PulsifiedOp::op(Box::new(::pulse::delay::Delay::new(inputs[0].clone(), delay, 0)), fact))
        }
        unimplemented!();
    }

}

impl StatelessOp for Slice {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(
            self, input
        ))?))
    }
}

impl InferenceRulesOp for Slice {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        for (ix, &(a, b)) in self.prune.iter().enumerate() {
            s.equals(
                &inputs[0].shape[ix],
                outputs[0].shape[ix].bex() + a.to_dim() + b.to_dim(),
            )?;
        }
        Ok(())
    }
}
