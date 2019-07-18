use crate::internal::*;
use super::Generic;

impl Op for Generic<TypedTensorInfo, Box<Op>> {
    fn name(&self) -> Cow<str> {
        "Generic".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &Model)> {
        vec!(("loop".into(), &self.body as _))
    }
}

impl StatelessOp for Generic<TypedTensorInfo, Box<Op>> {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let hidden_state_len = inputs.len() - self.num_scan_inputs - self.closure_inputs;

        // extract hidden state original values from inputs
        let mut state: TVec<Tensor> = tvec!();
        for _ in 0..hidden_state_len {
            state.push(inputs.remove(0).into_tensor());
        }

        let first_input_axis = self.scan_input_axes.get(0).cloned().unwrap_or(0);
        let iters = if self.prune_scanning_dim {
            inputs[0].shape()[first_input_axis]
        } else {
            inputs[0].shape()[first_input_axis].div_ceil(
                self.body.input_fact(hidden_state_len)?.shape.dim(first_input_axis).to_integer()?
                    as usize,
            )
        };

        let mut scan_outputs = tvec!();
        for i in 0..(self.body.output_outlets()?.len() - hidden_state_len) {
            let fact = self.body.output_fact(hidden_state_len + i)?;
            let dt = fact.datum_type;
            let mut shape: TVec<usize> = fact.shape.as_finite().unwrap().into();
            let axis = self.scan_output_axes.get(i).cloned().unwrap_or(0);
            let scanning_dim = self
                .scan_output_len_hint
                .get(0)
                .and_then(|x| x.as_ref())
                .and_then(|i| i.to_integer().ok())
                .map(|i| i as usize)
                .unwrap_or(if self.prune_scanning_dim { iters } else { shape[axis] * iters });
            if self.prune_scanning_dim {
                shape.insert(axis, scanning_dim)
            } else {
                shape[axis] = scanning_dim
            }
            let t = dispatch_datum!(Self::alloc_output_t(dt)(self, &*shape))?;
            scan_outputs.push(t);
        }

        let plan = SimplePlan::new(&self.body)?;

        for i in 0..iters {
            // body inputs are state + one slice of each input
            let mut iter_inputs: TVec<Tensor> = state.drain().collect();
            for input in 0..self.num_scan_inputs {
                let fact = self.body.input_fact(input + hidden_state_len)?;
                let axis = self.scan_input_axes.get(input).cloned().unwrap_or(0);
                let count = fact.shape.dim(axis);
                let tensor = dispatch_datum!(Self::slice_input_t(inputs[input].datum_type())(
                    self,
                    &*inputs,
                    input,
                    i,
                    count.to_integer()? as usize
                ))?;
                if cfg!(debug_assert) {
                    fact.to_tensor_fact().unify(&TensorFact::from(tensor.clone()))?;
                }
                iter_inputs.push(tensor);
            }
            for i in 0..self.closure_inputs {
                iter_inputs
                    .push(inputs[inputs.len() - self.closure_inputs + i].clone().into_tensor());
            }

            trace!("iter_inputs: {:?}", iter_inputs);
            let mut iter_outputs = plan.run(iter_inputs).chain_err(|| "Evaluating inner body")?;

            for _ in 0..hidden_state_len {
                state.push(iter_outputs.remove(0).into_tensor());
            }
            for (ix, o) in scan_outputs.iter_mut().enumerate() {
                dispatch_datum!(Self::assign_output_t(o.datum_type())(
                    self,
                    o,
                    ix,
                    &iter_outputs[ix],
                    i
                ))?;
            }
        }
        let mut output: TVec<Arc<Tensor>> =
            state.into_iter().map(|t| t.into_arc_tensor()).collect();
        output.extend(scan_outputs.into_iter().map(|t| t.into_arc_tensor()));
        Ok(output)
    }
}
