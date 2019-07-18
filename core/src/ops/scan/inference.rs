use crate::internal::*;
use super::Generic;

impl Op for Generic<TensorFact, Box<InferenceOp>> {
    fn name(&self) -> Cow<str> {
        "Inference".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &Model)> {
        vec!(("loop".into(), &self.body as _))
    }

    fn to_typed(&self) -> TractResult<Option<Box<Op>>> {
        let typed_model = self.body.clone().into_typed()?;
        Ok(Some(Box::new(Generic::new(
            typed_model,
            self.num_scan_inputs,
            self.closure_inputs,
            self.scan_input_axes.clone(),
            self.scan_output_axes.clone(),
            self.scan_output_len_hint.clone(),
            self.prune_scanning_dim,
        ))))
    }
}

impl StatelessOp for Generic<TensorFact, Box<InferenceOp>> {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!()
    }
}


impl Generic<TensorFact, Box<InferenceOp>> {
    fn unify_scanning_tensor_fact(
        outer: &mut TensorFact,
        inner: &mut TensorFact,
        outer_scan_axis: usize,
        prune_scanning_dim: bool,
    ) -> TractResult<()> {
        outer.datum_type.unify_with_mut(&mut inner.datum_type)?;
        let outer_rank = outer
            .shape
            .rank()
            .concretize()
            .or(inner.shape.rank().concretize().map(|r| r - prune_scanning_dim as usize as i32))
            .map(|r| r as usize);
        if let Some(outer_rank) = outer_rank {
            let inner_rank = outer_rank - prune_scanning_dim as usize;
            outer
                .shape
                .unify_with(&ShapeFact::closed(tvec!(GenericFact::Any; outer_rank as usize)))?;
            inner
                .shape
                .unify_with(&ShapeFact::closed(tvec!(GenericFact::Any; inner_rank as usize)))?;
            for outer_axis in 0..outer_rank {
                if outer_axis != outer_scan_axis {
                    let inner_axis =
                        outer_axis - (prune_scanning_dim && outer_axis > outer_scan_axis) as usize;
                    let value = outer.shape.dim(outer_axis).unwrap().concretize().or(inner
                        .shape
                        .dim(inner_axis)
                        .unwrap()
                        .concretize());
                    if let Some(value) = value {
                        outer.shape.set_dim(outer_axis, value.clone());
                        inner.shape.set_dim(inner_axis, value);
                    }
                }
            }
        }
        Ok(())
    }

    fn unify_facts(
        &mut self,
        inputs: &mut [TensorFact],
        outputs: &mut [TensorFact],
    ) -> TractResult<()> {
        let body_inputs = self.body.input_outlets()?.len();
        let body_outputs = self.body.output_outlets()?.len();
        let hidden_state_len = body_inputs - self.num_scan_inputs - self.closure_inputs;
        let num_scan_outputs = body_outputs - hidden_state_len;
        for i in 0..hidden_state_len {
            trace!("Unify hidden state #{}", i);
            let mut merged =
                self.body.input_fact(i)?.datum_type.unify(&self.body.output_fact(i)?.datum_type)?;
            Fact::unify_all(&mut [
                &mut merged,
                &mut inputs[i].datum_type,
                &mut outputs[i].datum_type,
            ])
            .map_err(|e| format!("while unifying hidden state datum_types #{}: {}", i, e))?;
            self.body.input_fact_mut(i)?.datum_type.unify_with(&mut merged)?;
            self.body.output_fact_mut(i)?.datum_type.unify_with(&mut merged)?;

            let mut merged =
                self.body.input_fact(i)?.shape.unify(&self.body.output_fact(i)?.shape)?;
            Fact::unify_all(&mut [&mut merged, &mut inputs[i].shape, &mut outputs[i].shape])
                .map_err(|e| format!("while unifying hidden state shapes #{}: {}", i, e))?;
            self.body.input_fact_mut(i)?.shape.unify_with(&mut merged)?;
            self.body.output_fact_mut(i)?.shape.unify_with(&mut merged)?;
        }
        for i in 0..self.num_scan_inputs {
            trace!("Unifying scan input #{}", hidden_state_len + i);
            let incoming = &mut inputs[hidden_state_len + i];
            let inner = self.body.input_fact_mut(hidden_state_len + i)?;
            let axis = self.scan_input_axes.get(i).cloned().unwrap_or(0);
            Self::unify_scanning_tensor_fact(incoming, inner, axis, self.prune_scanning_dim)?;
        }
        for i in 0..self.closure_inputs {
            let id = hidden_state_len + self.num_scan_inputs + i;
            trace!("Unifying closure input #{}", id);
            inputs[id].unify_with(self.body.input_fact_mut(id)?)?;
        }
        for i in 0..num_scan_outputs {
            trace!("Unifying scan output #{}", hidden_state_len + i);
            let outgoing = &mut outputs[hidden_state_len + i];
            let inner = self.body.output_fact_mut(hidden_state_len + i)?;
            let axis = self.scan_output_axes.get(i).cloned().unwrap_or(0);
            Self::unify_scanning_tensor_fact(outgoing, inner, axis, self.prune_scanning_dim)?;
        }
        Ok(())
    }
}

impl InferenceOp for Generic<TensorFact, Box<InferenceOp>> {
    fn infer_facts(
        &mut self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
        _observed: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>, TVec<TensorFact>)> {
        let body_inputs = self.body.input_outlets()?.len();
        let body_outputs = self.body.output_outlets()?.len();
        if inputs.len() != body_inputs {
            bail!("Generic receives {} inputs, inner model expects {}", inputs.len(), body_inputs)
        }
        if outputs.len() != body_outputs {
            bail!("Generic has {} outputs, inner model expects {}", outputs.len(), body_outputs)
        }
        let mut inputs: TVec<TensorFact> = inputs.into_iter().cloned().collect();
        let mut outputs: TVec<TensorFact> = outputs.into_iter().cloned().collect();
        self.unify_facts(&mut inputs, &mut outputs)?;
        trace!("Starting inner model analyse");
        for (ix, input) in self.body.input_outlets()?.iter().enumerate() {
            trace!("  Input inner model: {} {:?} {:?}", ix, input, self.body.input_fact(ix));
        }
        for (ix, output) in self.body.output_outlets()?.iter().enumerate() {
            trace!("  Output inner model: {} {:?} {:?}", ix, output, self.body.output_fact(ix));
        }
        self.body
            .analyse(false)
            .map_err(|e| format!("analysing inner model: {}\n{:#?}", e, self.body))?;
        trace!("Finished inner model analyse");
        self.unify_facts(&mut inputs, &mut outputs)?;
        Ok((inputs, outputs, tvec!()))
    }

    inference_op_as_op!();
}
