use crate::internal::*;

use super::*;

#[derive(Debug, Clone, new, Default)]
pub struct Inference {
    pub body: InferenceModel,
    pub hidden_state_len: usize,
    pub input_mapping: Vec<InputMapping<()>>,
    pub(super) scan_output_axes: Vec<usize>,
    pub(super) scan_output_len_hint: Vec<Option<TDim>>,
}

impl Op for Inference {
    fn name(&self) -> Cow<str> {
        "Scan::Inference".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &Model)> {
        vec![("loop".into(), &self.body)]
    }

    fn to_typed(&self) -> TractResult<Option<Box<Op>>> {
        let typed_model = self.body.clone().into_typed()?;
        let input_mapping = self
            .input_mapping
            .iter()
            .enumerate()
            .map(|(ix, im)| {
                Ok(match im {
                    InputMapping::Scan { axis, slot, chunk } => InputMapping::Scan {
                        axis: *axis,
                        slot: *slot,
                        chunk: typed_model.input_fact(ix)?.shape.dim(*axis),
                    },
                    InputMapping::Full { slot } => InputMapping::Full { slot: *slot },
                    InputMapping::State { initializer } => {
                        InputMapping::State { initializer: initializer.clone() }
                    }
                })
            })
            .collect::<TractResult<_>>()?;
        Ok(Some(Box::new(Typed::new(
            typed_model,
            self.hidden_state_len,
            input_mapping,
            self.scan_output_axes.clone(),
            self.scan_output_len_hint.clone(),
        ))))
    }
}

impl StatelessOp for Inference {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!()
    }
}

impl Inference {
    fn unify_scanning_tensor_fact(
        outer: &mut TensorFact,
        inner: &mut TensorFact,
        outer_scan_axis: usize,
    ) -> TractResult<()> {
        outer.datum_type.unify_with_mut(&mut inner.datum_type)?;
        let rank =
            outer.shape.rank().concretize().or(inner.shape.rank().concretize()).map(|r| r as usize);
        if let Some(rank) = rank {
            outer.shape.unify_with(&ShapeFact::closed(tvec!(GenericFact::Any; rank as usize)))?;
            inner.shape.unify_with(&ShapeFact::closed(tvec!(GenericFact::Any; rank as usize)))?;
            for axis in 0..rank {
                if axis != outer_scan_axis {
                    let value = outer.shape.dim(axis).unwrap().concretize().or(inner
                        .shape
                        .dim(axis)
                        .unwrap()
                        .concretize());
                    if let Some(value) = value {
                        outer.shape.set_dim(axis, value.clone());
                        inner.shape.set_dim(axis, value);
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
        for state_ix in 0..self.hidden_state_len {
            trace!("Unify hidden state #{}", state_ix);
            let (inner_model_input_ix, initializer) = self
                .input_mapping
                .iter()
                .enumerate()
                .filter_map(|(ix, m)| m.as_state().map(|init| (ix, init)))
                .nth(state_ix)
                .unwrap();
            let inner_model_output_ix = state_ix;
            match initializer {
                StateInitializer::Value(v) => {
                    let fact = TensorFact::dt_shape_from_tensor(v);
                    self.body.set_input_fact(inner_model_input_ix, fact.clone())?;
                    self.body.set_output_fact(inner_model_output_ix, fact)?;
                }
                StateInitializer::FromInput(outer_input_ix) => {
                    let mut facts = self.body.outlets_fact_mut(&[
                        self.body.input_outlets()?[inner_model_input_ix],
                        self.body.output_outlets()?[inner_model_output_ix],
                    ])?;
                    facts.push(&mut inputs[*outer_input_ix]);
                    Fact::unify_all(&mut facts)?;
                }
            }
        }
        for (ix, i) in self.input_mapping.iter().enumerate() {
            match i {
                InputMapping::State { .. } => {},
                InputMapping::Full { slot } => {
                    inputs[*slot].unify_with(self.body.input_fact_mut(ix)?)?;
                },
                InputMapping::Scan { slot, axis, .. } => {
                    let incoming = &mut inputs[*slot];
                    let inner = self.body.input_fact_mut(ix)?;
                    Self::unify_scanning_tensor_fact(incoming, inner, *axis)?;
                }
            }
        }
        for i in 0..(self.scan_output_axes.len()) {
            trace!("Unifying scan output #{}", self.hidden_state_len + i);
            let outgoing = &mut outputs[self.hidden_state_len + i];
            let inner = self.body.output_fact_mut(self.hidden_state_len + i)?;
            let axis = self.scan_output_axes[i];
            Self::unify_scanning_tensor_fact(outgoing, inner, axis)?;
        }
        Ok(())
    }
}

impl InferenceOp for Inference {
    fn infer_facts(
        &mut self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
        _observed: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>, TVec<TensorFact>)> {
        let body_inputs = self.body.input_outlets()?.len();
        let body_outputs = self.body.output_outlets()?.len();
        let expected_op_inputs = self.input_mapping.iter().filter(|m| !m.invisible()).count();
        if inputs.len() != expected_op_inputs {
            bail!("Scan receives {} inputs, mappings expects {}", inputs.len(), expected_op_inputs)
        }
        if body_inputs != self.input_mapping.len() {
            bail!("Scan body expect {} inputs, mappings expects {}", body_inputs, self.input_mapping.len())
        }
        if outputs.len() != body_outputs {
            bail!("Scan has {} outputs, inner model expects {}", outputs.len(), body_outputs)
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
