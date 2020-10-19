use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::scan::Scan;
pub use tract_core::ops::scan::{InputMapping, OutputMapping, StateInitializer};

#[derive(Debug, Clone, new, Default, Hash)]
pub struct InferenceScan {
    pub body: InferenceModel,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
    pub seq_length_input_slot: Option<usize>,
    pub clean_scan_counts: bool,
    pub iter_count_fact: GenericFactoid<TDim>,
}

tract_data::impl_dyn_hash!(InferenceScan);

impl Op for InferenceScan {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut lines = vec![];
        for (ix, im) in self.input_mapping.iter().enumerate() {
            lines.push(format!("Model input  #{}: {:?}", ix, im));
        }
        for (ix, om) in self.output_mapping.iter().enumerate() {
            lines.push(format!("Model output #{}: {:?}", ix, om));
        }
        Ok(lines)
    }

    op_hir!();
    not_a_typed_op!();
}

impl EvalOp for InferenceScan {
    fn is_stateless(&self) -> bool {
        true
    }

    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.to_mir_scan()?.state(session, node_id)
    }
}

impl InferenceScan {
    pub(super) fn to_mir_scan(&self) -> TractResult<Box<Scan>> {
        let typed_model = self.body.clone().into_typed()?;
        let input_mapping = self
            .input_mapping
            .iter()
            .enumerate()
            .map(|(ix, im)| {
                Ok(match im {
                    InputMapping::Scan { axis, slot, chunk: _ } => InputMapping::Scan {
                        axis: *axis,
                        slot: *slot,
                        chunk: typed_model.input_fact(ix)?.shape[*axis].to_isize()?,
                    },
                    InputMapping::Full { slot } => InputMapping::Full { slot: *slot },
                    InputMapping::State { initializer } => {
                        InputMapping::State { initializer: initializer.clone() }
                    }
                })
            })
            .collect::<TractResult<_>>()?;
        let output_mapping = self
            .output_mapping
            .iter()
            .enumerate()
            .map(|(ix, im)| {
                Ok(OutputMapping {
                    state: im.state,
                    axis: im.axis,
                    full_slot: im.full_slot,
                    full_dim_hint: im.full_dim_hint.clone(),
                    last_value_slot: im.last_value_slot,
                    chunk: typed_model.input_fact(ix)?.shape[im.axis].to_isize()?,
                })
            })
            .collect::<TractResult<_>>()?;
        Ok(Box::new(Scan::new(
            typed_model,
            input_mapping,
            output_mapping,
            self.seq_length_input_slot,
            0,
        )?))
    }

    fn unify_scanning_tensor_fact(
        outer: &mut InferenceFact,
        inner: &mut InferenceFact,
        outer_scan_axis: usize,
    ) -> TractResult<bool> {
        let mut changed = outer.datum_type.unify_with_mut(&mut inner.datum_type)?;
        let rank =
            outer.shape.rank().concretize().or(inner.shape.rank().concretize()).map(|r| r as usize);
        if let Some(rank) = rank {
            if outer
                .shape
                .unify_with(&ShapeFactoid::closed(tvec!(GenericFactoid::Any; rank as usize)))?
            {
                changed = true;
            }
            if inner
                .shape
                .unify_with(&ShapeFactoid::closed(tvec!(GenericFactoid::Any; rank as usize)))?
            {
                changed = true;
            }
            for axis in 0..rank {
                if axis != outer_scan_axis {
                    let value = outer.shape.dim(axis).unwrap().concretize().or(inner
                        .shape
                        .dim(axis)
                        .unwrap()
                        .concretize());
                    if let Some(value) = value {
                        if outer.shape.set_dim(axis, value.clone()) {
                            changed = true
                        }
                        if inner.shape.set_dim(axis, value) {
                            changed = true
                        }
                    }
                }
            }
        }
        Ok(changed)
    }

    fn unify_facts(
        &mut self,
        inputs: &mut [InferenceFact],
        outputs: &mut [InferenceFact],
    ) -> TractResult<bool> {
        let mut changed = false;
        let hidden_state_len = self.input_mapping.iter().filter_map(|m| m.as_state()).count();
        for state_ix in 0..hidden_state_len {
            trace!("Unify hidden state #{}", state_ix);
            let (inner_model_input_ix, initializer) = self
                .input_mapping
                .iter()
                .enumerate()
                .filter_map(|(ix, m)| m.as_state().map(|init| (ix, init)))
                .nth(state_ix)
                .unwrap();
            let inner_model_output_ix = self
                .output_mapping
                .iter()
                .enumerate()
                .filter(|(_ix, map)| map.state)
                .nth(state_ix)
                .unwrap()
                .0;
            match initializer {
                StateInitializer::Value(v) => {
                    let fact = InferenceFact::dt_shape_from_tensor(v);
                    if self.body.input_fact(inner_model_input_ix)? != &fact {
                        self.body.set_input_fact(inner_model_input_ix, fact.clone())?;
                        changed = true;
                    }
                    if self.body.output_fact(inner_model_input_ix)? != &fact {
                        self.body.set_output_fact(inner_model_output_ix, fact)?;
                        changed = true;
                    }
                }
                StateInitializer::FromInput(outer_input_ix) => {
                    let mut facts = self.body.outlets_fact_mut(&[
                        self.body.input_outlets()?[inner_model_input_ix],
                        self.body.output_outlets()?[inner_model_output_ix],
                    ])?;
                    facts.push(&mut inputs[*outer_input_ix]);
                    if Factoid::unify_all(
                        &mut *facts.iter_mut().map(|f| &mut f.datum_type).collect::<TVec<_>>(),
                    )? {
                        changed = true;
                    }
                    if Factoid::unify_all(
                        &mut *facts.iter_mut().map(|f| &mut f.shape).collect::<TVec<_>>(),
                    )? {
                        changed = true;
                    }
                }
            }
        }
        for (ix, i) in self.input_mapping.iter().enumerate() {
            match i {
                InputMapping::State { .. } => {}
                InputMapping::Full { slot } => {
                    if inputs[*slot].unify_with_mut(self.body.input_fact_mut(ix)?)? {
                        changed = true;
                    }
                }
                InputMapping::Scan { slot, axis, .. } => {
                    let incoming = &mut inputs[*slot];
                    let inner = self.body.input_fact_mut(ix)?;
                    if Self::unify_scanning_tensor_fact(incoming, inner, *axis)? {
                        changed = true;
                    };
                    if self.clean_scan_counts {
                        if incoming.shape.ensure_rank_at_least(*axis) {
                            changed = true;
                        }
                        let value =
                            self.iter_count_fact.unify(&incoming.shape.dim(*axis).unwrap())?;
                        if self.iter_count_fact != value {
                            changed = true;
                            self.iter_count_fact = value.clone();
                        }
                        if incoming.shape.dim(*axis).unwrap() != value {
                            changed = true;
                            incoming.shape.set_dim(*axis, value.concretize().unwrap());
                        }
                    }
                }
            }
        }
        for (ix, i) in self.output_mapping.iter().enumerate() {
            if let Some(slot) = i.full_slot {
                let outgoing = &mut outputs[slot];
                let inner = self.body.output_fact_mut(ix)?;
                if Self::unify_scanning_tensor_fact(outgoing, inner, i.axis)? {
                    changed = true
                }
                if self.clean_scan_counts {
                    if outgoing.shape.ensure_rank_at_least(i.axis) {
                        changed = true;
                    }
                    let value = self.iter_count_fact.unify(&outgoing.shape.dim(i.axis).unwrap())?;
                    if self.iter_count_fact != value {
                        changed = true;
                        self.iter_count_fact = value.clone();
                    }
                    if outgoing.shape.dim(i.axis).unwrap() != value {
                        changed = true;
                        outgoing.shape.set_dim(i.axis, value.concretize().unwrap());
                    }
                }
            }
            if let Some(slot) = i.last_value_slot {
                if outputs[slot].unify_with(self.body.output_fact_mut(ix)?)? {
                    changed = true;
                }
            }
        }
        Ok(changed)
    }
}

impl InferenceOp for InferenceScan {
    fn infer_facts(
        &mut self,
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        _observed: TVec<&InferenceFact>,
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>, TVec<InferenceFact>)> {
        let body_inputs = self.body.input_outlets()?.len();
        let body_outputs = self.body.output_outlets()?.len();
        let expected_op_inputs = self.input_mapping.iter().filter(|m| !m.invisible()).count();
        let expected_op_outputs = self.output_mapping.iter().filter(|m| !m.invisible()).count();
        if inputs.len() != expected_op_inputs {
            bail!("Scan receives {} inputs, mappings expects {}", inputs.len(), expected_op_inputs)
        }
        if body_inputs != self.input_mapping.len() {
            bail!(
                "Scan body expect {} inputs, mappings expects {}",
                body_inputs,
                self.input_mapping.len()
            )
        }
        if outputs.len() != expected_op_outputs {
            bail!("Scan has {} outputs, mappings expects {}", outputs.len(), expected_op_outputs);
        }
        if body_outputs != self.output_mapping.len() {
            bail!(
                "Scan body expect {} outputs, mappings expects {}",
                body_outputs,
                self.output_mapping.len()
            )
        }
        let mut inputs: TVec<InferenceFact> = inputs.into_iter().cloned().collect();
        let mut outputs: TVec<InferenceFact> = outputs.into_iter().cloned().collect();
        loop {
            trace!("Unify inner and outer interface");
            let mut changed = self.unify_facts(&mut inputs, &mut outputs)?;
            trace!("iters: {:?} changed: {:?}", self.iter_count_fact, changed);
            for (ix, input) in self.body.input_outlets()?.iter().enumerate() {
                trace!("  Input inner model: {} {:?} {:?}", ix, input, self.body.input_fact(ix));
            }
            for (ix, output) in self.body.output_outlets()?.iter().enumerate() {
                trace!("  Output inner model: {} {:?} {:?}", ix, output, self.body.output_fact(ix));
            }
            trace!("Inner model analyse");
            if self.body.analyse(false).context("analysing inner model")? {
                changed = true;
            }
            if !changed {
                break;
            }
            trace!("Finished inner model analyse");
        }
        Ok((inputs, outputs, tvec!()))
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|m| mapping[m]).collect::<TVec<_>>();
        target.wire_node(&*node.name, self.to_mir_scan()? as Box<dyn TypedOp>, &*inputs)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.output_mapping.iter().filter(|om| !om.invisible()).count())
    }

    as_op!();
}
