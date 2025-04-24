use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::scan::Scan;
use tract_core::ops::scan::ScanInfo;
pub use tract_core::ops::scan::{InputMapping, OutputMapping};

#[derive(Debug, Clone, new, Default)]
pub struct InferenceScan {
    pub body: InferenceModel,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
    pub clean_scan_counts: bool,
    pub iter_count_fact: GenericFactoid<TDim>,
}

impl Op for InferenceScan {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut lines = vec![];
        for (ix, im) in self.input_mapping.iter().enumerate() {
            lines.push(format!("Model input  #{ix}: {im:?}"));
        }
        for (ix, om) in self.output_mapping.iter().enumerate() {
            lines.push(format!("Model output #{ix}: {om:?}"));
        }
        Ok(lines)
    }

    not_a_typed_op!();
}

impl EvalOp for InferenceScan {
    fn is_stateless(&self) -> bool {
        false
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
                    InputMapping::Scan(info) => InputMapping::Scan(ScanInfo {
                        chunk: typed_model.input_fact(ix)?.shape[info.axis].to_isize()?,
                        ..*info
                    }),
                    other => other.clone(),
                })
            })
            .collect::<TractResult<_>>()?;
        let output_mapping = self
            .output_mapping
            .iter()
            .enumerate()
            .map(|(ix, im)| {
                let scan = if let Some((slot, scan)) = im.scan {
                    Some((
                        slot,
                        ScanInfo {
                            chunk: typed_model.input_fact(ix)?.shape[scan.axis].to_isize()?,
                            ..scan
                        },
                    ))
                } else {
                    None
                };
                Ok(OutputMapping {
                    state: im.state,
                    scan,
                    full_dim_hint: im.full_dim_hint.clone(),
                    last_value_slot: im.last_value_slot,
                })
            })
            .collect::<TractResult<_>>()?;
        Ok(Box::new(Scan::new(typed_model, input_mapping, output_mapping, 0)?))
    }

    fn unify_scanning_tensor_fact(
        outer: &mut InferenceFact,
        inner: &mut InferenceFact,
        outer_scan_axis: usize,
    ) -> TractResult<bool> {
        let mut changed = outer.datum_type.unify_with_mut(&mut inner.datum_type)?;
        let rank = outer
            .shape
            .rank()
            .concretize()
            .or_else(|| inner.shape.rank().concretize())
            .map(|r| r as usize);
        if let Some(rank) = rank {
            if outer.shape.unify_with(&ShapeFactoid::closed(tvec!(GenericFactoid::Any; rank)))? {
                changed = true;
            }
            if inner.shape.unify_with(&ShapeFactoid::closed(tvec!(GenericFactoid::Any; rank)))? {
                changed = true;
            }
            for axis in 0..rank {
                if axis != outer_scan_axis {
                    let value = outer
                        .shape
                        .dim(axis)
                        .unwrap()
                        .concretize()
                        .or_else(|| inner.shape.dim(axis).unwrap().concretize());
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
        let hidden_state_len = self.input_mapping.iter().filter(|m| m.is_state()).count();
        #[allow(clippy::needless_range_loop)]
        for state_ix in 0..hidden_state_len {
            trace!("Unify hidden state #{state_ix}");
            let inner_model_output_ix = self
                .output_mapping
                .iter()
                .enumerate()
                .filter(|(_ix, map)| map.state)
                .nth(state_ix)
                .unwrap()
                .0;
            let mut facts = self.body.outlets_fact_mut(&[
                self.body.input_outlets()?[state_ix],
                self.body.output_outlets()?[inner_model_output_ix],
            ])?;
            facts.push(&mut inputs[state_ix]);
            if Factoid::unify_all(
                &mut facts.iter_mut().map(|f| &mut f.datum_type).collect::<TVec<_>>(),
            )? {
                changed = true;
            }
            if Factoid::unify_all(&mut facts.iter_mut().map(|f| &mut f.shape).collect::<TVec<_>>())?
            {
                changed = true;
            }
        }
        for (slot, i) in self.input_mapping.iter().enumerate() {
            match i {
                InputMapping::State => {}
                InputMapping::Full => {
                    if inputs[slot].unify_with_mut(self.body.input_fact_mut(slot)?)? {
                        changed = true;
                    }
                }
                InputMapping::Scan(scan) => {
                    let incoming = &mut inputs[slot];
                    let inner = self.body.input_fact_mut(slot)?;
                    if Self::unify_scanning_tensor_fact(incoming, inner, scan.axis)? {
                        changed = true;
                    };
                    if self.clean_scan_counts {
                        if incoming.shape.ensure_rank_at_least(scan.axis) {
                            changed = true;
                        }
                        let value =
                            self.iter_count_fact.unify(&incoming.shape.dim(scan.axis).unwrap())?;
                        if self.iter_count_fact != value {
                            changed = true;
                            self.iter_count_fact = value.clone();
                        }
                        if incoming.shape.dim(scan.axis).unwrap() != value {
                            changed = true;
                            incoming.shape.set_dim(scan.axis, value.concretize().unwrap());
                        }
                    }
                }
            }
        }
        for (ix, i) in self.output_mapping.iter().enumerate() {
            if let Some((slot, scan)) = i.scan {
                let outgoing = &mut outputs[slot];
                let inner = self.body.output_fact_mut(ix)?;
                if Self::unify_scanning_tensor_fact(outgoing, inner, scan.axis)? {
                    changed = true
                }
                if self.clean_scan_counts {
                    if outgoing.shape.ensure_rank_at_least(scan.axis) {
                        changed = true;
                    }
                    let value =
                        self.iter_count_fact.unify(&outgoing.shape.dim(scan.axis).unwrap())?;
                    if self.iter_count_fact != value {
                        changed = true;
                        self.iter_count_fact = value.clone();
                    }
                    if outgoing.shape.dim(scan.axis).unwrap() != value {
                        changed = true;
                        outgoing.shape.set_dim(scan.axis, value.concretize().unwrap());
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
        let expected_op_inputs = self.input_mapping.len();
        let expected_op_outputs = self
            .output_mapping
            .iter()
            .filter_map(|om| om.last_value_slot)
            .chain(self.output_mapping.iter().filter_map(|om| om.scan.map(|si| si.0)))
            .max()
            .context("No output slot found")?
            + 1;
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
        target.wire_node(&*node.name, self.to_mir_scan()? as Box<dyn TypedOp>, &inputs)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.output_mapping.iter().filter(|om| !om.invisible()).count())
    }

    as_op!();
}
