use crate::ops::einsum::EinSum;
use crate::ops::konst::Const;
use crate::optim::OptimizerSession;

use super::lir::{LirScan, LirScanOpParams};
use tract_data::internal::*;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct Scan {
    pub skip: usize,
    pub body: TypedModel,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
}

impl Scan {
    pub fn to_codegen_op(&self, optimize_inner: bool) -> TractResult<LirScan> {
        let mut model = self.body.clone();
        if optimize_inner {
            model = model.into_optimized()?;
        }
        let plan = SimplePlan::new(model)?;

        Ok(LirScan::new(Arc::new(LirScanOpParams::new(
            self.skip,
            Arc::new(plan),
            self.input_mapping.clone(),
            self.output_mapping.clone(),
        ))))
    }

    pub fn new(
        body: TypedModel,
        input_mapping: Vec<InputMapping>,
        output_mapping: Vec<OutputMapping<TDim>>,
        skip: usize,
    ) -> TractResult<Scan> {
        body.check_consistency()?;
        ensure!(input_mapping.len() == body.input_outlets()?.len());
        ensure!(output_mapping.len() == body.output_outlets()?.len());
        Ok(Scan { skip, body, input_mapping, output_mapping })
    }

    pub fn iteration_count(&self, inputs: &[&TypedFact]) -> Option<TDim> {
        self.to_codegen_op(false).unwrap().iteration_count(inputs)
    }

    fn body_bounds(&self) -> TractResult<TVec<TVec<OutletId>>> {
        let input_state_outlets = self
            .input_mapping
            .iter()
            .zip(self.body.input_outlets()?.iter())
            .filter(|(m, _)| m.is_state())
            .map(|(_, o)| o);
        let output_state_outlets = self
            .output_mapping
            .iter()
            .zip(self.body.output_outlets()?.iter())
            .filter(|(m, _)| m.state)
            .map(|(_, o)| o);
        Ok(input_state_outlets.zip(output_state_outlets).map(|(&i, &o)| tvec!(i, o)).collect())
    }

    fn body_locked_outlets(&self, node_input_facts: &[&TypedFact]) -> TractResult<TVec<OutletId>> {
        let input_outlets =
            self.body.input_outlets()?.iter().enumerate().filter_map(|(slot, o)| {
                if node_input_facts[slot].konst.is_none() {
                    Some(o)
                } else {
                    None
                }
            });
        let output_outlets = self
            .output_mapping
            .iter()
            .zip(self.body.output_outlets()?.iter())
            .filter(|(m, _)| !m.invisible())
            .map(|(_, o)| o);
        Ok(input_outlets.chain(output_outlets).cloned().collect())
    }

    fn try_body_axes_change(
        &self,
        change: AxisChange,
        locked_interface: bool,
        node_input_facts: &[&TypedFact],
    ) -> TractResult<Option<AxisChangeConsequence>> {
        self.body.check_consistency()?;
        let locked_outlets = self.body_locked_outlets(node_input_facts)?;
        let (body_patch, body_changed_wires) = if let Some(changes) =
            crate::optim::change_axes::change_axes(
                &self.body,
                &change,
                if locked_interface { &locked_outlets } else { &[] },
                &self.body_bounds()?,
            )? {
            changes
        } else {
            return Ok(None);
        };
        let mut body = self.body.clone();
        body_patch.apply(&mut body)?;
        body.compact()?;
        let mut wire_changes = tvec!();
        let mut input_mapping: Vec<InputMapping> = self.input_mapping.clone();
        for (slot, m) in input_mapping.iter_mut().enumerate() {
            if let Some(change) = body_changed_wires
                .iter()
                .find(|(iface, _change)| iface == &InOut::In(slot))
                .map(|pair| pair.1.clone())
            {
                wire_changes.push((InOut::In(slot), change.clone()));
                if let InputMapping::Scan(info) = m {
                    if let Some(axis) = change.transform_axis(info.axis) {
                        info.axis = axis;
                    } else {
                        return Ok(None);
                    };
                };
            }
        }
        let mut output_mapping: Vec<OutputMapping<TDim>> = self.output_mapping.clone();
        for (ix, m) in output_mapping.iter_mut().enumerate() {
            if let Some(change) = body_changed_wires
                .iter()
                .find(|(iface, _change)| iface == &InOut::Out(ix))
                .map(|pair| pair.1.clone())
            {
                if let Some((slot, info)) = m.scan.as_mut() {
                    if let Some(new_axis) = change.transform_axis(info.axis) {
                        info.axis = new_axis;
                    } else {
                        return Ok(None);
                    }
                    wire_changes.push((InOut::Out(*slot), change.clone()));
                }
                if let Some(slot) = m.last_value_slot {
                    wire_changes.push((InOut::Out(slot), change.clone()));
                }
            };
        }
        body.check_consistency()?;
        let op = Some(Box::new(Scan { body, input_mapping, output_mapping, ..self.clone() }) as _);
        Ok(Some(AxisChangeConsequence { substitute_op: op, wire_changes }))
    }
}

impl Op for Scan {
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

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for Scan {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.to_codegen_op(false)?.state(session, node_id)
    }
}

impl TypedOp for Scan {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == self.body.inputs.len());
        anyhow::ensure!(self.input_mapping.len() == self.body.inputs.len());
        anyhow::ensure!(
            self.input_mapping.iter().filter(|m| m.is_state()).count()
                == self.output_mapping.iter().filter(|m| m.state).count()
        );
        for (i, o) in
            self.input_mapping.iter().enumerate().filter(|(_, m)| m.is_state()).map(|(i, _)| i).zip(
                self.output_mapping.iter().enumerate().filter(|(_, m)| m.state).map(|(o, _)| o),
            )
        {
            anyhow::ensure!(
                self.body.outlet_fact(self.body.inputs[i])?
                    == self.body.outlet_fact(self.body.outputs[o])?
            )
        }
        let mut outputs = tvec!();
        let iters = super::iteration_count(&self.input_mapping, inputs).context("No scan input")?;
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.body.output_fact(ix)?;
            if let Some((slot, info)) = output.scan {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape[info.axis].clone() * &iters);
                shape.set(info.axis, scanning_dim);
                outputs.push((slot, fact.datum_type.fact(shape)));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, fact.datum_type.fact(fact.shape.clone())));
            }
        }
        outputs.sort_by_key(|a| a.0);
        anyhow::ensure!(outputs.iter().enumerate().all(|(ix, (slot, _))| ix == *slot));
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut mappings = vec![];
        let body_invs = self.body.axes_mapping().with_context(|| "Computing body axes mapping")?;
        for body_axis in body_invs.iter_all_axes() {
            let mut info = Axis::new(body_axis.repr, inputs.len(), outputs.len());
            info.inputs = body_axis.inputs.clone();
            for (ix, output_mapping) in self.output_mapping.iter().enumerate() {
                let mut slots = vec![];
                if let Some((slot, _scan)) = output_mapping.scan {
                    slots.push(slot);
                }
                if let Some(slot) = output_mapping.last_value_slot {
                    slots.push(slot);
                }
                for slot in slots {
                    info.outputs[slot] = body_axis.outputs[ix].clone();
                }
            }
            if info.inputs.iter().any(|i| i.len() > 0) || info.outputs.iter().any(|i| i.len() > 0) {
                mappings.push(info);
            }
        }
        AxesMapping::new(inputs.len(), outputs.len(), mappings)
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        let mut suggestions = tvec!();
        for (slot, input) in self.input_mapping.iter().enumerate() {
            if let InputMapping::Scan(info) = input {
                if info.axis != 0 {
                    suggestions.push((InOut::In(slot), AxisOp::Move(info.axis, 0)))
                }
            }
        }
        for output in &self.output_mapping {
            if let Some((slot, scan)) = output.scan {
                if scan.axis != 0 {
                    suggestions.push((InOut::Out(slot), AxisOp::Move(scan.axis, 0)))
                }
            }
        }
        Ok(suggestions)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        trace!("Propagating through {}: {:?} {:?}", node, io, change);
        let body_leading_outlet = match io {
            InOut::In(ix) => self.body.input_outlets()?[ix],
            InOut::Out(slot) => {
                let output = self
                    .output_mapping
                    .iter()
                    .position(|im| {
                        im.scan.map(|(slot, _i)| slot) == Some(slot)
                            || im.last_value_slot == Some(slot)
                    })
                    .unwrap();
                self.body.output_outlets()?[output]
            }
        };
        let axis_change = AxisChange { outlet: body_leading_outlet, op: change.clone() };
        let node_input_facts = model.node_input_facts(node.id)?;
        let result = self
            .try_body_axes_change(axis_change, false, &node_input_facts)
            .with_context(|| "Attemping to run change through scan body".to_string())?;
        if result.is_some() {
            trace!("{} accepted axis change", node);
        } else {
            trace!("{} rejected axis change", node);
        }
        Ok(result)
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|o| mapping[o]).collect::<TVec<_>>();
        let op = Self {
            output_mapping: self
                .output_mapping
                .iter()
                .map(|om| om.concretize_dims(values))
                .collect::<TractResult<Vec<_>>>()?,
            body: self.body.concretize_dims(values)?,
            ..self.clone()
        };
        target.wire_node(&node.name, op, &inputs)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs,
            self.to_codegen_op(true)?,
        )?))
    }
}
