use super::codegen::Codegen;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct TypedScan {
    pub skip: usize,
    pub body: TypedModel,
    decluttered: bool,
    pub seq_length_input_slot: Option<usize>,
    pub input_mapping: Vec<InputMapping<TDim>>,
    pub output_mapping: Vec<OutputMapping<TDim, TDim>>,
}

impl TypedScan {
    pub fn to_codegen_op(&self) -> TractResult<Codegen> {
        trace!("Optimizing(Codegen) inner model");
        let plan = SimplePlan::new(self.body.clone().into_optimized()?)?;
        trace!("Optimizing(Codegen) inner model done");
        let input_mapping = self
            .input_mapping
            .iter()
            .map(|im| {
                Ok(match im {
                    InputMapping::Scan { axis, slot, chunk } => InputMapping::Scan {
                        axis: *axis,
                        slot: *slot,
                        chunk: chunk.to_integer()? as usize,
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
            .map(|im| {
                Ok(OutputMapping {
                    state: im.state,
                    axis: im.axis,
                    full_slot: im.full_slot,
                    full_dim_hint: im.full_dim_hint.clone(),
                    last_value_slot: im.last_value_slot,
                    chunk: im.chunk.to_integer()? as usize,
                })
            })
            .collect::<TractResult<_>>()?;

        Ok(Codegen::new(self.skip, Arc::new(plan), input_mapping, output_mapping))
    }

    pub fn new(
        body: TypedModel,
        input_mapping: Vec<InputMapping<TDim>>,
        output_mapping: Vec<OutputMapping<TDim, TDim>>,
        seq_length_input_slot: Option<usize>,
    ) -> TractResult<TypedScan> {
        assert_eq!(input_mapping.len(), body.input_outlets()?.len());
        assert_eq!(output_mapping.len(), body.output_outlets()?.len());
        Ok(TypedScan {
            skip: 0,
            body,
            decluttered: false,
            input_mapping,
            output_mapping,
            seq_length_input_slot,
        })
    }
}

impl Op for TypedScan {
    fn name(&self) -> Cow<str> {
        "Scan::Typed".into()
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

    fn nested_models(&self) -> Vec<(Cow<str>, &dyn Model)> {
        vec![("loop".into(), &self.body)]
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatefullOp for TypedScan {
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.to_codegen_op()?.state(session, node_id)
    }
}

impl TypedOp for TypedScan {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut outputs = tvec!();
        let iters = {
            let (outside_slot, axis, chunk) = self
                .input_mapping
                .iter()
                .filter_map(|it| match it {
                    InputMapping::Scan { axis, slot, chunk } => Some((*slot, *axis, chunk.clone())),
                    _ => None,
                })
                .next()
                .unwrap();
            inputs[outside_slot].shape.dim(axis).div_ceil(chunk.to_dim())
        };
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.body.output_fact(ix)?;
            if let Some(slot) = output.full_slot {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape.dim(output.axis) * &iters);
                shape.set_dim(output.axis, scanning_dim)?;
                outputs.push((slot, TypedFact::dt_shape(fact.datum_type, shape)?));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, TypedFact::dt_shape(fact.datum_type, fact.shape.clone())?));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if !self.decluttered {
            let mut new = self.clone();
            new.body = self.body.clone().declutter()?;
            new.decluttered = true;
            return Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, new)?));
        }
        for (model_input, input) in self.input_mapping.iter().enumerate() {
            if let Some((slot, axis, chunk)) = input.as_scan() {
                let scan_source = self.body.input_outlets()?[model_input];
                let scan_source_node = self.body.node(scan_source.node);
                for successor in &scan_source_node.outputs[0].successors {
                    let successor_node = self.body.node(successor.node);
                    if successor_node.inputs.len() != 1 || successor_node.outputs.len() != 1 {
                        continue;
                    }
                    let invariants = successor_node
                        .op()
                        .as_typed()
                        .unwrap()
                        .invariants(&self.body, &successor_node)?;
                    if let Some(axis_after) = invariants.unary_track_axis_down(axis, false) {
                        let mut outside_patch = TypedModelPatch::default();
                        let mut patch_inputs = node
                            .inputs
                            .iter()
                            .map(|&i| outside_patch.tap_model(model, i))
                            .collect::<TractResult<TVec<_>>>()?;
                        let input = outside_patch.tap_model(&model, node.inputs[slot])?;
                        let new_input_wire = outside_patch.wire_node(
                            format!("{}-extracted-{}", node.name, successor_node.name),
                            objekt::clone_box(successor_node.op().as_typed().unwrap()),
                            &[input],
                        )?[0];
                        patch_inputs.push(new_input_wire);
                        let new_input_outer_fact = outside_patch.outlet_fact(new_input_wire)?;
                        let mut new_input_inner_fact = new_input_outer_fact.clone();
                        new_input_inner_fact.shape.set_dim(axis_after, chunk.clone())?;

                        let mut new_body = self.body.clone();
                        let new_source_wire = new_body.add_source(
                            format!("{}-extracted-{}", node.name, successor_node.name),
                            new_input_inner_fact,
                        )?;
                        let mut inner_patch = TypedModelPatch::default();
                        let new_source_wire_in_patch =
                            inner_patch.tap_model(&new_body, new_source_wire)?;
                        inner_patch.shunt_outside(
                            OutletId::new(successor.node, 0),
                            new_source_wire_in_patch,
                        )?;
                        inner_patch.apply(&mut new_body)?;

                        let mut input_mapping = self.input_mapping.clone();
                        input_mapping.push(InputMapping::Scan {
                            axis: axis_after,
                            chunk: chunk.clone(),
                            slot: node.inputs.len(),
                        });

                        let new_op = Self {
                            input_mapping,
                            output_mapping: self.output_mapping.clone(),
                            decluttered: false,
                            body: new_body,
                            skip: self.skip,
                            seq_length_input_slot: self.seq_length_input_slot,
                        };
                        let output_wires =
                            outside_patch.wire_node(&*node.name, new_op, &patch_inputs)?;
                        for w in output_wires {
                            outside_patch.shunt_outside(OutletId::new(node.id, w.slot), w)?;
                        }
                        return Ok(Some(outside_patch));
                    }
                }
            }
        }
        Ok(None)
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        for input_id in 0..node.inputs.len() {
            let input = mapping[&node.inputs[input_id]];
            let input_fact = target.outlet_fact(input)?;
            let (_slot, axis, _chunk) = self
                .input_mapping
                .iter()
                .filter_map(InputMapping::as_scan)
                .find(|mapping| mapping.0 == input_id)
                .unwrap();
            if input_fact.axis != axis {
                bail!("Scan pulsification limited to scanning axis");
            }
        }

        let pulse_inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();

        let mut op = self.clone();
        op.skip = target.outlet_fact(pulse_inputs[0])?.delay;
        op.output_mapping.iter_mut().find(|om| om.full_slot.is_some()).unwrap().full_dim_hint =
            None;
        target.wire_node(&*node.name, op, &pulse_inputs)
    }

    fn nested_model_multipliers(&self, inputs: &[&TypedFact]) -> Vec<(Cow<str>, f32)> {
        self.to_codegen_op()
            .unwrap()
            .nested_model_multipliers(inputs)
            .into_iter()
            .map(|(c, n)| (c.into_owned().into(), n))
            .collect()
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::replace_single_op(
            &model,
            node,
            &node.inputs,
            self.to_codegen_op()?,
        )?))
    }
}

impl PulsedOp for TypedScan {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let (output_body_ix, output_mapping) = self
            .output_mapping
            .iter()
            .enumerate()
            .find(|(_ix, om)| om.full_slot == Some(0))
            .unwrap();
        let output_body_fact = self.body.output_fact(output_body_ix)?;
        let shape = output_body_fact
            .shape
            .iter()
            .enumerate()
            .map(|(axis, d)| {
                if axis == output_mapping.axis {
                    inputs[0].pulse()
                } else {
                    d.to_integer().unwrap() as usize
                }
            })
            .collect();
        let fact = PulsedFact {
            datum_type: output_body_fact.datum_type,
            shape,
            axis: output_mapping.axis,
            dim: inputs[0].dim.clone(),
            delay: inputs[0].delay,
        };
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
