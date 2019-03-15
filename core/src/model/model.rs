use super::*;
use crate::ops::Op;

#[derive(Clone, Debug)]
pub struct Model<TI: TensorInfo> {
    pub(super) nodes: Vec<Node<TI>>,
    nodes_by_name: HashMap<String, usize>,
    pub(super) inputs: Vec<OutletId>,
    pub(super) outputs: Vec<OutletId>,
}

impl<TI: TensorInfo> Default for Model<TI> {
    fn default() -> Model<TI> {
        Model {
            nodes: vec![],
            nodes_by_name: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
        }
    }
}

impl<TI: TensorInfo> Model<TI> {

    pub fn add_node(
        &mut self,
        name: String,
        op: Box<Op>,
        outputs_fact: TVec<TI>,
    ) -> TractResult<usize> {
        self.add_node_disable_output_guess(name, op, outputs_fact, false)
    }

    pub(crate) fn add_node_disable_output_guess(
        &mut self,
        name: String,
        op: Box<Op>,
        outputs_fact: TVec<TI>,
        disable_output_guess: bool,
    ) -> TractResult<usize> {
        let id = self.nodes.len();
        self.nodes_by_name.insert(name.clone(), id);
        let is_input = op.name() == "Source";
        let noutputs = op.noutputs();
        let outputs =
            outputs_fact.into_iter().map(|fact| OutletFact { fact, successors: tvec!() }).collect();
        let node = Node { id, name, op, inputs: vec![], outputs };
        if is_input {
            self.inputs.push(OutletId::new(id, 0));
        }
        if !disable_output_guess {
            for o in 0..noutputs {
                self.outputs.push(OutletId::new(id, o));
            }
        }
        self.nodes.push(node);
        Ok(id)
    }

    pub fn clear_inputs(&mut self, node: usize) -> TractResult<()> {
        for ix in 0..self.nodes[node].inputs.len() {
            let previous = self.nodes[node].inputs[ix];
            self.nodes[previous.node].outputs[previous.slot]
                .successors
                .retain(|succ| succ.node != node);
        }
        self.nodes[node].inputs.clear();
        Ok(())
    }

    pub fn add_edge(&mut self, outlet: OutletId, inlet: InletId) -> TractResult<()> {
        if let Some(previous) = self.nodes[inlet.node].inputs.get(inlet.slot).cloned() {
            self.nodes[previous.node].outputs[previous.slot]
                .successors
                .retain(|&mut succ| succ != inlet);
        }
        {
            let prec = &mut self.nodes[outlet.node];
            prec.outputs[outlet.slot].successors.push(inlet);
            self.outputs.retain(|&o| o != outlet);
        }
        let succ = &mut self.nodes[inlet.node];
        if inlet.slot == succ.inputs.len() {
            succ.inputs.push(outlet);
        } else if inlet.slot < succ.inputs.len() {
            succ.inputs[inlet.slot] = outlet;
        } else {
            bail!("Edges must be added in order and consecutive. Trying to connect input {:?} of node {:?} ", inlet.slot, succ)
        }
        Ok(())
    }

    pub fn set_inputs(
        &mut self,
        inputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<()> {
        use crate::ops::source::Source;
        let ids: Vec<OutletId> = inputs
            .into_iter()
            .map(|s| self.node_by_name(s.as_ref()).map(|n| OutletId::new(n.id, 0)))
            .collect::<TractResult<_>>()?;
        self.inputs = ids;
        for &i in &self.inputs {
            self.nodes[i.node].inputs.clear();
            self.nodes[i.node].op = Box::new(Source::default());
        }
        Ok(())
    }

    pub fn set_outputs(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<()> {
        let ids: Vec<OutletId> = outputs
            .into_iter()
            .map(|s| self.node_by_name(s.as_ref()).map(|n| OutletId::new(n.id, 0)))
            .collect::<TractResult<_>>()?;
        self.outputs = ids;
        Ok(())
    }

    pub fn set_outputs_outlets(&mut self, outputs: &[OutletId]) -> TractResult<()> {
        self.outputs = outputs.to_vec();
        Ok(())
    }

    pub fn set_fact(&mut self, outlet: OutletId, fact: TI) -> TractResult<()> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        if outlets.len() <= outlet.slot {
            bail!("Invalid outlet refererence: {:?}", outlet)
        }
        outlets[outlet.slot].fact = fact;
        Ok(())
    }

    pub fn set_input_fact(&mut self, input: usize, fact: TI) -> TractResult<()> {
        let outlet = self.inputs()?[input];
        self.set_fact(outlet, fact)
    }

    pub fn facts(&self, id: usize) -> TractResult<(TVec<&TI>, TVec<&TI>)> {
        let node = &self.nodes[id];

        let inputs: TVec<&TI> = node
            .inputs
            .iter()
            .enumerate()
            .map(|(ix, outlet)| (ix, outlet, self.fact(*outlet).unwrap()))
            .inspect(|(ix, outlet, fact)| {
                trace!("Input {} from {:?}: {:?}", ix, outlet, fact);
            })
            .map(|(_, _, fact)| fact)
            .collect();

        let outputs = node
            .outputs
            .iter()
            .map(|outlet| &outlet.fact)
            .enumerate()
            .inspect(|(ix, fact)| trace!("Output {}: {:?}", ix, fact))
            .map(|(_ix, f)| f)
            .collect();

        Ok((inputs, outputs))
    }

    pub fn eval_order(&self) -> TractResult<Vec<usize>> {
        eval_order(&self)
    }

    pub fn node_input_facts(&self, node_id: usize) -> TractResult<TVec<&TI>> {
        self.nodes[node_id].inputs.iter().map(|o| self.fact(*o)).collect()
    }

    pub fn node_output_facts(&self, node_id: usize) -> TractResult<TVec<&TI>> {
        Ok(self.nodes[node_id].outputs.iter().map(|o| &o.fact).collect())
    }

    pub fn node_by_name(&self, name: &str) -> TractResult<&Node<TI>> {
        let id: &usize =
            self.nodes_by_name.get(name).ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&self.nodes[*id])
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| &*s.name).collect()
    }

    pub fn node(&self, id: usize) -> &Node<TI> {
        &self.nodes[id]
    }

    pub fn node_mut(&mut self, id: usize) -> &mut Node<TI> {
        &mut self.nodes[id]
    }

    pub fn nodes(&self) -> &[Node<TI>] {
        &*self.nodes
    }

    pub fn mut_nodes(&mut self) -> &mut [Node<TI>] {
        &mut *self.nodes
    }

    pub fn fact(&self, outlet: OutletId) -> TractResult<&TI> {
        let outlets = &self.nodes[outlet.node].outputs;
        Ok(&outlets[outlet.slot].fact)
    }

    pub fn inputs_fact(&self, ix: usize) -> TractResult<&TI> {
        let input = self.inputs()?[ix];
        self.fact(input)
    }

    pub fn input_fact(&self) -> TractResult<&TI> {
        self.inputs_fact(0)
    }

    pub fn inputs(&self) -> TractResult<&[OutletId]> {
        Ok(&self.inputs)
    }

    pub fn outputs_fact(&self, ix: usize) -> TractResult<&TI> {
        let output = self.outputs()?[ix];
        self.fact(output)
    }

    pub fn output_fact(&self) -> TractResult<&TI> {
        self.outputs_fact(0)
    }

    pub fn outputs(&self) -> TractResult<&[OutletId]> {
        Ok(&self.outputs)
    }

    pub fn into_arc(self) -> Arc<Model<TI>> {
        Arc::new(self)
    }

    pub fn check_edges(&self) -> TractResult<()> {
        for node in self.eval_order()? {
            let node = &self.nodes[node];
            for (ix, input) in node.inputs.iter().enumerate() {
                let prec = &self.nodes[input.node];
                if !prec.outputs[input.slot].successors.contains(&InletId::new(node.id, ix)) {
                    bail!(
                        "Mismatched oncoming edge, node:{} input:{} to {:?} not reciprocated",
                        node.id,
                        ix,
                        prec
                    )
                }
            }
            for (ix, output) in node.outputs.iter().enumerate() {
                for succ in &output.successors {
                    if self.nodes[succ.node].inputs[succ.slot] != OutletId::new(node.id, ix) {
                        bail!(
                            "Mismatched outgoing edge, node:{} output:{} to {:?} not reciprocated",
                            node.id,
                            ix,
                            succ
                        )
                    }
                }
            }
        }
        Ok(())
    }
}

