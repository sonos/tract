use super::*;
use crate::ops::Op;
use std::fmt;

/// Main model class
///
/// Parameterized by a TensorInfo class.
#[derive(Clone, Debug)]
pub struct Model<TI, O>
where
    TI: TensorInfo,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op>,
{
    /// all nodes in the model
    pub(super) nodes: Vec<BaseNode<TI, O>>,
    /// index of nodes per name
    nodes_by_name: HashMap<String, usize>,
    /// model inputs
    pub(crate) inputs: Vec<OutletId>,
    /// model outputs
    pub(crate) outputs: Vec<OutletId>,
}

impl<TI, O> Default for Model<TI, O>
where
    TI: TensorInfo,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op>,
{
    fn default() -> Model<TI, O> {
        Model { nodes: vec![], nodes_by_name: HashMap::new(), inputs: vec![], outputs: vec![] }
    }
}

impl<TI, O> Model<TI, O>
where
    TI: TensorInfo,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op>,
{
    /// add a node to the model, returning its id
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        output_facts: TVec<TI>,
    ) -> TractResult<usize> {
        self.add_node_disable_output_guess(name, op, output_facts, false)
    }

    pub(crate) fn add_node_disable_output_guess(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        output_facts: TVec<TI>,
        disable_output_guess: bool,
    ) -> TractResult<usize> {
        let op = op.into();
        let name = name.into();
        let id = self.nodes.len();
        self.nodes_by_name.insert(name.clone(), id);
        let noutputs = output_facts.len();
        let outputs =
            output_facts.into_iter().map(|fact| OutletFact { fact, successors: tvec!() }).collect();
        let node = BaseNode { id, name, op, inputs: vec![], control_inputs: vec!(), outputs };
        if node.op_is::<crate::ops::source::Source>() {
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

    pub(crate) fn clear_inputs(&mut self, node: usize) -> TractResult<()> {
        for ix in 0..self.nodes[node].inputs.len() {
            let previous = self.nodes[node].inputs[ix];
            self.nodes[previous.node].outputs[previous.slot]
                .successors
                .retain(|succ| succ.node != node);
        }
        self.nodes[node].inputs.clear();
        Ok(())
    }

    /// Connect a node outlet to a node inlet.
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

    // Inputs

    /// Get model inputs.
    pub fn input_outlets(&self) -> TractResult<&[OutletId]> {
        Ok(&self.inputs)
    }

    /// Change model inputs.
    pub fn set_input_outlets(&mut self, inputs: &[OutletId]) -> TractResult<()> {
        self.inputs = inputs.to_vec();
        Ok(())
    }

    /// Set model inputs by the node name.
    pub fn set_input_names(
        &mut self,
        inputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<()> {
        let mut ids = vec![];
        for i in inputs.into_iter() {
            let node = self.node_by_name(i.as_ref())?;
            for o in 0..node.outputs.len() {
                ids.push(OutletId::new(node.id, o))
            }
        }
        self.inputs = ids;
        Ok(())
    }

    /// Get the `ix`-th input tensor type information.
    pub fn input_fact(&self, ix: usize) -> TractResult<&TI> {
        let input = self.input_outlets()?[ix];
        self.outlet_fact(input)
    }

    /// Get the `ix`-th input tensor type information, mutably.
    pub fn input_fact_mut(&mut self, ix: usize) -> TractResult<&mut TI> {
        let input = self.input_outlets()?[ix];
        self.outlet_fact_mut(input)
    }

    /// Set the `ix`-th input tensor type information.
    pub fn set_input_fact(&mut self, input: usize, fact: TI) -> TractResult<()> {
        let outlet = self.inputs[input];
        self.set_outlet_fact(outlet, fact)
    }

    // Outputs
    /// Get model outputs.
    pub fn output_outlets(&self) -> TractResult<&[OutletId]> {
        Ok(&self.outputs)
    }

    /// Change model outputs.
    pub fn set_output_outlets(&mut self, outputs: &[OutletId]) -> TractResult<()> {
        self.outputs = outputs.to_vec();
        Ok(())
    }

    /// Set model outputs by node names.
    pub fn set_output_names(
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

    /// Get the `ix`-th input tensor type information.
    pub fn output_fact(&self, ix: usize) -> TractResult<&TI> {
        let output = self.output_outlets()?[ix];
        self.outlet_fact(output)
    }

    /// Get the `ix`-th input tensor type information, mutably.
    pub fn output_fact_mut(&mut self, ix: usize) -> TractResult<&mut TI> {
        let output = self.output_outlets()?[ix];
        self.outlet_fact_mut(output)
    }

    /// Set the `ix`-th output tensor type information.
    pub fn set_output_fact(&mut self, output: usize, fact: TI) -> TractResult<()> {
        let outlet = self.outputs[output];
        self.set_outlet_fact(outlet, fact)
    }

    // nodes and their facts

    /// Iterate over all node names.
    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|s| &*s.name)
    }

    /// Find a node by its name.
    pub fn node_by_name(&self, name: &str) -> TractResult<&BaseNode<TI, O>> {
        let id: &usize =
            self.nodes_by_name.get(name).ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&self.nodes[*id])
    }

    /// Borrow mutably a node by its name.
    pub fn node_by_name_mut(&mut self, name: &str) -> TractResult<&mut BaseNode<TI, O>> {
        let id: &usize =
            self.nodes_by_name.get(name).ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&mut self.nodes[*id])
    }

    /// Find a node by its id.
    pub fn node(&self, id: usize) -> &BaseNode<TI, O> {
        &self.nodes[id]
    }

    /// Find a node by its id.
    pub fn node_mut(&mut self, id: usize) -> &mut BaseNode<TI, O> {
        &mut self.nodes[id]
    }

    /// Access the nodes table.
    pub fn nodes(&self) -> &[BaseNode<TI, O>] {
        &*self.nodes
    }

    /// Access the nodes table.
    pub fn nodes_mut(&mut self) -> &mut [BaseNode<TI, O>] {
        &mut *self.nodes
    }

    /// Get input and output tensor information for a node.
    pub fn node_facts(&self, id: usize) -> TractResult<(TVec<&TI>, TVec<&TI>)> {
        Ok((self.node_input_facts(id)?, self.node_output_facts(id)?))
    }

    /// Get input tensor information for a node.
    pub fn node_input_facts(&self, node_id: usize) -> TractResult<TVec<&TI>> {
        self.nodes[node_id].inputs.iter().map(|o| self.outlet_fact(*o)).collect()
    }

    /// Get output tensor information for a node.
    pub fn node_output_facts(&self, node_id: usize) -> TractResult<TVec<&TI>> {
        Ok(self.nodes[node_id].outputs.iter().map(|o| &o.fact).collect())
    }

    // outlets

    /// Get tensor information for a single outlet.
    pub fn outlet_fact(&self, outlet: OutletId) -> TractResult<&TI> {
        let outlets = &self.nodes[outlet.node].outputs;
        outlets.get(outlet.slot).map(|o| &o.fact)
            .ok_or_else(|| format!("Invalid outlet reference: {:?}", outlet).into())
    }

    /// Get tensor information for a single outlet.
    pub fn outlet_fact_mut(&mut self, outlet: OutletId) -> TractResult<&mut TI> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        outlets.get_mut(outlet.slot).map(|o| &mut o.fact)
            .ok_or_else(|| format!("Invalid outlet reference: {:?}", outlet).into())
    }

    /// Set tensor information for a single outlet.
    pub fn set_outlet_fact(&mut self, outlet: OutletId, fact: TI) -> TractResult<()> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        if outlets.len() <= outlet.slot {
            bail!("Invalid outlet refererence: {:?}", outlet)
        }
        outlets[outlet.slot].fact = fact;
        Ok(())
    }

    // misc

    /// Computes an evalutation order for the graph inputs and outputs
    pub fn eval_order(&self) -> TractResult<Vec<usize>> {
        eval_order(&self)
    }

    /// Performs a sanity check on network connections.
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
