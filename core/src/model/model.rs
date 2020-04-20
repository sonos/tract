use super::*;
use crate::ops::Op;
use std::fmt;
use std::hash::Hash;

/// Main model class
///
/// Parameterized by a Fact class.
#[derive(Clone, Debug, Educe)]
#[educe(Hash)]
pub struct ModelImpl<F, O>
where
    F: Fact + Hash + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    pub label: Option<String>,
    /// all nodes in the model
    pub nodes: Vec<BaseNode<F, O>>,
    /// index of nodes per name
    #[educe(Hash(ignore))]
    nodes_by_name: HashMap<String, usize>,
    /// model inputs
    pub inputs: Vec<OutletId>,
    /// model outputs
    pub outputs: Vec<OutletId>,
    /// outlet labels
    #[educe(Hash(method="hash_outlet_labels"))]
    pub outlet_labels: HashMap<OutletId, String>,
}

fn hash_outlet_labels<H: std::hash::Hasher>(it: &HashMap<OutletId, String>, state: &mut H) {
    it.iter().sorted().for_each(|ol| ol.hash(state))
}

impl<F, O> DynHash for ModelImpl<F, O>
where
    F: Fact + Hash + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        tract_linalg::hash::dyn_hash(self, hasher)
    }
}

impl<F, O> Default for ModelImpl<F, O>
where
    F: Fact + Hash + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    fn default() -> ModelImpl<F, O> {
        ModelImpl {
            label: None,
            nodes: vec![],
            nodes_by_name: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
            outlet_labels: HashMap::new(),
        }
    }
}

impl<F, O> ModelImpl<F, O>
where
    F: Fact + Hash + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
    ModelImpl<F, O>: Model,
{
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        output_facts: TVec<F>,
    ) -> TractResult<usize> {
        let op = op.into();
        let name = name.into();
        let id = self.nodes.len();
        self.nodes_by_name.insert(name.clone(), id);
        let outputs =
            output_facts.into_iter().map(|fact| OutletFact { fact, successors: tvec!() }).collect();
        let node = BaseNode { id, name, op, inputs: vec![], outputs };
        self.nodes.push(node);
        Ok(id)
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
    pub fn input_fact(&self, ix: usize) -> TractResult<&F> {
        let input = self.input_outlets()?[ix];
        self.outlet_fact(input)
    }

    /// Get the `ix`-th input tensor type information, mutably.
    pub fn input_fact_mut(&mut self, ix: usize) -> TractResult<&mut F> {
        let input = self.input_outlets()?[ix];
        self.outlet_fact_mut(input)
    }

    /// Set the `ix`-th input tensor type information.
    pub fn set_input_fact(&mut self, input: usize, fact: F) -> TractResult<()> {
        let outlet = self.inputs[input];
        self.set_outlet_fact(outlet, fact)
    }

    // Outputs
    /// Get model outputs.
    pub fn output_outlets(&self) -> TractResult<&[OutletId]> {
        Ok(&self.outputs)
    }

    /// Guess outputs from the topology: node or nodes with no successors.
    pub fn auto_outputs(&mut self) -> TractResult<()> {
        let outputs = self
            .nodes
            .iter()
            .flat_map(|n| {
                let id = n.id;
                n.outputs.iter().enumerate().map(move |(ix, output_fact)| {
                    (OutletId::new(id, ix), output_fact.successors.len())
                })
            })
            .filter(|(_f, succs)| *succs == 0)
            .map(|(f, _)| f)
            .collect();
        self.outputs = outputs;
        Ok(())
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
    pub fn output_fact(&self, ix: usize) -> TractResult<&F> {
        let output = self.output_outlets()?[ix];
        self.outlet_fact(output)
    }

    /// Get the `ix`-th input tensor type information, mutably.
    pub fn output_fact_mut(&mut self, ix: usize) -> TractResult<&mut F> {
        let output = self.output_outlets()?[ix];
        self.outlet_fact_mut(output)
    }

    /// Set the `ix`-th output tensor type information.
    pub fn set_output_fact(&mut self, output: usize, fact: F) -> TractResult<()> {
        let outlet = self.outputs[output];
        self.set_outlet_fact(outlet, fact)
    }

    // nodes and their facts

    /// Iterate over all node names.
    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|s| &*s.name)
    }

    /// Find a node by its name.
    pub fn node_by_name<S: AsRef<str>>(&self, name: S) -> TractResult<&BaseNode<F, O>> {
        let id: usize = self.node_id_by_name(name.as_ref())?;
        Ok(&self.nodes[id])
    }

    /// Borrow mutably a node by its name.
    pub fn node_by_name_mut(&mut self, name: &str) -> TractResult<&mut BaseNode<F, O>> {
        let id: &usize =
            self.nodes_by_name.get(name).ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&mut self.nodes[*id])
    }

    pub fn rename_node(&mut self, id: usize, name: &str) -> TractResult<()> {
        self.node_mut(id).name = name.to_string();
        self.nodes_by_name.insert(name.to_string(), id);
        Ok(())
    }

    /// Find a node by its id.
    pub fn node(&self, id: usize) -> &BaseNode<F, O> {
        &self.nodes[id]
    }

    /// Find a node by its id.
    pub fn node_mut(&mut self, id: usize) -> &mut BaseNode<F, O> {
        &mut self.nodes[id]
    }

    /// Access the nodes table.
    pub fn nodes(&self) -> &[BaseNode<F, O>] {
        &*self.nodes
    }

    /// Access the nodes table.
    pub fn nodes_mut(&mut self) -> &mut [BaseNode<F, O>] {
        &mut *self.nodes
    }

    /// Get input and output tensor information for a node.
    pub fn node_facts(&self, id: usize) -> TractResult<(TVec<&F>, TVec<&F>)> {
        Ok((self.node_input_facts(id)?, self.node_output_facts(id)?))
    }

    /// Get input tensor information for a node.
    pub fn node_input_facts(&self, node_id: usize) -> TractResult<TVec<&F>> {
        self.nodes[node_id].inputs.iter().map(|o| self.outlet_fact(*o)).collect()
    }

    /// Get output tensor information for a node.
    pub fn node_output_facts(&self, node_id: usize) -> TractResult<TVec<&F>> {
        Ok(self.nodes[node_id].outputs.iter().map(|o| &o.fact).collect())
    }

    // outlets

    /// Get tensor information for a single outlet.
    pub fn outlet_fact(&self, outlet: OutletId) -> TractResult<&F> {
        let outlets = &self.nodes[outlet.node].outputs;
        outlets
            .get(outlet.slot)
            .map(|o| &o.fact)
            .ok_or_else(|| format!("Invalid outlet reference: {:?}", outlet).into())
    }

    /// Get tensor information for a single outlet.
    pub fn outlet_fact_mut(&mut self, outlet: OutletId) -> TractResult<&mut F> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        outlets
            .get_mut(outlet.slot)
            .map(|o| &mut o.fact)
            .ok_or_else(|| format!("Invalid outlet reference: {:?}", outlet).into())
    }

    /// Get multiple mutable tensor information for outlets.
    pub fn outlets_fact_mut(&mut self, outlets: &[OutletId]) -> TractResult<TVec<&mut F>> {
        assert!(outlets.iter().tuple_combinations().all(|(a, b)| a != b));
        Ok(unsafe {
            outlets
                .iter()
                .map(|o| &mut *(&self.nodes[o.node].outputs[o.slot].fact as *const F as *mut F))
                .collect()
        })
    }

    /// Set tensor information for a single outlet.
    pub fn set_outlet_fact(&mut self, outlet: OutletId, fact: F) -> TractResult<()> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        if outlets.len() <= outlet.slot {
            bail!("Invalid outlet refererence: {:?}", outlet)
        }
        outlets[outlet.slot].fact = fact;
        Ok(())
    }

    // outlet labels

    /// Get label for an outlet.
    pub fn outlet_label(&self, outlet: OutletId) -> Option<&str> {
        self.outlet_labels.get(&outlet).map(|s| &**s)
    }

    /// Set label for an outlet.
    pub fn set_outlet_label(&mut self, outlet: OutletId, label: String) {
        self.outlet_labels.insert(outlet, label);
    }

    /// Find outlet by label.
    pub fn find_outlet_label(&self, label: &str) -> Option<OutletId> {
        self.outlet_labels.iter().find(|(_k, v)| &**v == label).map(|(k, _v)| *k)
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

impl<F, O> Model for ModelImpl<F, O>
where
    F: Fact + Hash + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    fn model_label(&self) -> Option<&str> {
        self.label.as_ref().map(|s| &**s)
    }

    fn node_id_by_name(&self, name: &str) -> TractResult<usize> {
        Ok(self
            .nodes_by_name
            .get(name)
            .ok_or_else(|| format!("No node found for name: \"{}\"", name))
            .map(|x| *x)?)
    }

    fn node_name(&self, id: usize) -> &str {
        &*self.nodes[id].name
    }

    fn node_inputs(&self, id: usize) -> &[OutletId] {
        &*self.nodes[id].inputs
    }

    fn node_output_count(&self, id: usize) -> usize {
        self.nodes[id].outputs.len()
    }

    fn nodes_len(&self) -> usize {
        self.nodes.len()
    }

    fn node_display(&self, id: usize) -> String {
        format!("{}", self.nodes[id])
    }

    fn node_debug(&self, id: usize) -> String {
        format!("{:?}", self.nodes[id])
    }

    fn eval_order(&self) -> TractResult<Vec<usize>> {
        crate::model::eval_order(&self)
    }

    fn eval_order_for_io(&self, inputs: &[usize], outputs: &[usize]) -> TractResult<Vec<usize>> {
        crate::model::order::eval_order_for_nodes(&self.nodes, inputs, outputs, &[])
    }

    fn input_outlets(&self) -> &[OutletId] {
        &*self.inputs
    }

    fn output_outlets(&self) -> &[OutletId] {
        &*self.outputs
    }

    fn node_op(&self, id: usize) -> &dyn Op {
        self.nodes[id].op.as_ref()
    }

    fn outlet_typedfact(&self, outlet: OutletId) -> TractResult<TypedFact> {
        self.outlet_fact(outlet)?.to_typed_fact()
    }

    fn outlet_fact_format(&self, outlet: OutletId) -> String {
        format!("{:?}", self.outlet_fact(outlet).unwrap())
    }

    fn outlet_label(&self, id: OutletId) -> Option<&str> {
        self.outlet_label(id)
    }

    fn outlet_successors(&self, outlet: OutletId) -> &[InletId] {
        &self.nodes[outlet.node].outputs[outlet.slot].successors
    }
}

impl<F, O> fmt::Display for ModelImpl<F, O>
where
    F: Fact + Hash + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.nodes_len() {
            let input_1 =
                self.node_inputs(i).get(0).map(|o| format!("{:?}", o)).unwrap_or("".to_string());
            let input_2 =
                self.node_inputs(i).get(1).map(|o| format!("{:?}", o)).unwrap_or("".to_string());
            let output_1 = self
                .outlet_successors(OutletId::new(i, 0))
                .get(0)
                .map(|o| format!("{:?}", o))
                .unwrap_or("".to_string());
            let output_2 = self
                .outlet_successors(OutletId::new(i, 0))
                .get(1)
                .map(|o| format!("{:?}", o))
                .unwrap_or("".to_string());
            writeln!(
                fmt,
                "{:8} {:8} -> {:5} -> {:8} {:8} | {:15} {}",
                input_1,
                input_2,
                i,
                output_1,
                output_2,
                self.node_op(i).name(),
                self.node_name(i),
            )?;
            if self.node_inputs(i).len() > 2 {
                writeln!(
                    fmt,
                    "                                                |   * inputs: {}",
                    self.node_inputs(i).iter().map(|s| format!("{:?}", s)).join(", ")
                )?;
            }
            if self.node_output_count(i) > 1 || self.outlet_successors((i, 0).into()).len() > 2 {
                for o in 0..self.node_output_count(i) {
                    if self.outlet_successors((i,o).into()).len() > 0 {
                        writeln!(
                            fmt,
                            "                                                |   * output #{}: {}",
                            o,
                            self.outlet_successors((i, o).into())
                                .iter()
                                .map(|s| format!("{:?}", s))
                                .join(", ")
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::internal::*;

    #[test]
    fn hashable() {
        let mut model = TypedModel::default();
        let _s = model.add_source("source", TypedFact::dt_shape(DatumType::F32, [1,2,3].as_ref()).unwrap()).unwrap();
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        model.hash(&mut hasher);
    }
}
