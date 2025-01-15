use super::*;
use crate::internal::*;
use crate::ops::Op;
use crate::plan::PlanOptions;
use crate::prelude::*;

use std::fmt;
use tract_data::internal::*;
use tract_itertools::Itertools;

pub trait SpecialOps<F, O> {
    fn create_dummy(&self) -> O;
    fn create_source(&self, fact: F) -> O;
    fn is_source(op: &O) -> bool;
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>>;
    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId>;
}

/// Main model class
///
/// Parameterized by a Fact class.
#[derive(Clone, Debug)]
pub struct Graph<F, O>
where
    F: Fact + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    /// all nodes in the model
    pub nodes: Vec<Node<F, O>>,
    /// model inputs
    pub inputs: Vec<OutletId>,
    /// model outputs
    pub outputs: Vec<OutletId>,
    /// outlet labels
    pub outlet_labels: HashMap<OutletId, String>,
    /// model properties
    pub properties: HashMap<String, Arc<Tensor>>,
    /// symbol scope, including table
    pub symbols: SymbolScope,
}

impl<F, O> Default for Graph<F, O>
where
    F: Fact + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn default() -> Graph<F, O> {
        Graph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            outlet_labels: HashMap::new(),
            properties: HashMap::new(),
            symbols: Default::default(),
        }
    }
}

impl<F, O> Graph<F, O>
where
    F: Fact + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Graph<F, O>: SpecialOps<F, O>,
{
    pub fn add_source(&mut self, name: impl Into<String>, fact: F) -> TractResult<OutletId> {
        let source = self.create_source(fact.clone());
        let id = self.add_node(name, source, tvec!(fact))?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }
}

impl<F, O> Graph<F, O>
where
    F: Fact + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
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
        let outputs =
            output_facts.into_iter().map(|fact| Outlet { fact, successors: tvec!() }).collect();
        let node = Node { id, name, op, inputs: vec![], outputs };
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
        #[allow(clippy::comparison_chain)]
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

    /// Change model inputs and return `self`.
    pub fn with_input_outlets(mut self, inputs: &[OutletId]) -> TractResult<Self> {
        self.set_input_outlets(inputs)?;
        Ok(self)
    }

    /// Set model inputs by the node name.
    pub fn set_input_names(
        &mut self,
        inputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<()> {
        let mut ids = vec![];
        for i in inputs.into_iter() {
            let node = self.node_by_name(&i)?;
            for o in 0..node.outputs.len() {
                ids.push(OutletId::new(node.id, o))
            }
        }
        self.inputs = ids;
        Ok(())
    }

    /// Set model inputs by the node name and return `self`.
    pub fn with_input_names(
        mut self,
        inputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<Self> {
        self.set_input_names(inputs)?;
        Ok(self)
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

    /// Set the `ix`-th input tensor type information and return `self`.
    pub fn with_input_fact(mut self, input: usize, fact: F) -> TractResult<Self> {
        self.set_input_fact(input, fact)?;
        Ok(self)
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

    /// Change model outputs and return `self`.
    pub fn with_output_outlets(mut self, outputs: &[OutletId]) -> TractResult<Self> {
        self.set_output_outlets(outputs)?;
        Ok(self)
    }

    /// Set model outputs by node names.
    pub fn set_output_names(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<()> {
        let mut labels: HashMap<Cow<str>, OutletId> =
            self.outlet_labels.iter().map(|(o, s)| (Cow::Borrowed(&**s), *o)).collect();
        for n in self.nodes() {
            for ix in 0..n.outputs.len() {
                labels.insert(Cow::Owned(format!("{}:{}", &n.name, ix)), OutletId::new(n.id, ix));
            }
        }
        let ids: Vec<OutletId> = outputs
            .into_iter()
            .map(|s| {
                let s = s.as_ref();
                labels
                    .get(s)
                    .cloned()
                    .or_else(|| self.nodes.iter().find(|n| n.name == s).map(|n| n.id.into()))
                    .ok_or_else(|| format_err!("Node {} not found", s))
            })
            .collect::<TractResult<_>>()?;
        self.outputs = ids;
        Ok(())
    }

    /// Set model outputs by node names and return `self`.
    pub fn with_output_names(
        mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<Self> {
        self.set_output_names(outputs)?;
        Ok(self)
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

    /// Set the `ix`-th output tensor type information and return `self`.
    pub fn with_output_fact(mut self, output: usize, fact: F) -> TractResult<Self> {
        self.set_output_fact(output, fact)?;
        Ok(self)
    }

    // nodes and their facts

    /// Iterate over all node names.
    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|s| &*s.name)
    }

    pub fn node_id_by_name(&self, name: &str) -> TractResult<usize> {
        self.nodes
            .iter()
            .find(|n| n.name == name)
            .map(|n| n.id)
            .with_context(|| format!("No node found for name: \"{name}\""))
    }

    /// Find a node by its name.
    pub fn node_by_name(&self, name: impl AsRef<str>) -> TractResult<&Node<F, O>> {
        let id: usize = self.node_id_by_name(name.as_ref())?;
        Ok(&self.nodes[id])
    }

    /// Borrow mutably a node by its name.
    pub fn node_by_name_mut(&mut self, name: impl AsRef<str>) -> TractResult<&mut Node<F, O>> {
        let id: usize = self.node_id_by_name(name.as_ref())?;
        Ok(&mut self.nodes[id])
    }

    pub fn rename_node(&mut self, id: usize, name: &str) -> TractResult<()> {
        self.node_mut(id).name = name.to_string();
        Ok(())
    }

    /// Find a node by its id.
    pub fn node(&self, id: usize) -> &Node<F, O> {
        &self.nodes[id]
    }

    /// Find a node by its id.
    pub fn node_mut(&mut self, id: usize) -> &mut Node<F, O> {
        &mut self.nodes[id]
    }

    /// Access the nodes table.
    pub fn nodes(&self) -> &[Node<F, O>] {
        &self.nodes
    }

    /// Access the nodes table.
    pub fn nodes_mut(&mut self) -> &mut [Node<F, O>] {
        &mut self.nodes
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
        ensure!(outlet.node < self.nodes.len(), "Invalid outlet for graph");
        let outlets = &self.nodes[outlet.node].outputs;
        outlets
            .get(outlet.slot)
            .map(|o| &o.fact)
            .with_context(|| format!("Invalid outlet reference: {outlet:?}"))
    }

    /// Get tensor information for a single outlet.
    pub fn outlet_fact_mut(&mut self, outlet: OutletId) -> TractResult<&mut F> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        outlets
            .get_mut(outlet.slot)
            .map(|o| &mut o.fact)
            .with_context(|| format!("Invalid outlet reference: {outlet:?}"))
    }

    /// Get multiple mutable tensor information for outlets.
    pub fn outlets_fact_mut(&mut self, outlets: &[OutletId]) -> TractResult<TVec<&mut F>> {
        assert!(outlets.iter().tuple_combinations().all(|(a, b)| a != b));
        unsafe {
            outlets
                .iter()
                .map(|o| Ok((self.outlet_fact(*o)? as *const F as *mut F).as_mut().unwrap()))
                .collect()
        }
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

    /// Set tensor information for a single outlet and return `self`.
    pub fn with_outlet_fact(mut self, outlet: OutletId, fact: F) -> TractResult<Self> {
        self.set_outlet_fact(outlet, fact)?;
        Ok(self)
    }

    // outlet labels

    /// Get label for an outlet.
    pub fn outlet_label(&self, outlet: OutletId) -> Option<&str> {
        self.outlet_labels.get(&outlet).map(|s| &**s)
    }

    /// Set label for an outlet.
    pub fn set_outlet_label(&mut self, outlet: OutletId, label: String) -> TractResult<()> {
        self.outlet_labels.insert(outlet, label);
        Ok(())
    }

    /// Set label for an outlet and return `self`.
    pub fn with_outlet_label(mut self, outlet: OutletId, label: String) -> TractResult<Self> {
        self.set_outlet_label(outlet, label)?;
        Ok(self)
    }

    /// Find outlet by label.
    pub fn find_outlet_label(&self, label: &str) -> Option<OutletId> {
        self.outlet_labels.iter().find(|(_k, v)| **v == label).map(|(k, _v)| *k)
    }

    // misc

    /// Computes an evalutation order for the graph inputs and outputs
    pub fn eval_order(&self) -> TractResult<Vec<usize>> {
        super::order::eval_order(self)
    }

    /// Computes an evalutation order for the graph inputs and outputs. This order will minimize
    /// temporary buffers.
    pub fn eval_order_opt_ram(&self) -> TractResult<Vec<usize>> {
        super::order::eval_order_opt_ram(self)
    }

    #[cfg(not(all(debug_assertions, feature = "paranoid_assertions")))]
    #[inline]
    pub fn check_edges(&self) -> TractResult<()> {
        Ok(())
    }

    /// Performs a sanity check on network connections.
    #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
    pub fn check_edges(&self) -> TractResult<()> {
        for node_id in self.eval_order()? {
            let node = &self.nodes[node_id];
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

    /// Evaluate temporary memory usage with its related node at each step of the given order.
    pub fn eval_tmp_memory_usage<Flushable>(
        &self,
        order: &[usize],
        flushable: Flushable,
    ) -> TractResult<TVec<(usize, TDim)>>
    where
        Flushable: Fn(&Node<F, O>) -> bool,
    {
        super::memory::eval_tmp_memory_usage(self, order, flushable)
    }

    #[cfg(not(all(debug_assertions, feature = "paranoid_assertions")))]
    #[inline]
    pub fn check_names(&self) -> TractResult<()> {
        Ok(())
    }

    /// Performs a sanity check on network connections.
    #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
    pub fn check_names(&self) -> TractResult<()> {
        let dups =
            self.eval_order()?.iter().map(|n| &self.nodes[*n].name).duplicates().collect_vec();
        ensure!(dups.len() == 0, "Duplicate node name(s) : {:?}\n{}", dups, &self);
        Ok(())
    }

    /// Converts the model into a `RunnableModel` to actually process user data.
    pub fn into_runnable(self) -> TractResult<RunnableModel<F, O, Self>> {
        crate::plan::SimplePlan::new_with_options(self, &PlanOptions::default())
    }

    /// Converts the model into a `RunnableModel` to actually process user data. This variant
    /// accepts options.
    pub fn into_runnable_with_options(
        self,
        options: &PlanOptions,
    ) -> TractResult<RunnableModel<F, O, Self>> {
        crate::plan::SimplePlan::new_with_options(self, options)
    }

    pub fn single_prec(&self, id: usize) -> TractResult<Option<&Node<F, O>>> {
        let node = &self.nodes()[id];
        if node.inputs.len() != 1 {
            return Ok(None);
        }
        let prec = &self.nodes()[node.inputs[0].node];
        if prec.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
            return Ok(None);
        }
        Ok(Some(prec))
    }

    pub fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<F, O>>> {
        let mut node = self.node(id);
        for _ in 0..count {
            if let Some(next) = self.single_prec(node.id)? {
                node = next
            } else {
                return Ok(None);
            }
        }
        Ok(Some(node))
    }

    pub fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<F, O>>> {
        let mut node = self.node(id);
        for _ in 0..count {
            if let Some(next) = self.single_succ(node.id)? {
                node = next
            } else {
                return Ok(None);
            }
        }
        Ok(Some(node))
    }

    /// single_succ is only intended for optimisation of simple operators
    /// with 1 output, and only 1 output successors (successor with only 1 input)
    pub fn single_succ(&self, id: usize) -> TractResult<Option<&Node<F, O>>> {
        let node = &self.nodes()[id];

        if node.outputs.len() != 1 || node.outputs[0].successors.len() != 1 {
            return Ok(None);
        }
        let succ = node.outputs[0].successors[0];
        let succ = &self.nodes()[succ.node];
        if succ.inputs.len() != 1 {
            return Ok(None);
        }
        Ok(Some(succ))
    }

    pub fn outlet_successors(&self, outlet: OutletId) -> &[InletId] {
        &self.nodes[outlet.node].outputs[outlet.slot].successors
    }

    /// retrieve of create a symbol
    pub fn sym(&self, s: &str) -> Symbol {
        self.symbols.sym(s)
    }

    /// create a new symbol with the prefix
    pub fn new_sym_with_prefix(&self, prefix: &str) -> Symbol {
        self.symbols.new_with_prefix(prefix)
    }

    /// generates a name for a new node in the model that will not conflict (by suffixing with a
    /// dot and number)
    pub fn unique_name<'n>(&self, prefix: impl Into<Cow<'n, str>>) -> Cow<'n, str> {
        let prefix = prefix.into();
        if self.nodes.iter().all(|n| n.name != *prefix) {
            return prefix;
        }
        for i in 1.. {
            let s = format!("{prefix}.{i}");
            if self.nodes.iter().all(|n| n.name != s) {
                return Cow::Owned(s);
            }
        }
        unreachable!();
    }
}

impl<F, O> fmt::Display for Graph<F, O>
where
    F: Fact + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.nodes.len() {
            let input_1 =
                self.nodes[i].inputs.first().map(|o| format!("{o:?}")).unwrap_or_default();
            let input_2 = self.nodes[i].inputs.get(1).map(|o| format!("{o:?}")).unwrap_or_default();
            let successors = self.nodes[i]
                .outputs
                .first()
                .iter()
                .flat_map(|o| o.successors.iter())
                .collect_vec();
            let output_1 = successors.first().map(|o| format!("{o:?}")).unwrap_or_default();
            let output_2 = successors.get(1).map(|o| format!("{o:?}")).unwrap_or_default();
            writeln!(
                fmt,
                "{:5} | {:8} {:8} -> {:8} {:8} | {:25} {:50} {} => {}",
                i,
                input_1,
                input_2,
                output_1,
                output_2,
                self.nodes[i].op().name(),
                self.nodes[i].name,
                self.node_input_facts(i).unwrap().iter().map(|f| format!("{f:?}")).join(" ; "),
                self.node_output_facts(i).unwrap().iter().map(|f| format!("{f:?}")).join(" ; "),
            )?;
            if self.nodes[i].inputs.len() > 2 {
                writeln!(
                    fmt,
                    "                                               |   * inputs: {}",
                    self.nodes[i].inputs.iter().map(|s| format!("{s:?}")).join(", ")
                )?;
            }
            if self.nodes[i].outputs.len() > 1
                || successors.len() > 2
                || (self.outlet_label(i.into()).is_some()
                    && self.outlet_label(i.into()).unwrap() != self.nodes[i].name)
            {
                for o in 0..self.nodes[i].outputs.len() {
                    if self.outlet_successors((i, o).into()).len() > 0 {
                        writeln!(
                                    fmt,
                                    "                                               |   * output #{}: {} {}",
                                    o,
                                    self.outlet_label((i, o).into()).unwrap_or(""),
                                    self.outlet_successors((i, o).into())
                                    .iter()
                                    .map(|s| format!("{s:?}"))
                                    .join(", "),
                                    )?;
                    }
                }
            }
        }
        writeln!(fmt, "outputs: {}", self.outputs.iter().map(|o| format!("{o:?}")).join(", "))?;
        Ok(())
    }
}

impl<F, O> Graph<F, O>
where
    F: Fact + Clone + 'static + for<'a> std::convert::From<&'a F>,
    O: std::fmt::Display
        + std::fmt::Debug
        + Clone
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        + 'static
        + for<'a> std::convert::From<&'a O>,
    Graph<F, O>: SpecialOps<F, O>,
{
    #[cfg(debug_assertions)]
    pub fn check_compact(&self) -> TractResult<()> {
        let order = self.eval_order()?;
        let useless_sources = self
            .input_outlets()?
            .iter()
            .filter(|io| {
                self.outlet_successors(**io).len() == 0
                    && !self.output_outlets().unwrap().contains(io)
            })
            .count();
        if order.len() + useless_sources != self.nodes.len() {
            bail!(
                "Eval order is {} long, nodes are {}, including {} unused sources",
                order.len(),
                self.nodes.len(),
                useless_sources
            );
        }
        if (0..order.len()).any(|ix| order[ix] != ix) {
            bail!("eval order is not trivial");
        }
        let mut seen = std::collections::HashSet::new();
        for (ix, n) in self.nodes.iter().enumerate() {
            if ix != n.id {
                bail!("Invalid node id: position is {}, node is {}", ix, n);
            }
            if seen.contains(&n.name) {
                bail!("duplicate name {}", n.name);
            }
            seen.insert(&n.name);
        }
        Ok(())
    }

    pub fn compact(&mut self) -> TractResult<()> {
        let mut order = self.eval_order()?;
        if order.len() == self.nodes.len() && order.iter().enumerate().all(|(a, b)| a == *b) {
            return Ok(());
        }
        for i in &self.inputs {
            if !order.contains(&i.node) {
                order.push(i.node);
            }
        }
        let mut old_to_new = vec![usize::MAX; self.nodes.len()];
        let mut new_nodes = vec![
            Node {
                id: self.nodes.len(),
                name: "".to_string(),
                inputs: vec![],
                op: self.create_dummy(),
                outputs: tvec!(),
            };
            order.len()
        ];
        for (ix, id) in order.iter().enumerate() {
            old_to_new[*id] = ix;
            std::mem::swap(&mut new_nodes[ix], &mut self.nodes[*id]);
        }
        for node in &mut new_nodes {
            if self.inputs.iter().any(|n| n.node == node.id) && !Self::is_source(&node.op) {
                node.inputs.clear();
                node.op = self.create_source(node.outputs[0].fact.clone());
            }
            node.id = old_to_new[node.id];
            for input in &mut node.inputs {
                assert!(old_to_new[input.node] < order.len());
                input.node = old_to_new[input.node];
            }
            for output in &mut node.outputs {
                for succ in &mut output.successors {
                    succ.node = old_to_new[succ.node];
                }
                output.successors.retain(|s| s.node < order.len());
                output.successors.sort();
            }
        }
        self.nodes = new_nodes;
        for input in &mut self.inputs {
            assert!(old_to_new[input.node] < order.len());
            input.node = old_to_new[input.node];
        }
        for output in &mut self.outputs {
            assert!(old_to_new[output.node] < order.len());
            output.node = old_to_new[output.node];
        }
        self.outlet_labels = std::mem::take(&mut self.outlet_labels)
            .into_iter()
            .map(|(k, v)| (OutletId::new(old_to_new[k.node], k.slot), v))
            .filter(|(k, _)| k.node < order.len())
            .collect();
        ensure!(self.nodes.iter().enumerate().all(|(ix, n)| n.id == ix));
        #[cfg(debug_assertions)]
        {
            self.check_compact().context("after graph compaction")?;
        }
        Ok(())
    }

    pub fn into_compact(mut self) -> TractResult<Self> {
        self.compact()?;
        Ok(self)
    }
}
