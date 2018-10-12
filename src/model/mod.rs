use std::collections::HashMap;
use std::str;
use std::sync::Arc;

use bit_set;

mod dsl;
mod order;
pub use self::order::eval_order;
pub use analyser::types::TensorFact;

use {ops, TfdResult};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub inputs: Vec<OutletId>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: Box<ops::Op>,
    pub outputs: TVec<OutletFact>,
}

impl Node {
    pub fn op(&self) -> &ops::Op {
        &*self.op
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletFact {
    pub fact: TensorFact,
    pub successors: Vec<InletId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletId {
    pub node: usize,
    pub slot: usize,
}

impl OutletId {
    pub fn new(node: usize, slot: usize) -> OutletId {
        OutletId { node, slot }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct InletId {
    pub node: usize,
    pub slot: usize,
}

impl InletId {
    pub fn new(node: usize, slot: usize) -> InletId {
        InletId { node, slot }
    }
}

pub type TVec<T> = ::smallvec::SmallVec<[T; 4]>;

/// Model is Tfdeploy workhouse.
#[derive(Clone, Debug, Default)]
pub struct Model {
    nodes: Vec<Node>,
    nodes_by_name: HashMap<String, usize>,
    inputs: Vec<OutletId>,
    outputs: Vec<OutletId>,
}

impl Model {
    pub fn add_node(&mut self, name: String, op: Box<ops::Op>) -> TfdResult<usize> {
        let id = self.nodes.len();
        self.nodes_by_name.insert(name.clone(), id);
        let is_input = op.name() == "Source";
        let node = Node {
            id,
            name,
            op,
            inputs: vec![],
            outputs: tvec!(OutletFact::default()),
        };
        if is_input {
            self.inputs.push(OutletId::new(id, 0));
        }
        self.outputs.push(OutletId::new(id, 0));
        self.nodes.push(node);
        Ok(id)
    }

    pub fn add_edge(&mut self, outlet: OutletId, inlet: InletId) -> TfdResult<()> {
        {
            let prec = &mut self.nodes[outlet.node];
            while prec.outputs.len() <= outlet.slot {
                prec.outputs.push(OutletFact::default());
            }
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
    ) -> TfdResult<()> {
        let ids: Vec<OutletId> = inputs
            .into_iter()
            .map(|s| {
                self.node_by_name(s.as_ref())
                    .map(|n| OutletId::new(n.id, 0))
            }).collect::<TfdResult<_>>()?;
        self.inputs = ids;
        Ok(())
    }

    pub fn set_outputs(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TfdResult<()> {
        let ids: Vec<OutletId> = outputs
            .into_iter()
            .map(|s| {
                self.node_by_name(s.as_ref())
                    .map(|n| OutletId::new(n.id, 0))
            }).collect::<TfdResult<_>>()?;
        self.outputs = ids;
        Ok(())
    }

    pub fn set_fact(&mut self, outlet: OutletId, fact: TensorFact) -> TfdResult<()> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        if outlets.len() <= outlet.slot {
            outlets.push(OutletFact::default());
        }
        outlets[outlet.slot].fact = fact;
        Ok(())
    }

    pub fn analyse(&mut self) -> TfdResult<()> {
        ::analyser::Analyser::new(self)?.analyse()
    }

    pub fn reduce(&mut self) -> TfdResult<()> {
        for id in self.eval_order()? {
            let reduced = {
                let node = &self.nodes[id];
                let input_facts: TVec<&TensorFact> = node
                    .inputs
                    .iter()
                    .map(|o| self.fact(*o))
                    .collect::<TfdResult<_>>()?;
                let output_facts: TVec<&TensorFact> =
                    node.outputs.iter().map(|o| &o.fact).collect();
                node.op.reduce(input_facts, output_facts)?
            };
            if let Some(red) = reduced {
                let mut node = &mut self.nodes[id];
                let ::ops::ReducedOpRewire { new_op, rewired } = red;
                node.op = new_op;
                let new_inputs = rewired.into_iter().map(|ix| node.inputs[ix]).collect();
                node.inputs = new_inputs;
            }
        }
        Ok(())
    }

    pub fn prop_constants(&mut self) -> TfdResult<()> {
        let mut done = bit_set::BitSet::with_capacity(self.nodes.len());
        let mut needed: Vec<usize> = vec![];
        for t in self.outputs()?.iter().map(|n| n.node) {
            needed.push(t);
        }
        while let Some(&node) = needed.last() {
            if done.contains(node) {
                needed.pop();
                continue;
            }
            if self.nodes[node]
                .inputs
                .iter()
                .all(|i| done.contains(i.node))
            {
                needed.pop();
                done.insert(node);
            } else {
                for ix in 0..self.nodes[node].inputs.len() {
                    use analyser::types::Fact;
                    let source = self.nodes[node].inputs[ix];
                    if self.nodes[source.node].op().name() != "Const"
                        && self.fact(source)?.is_concrete()
                    {
                        use self::dsl::ModelDsl;
                        let konst = self.fact(source)?.concretize().unwrap();
                        let id = self.nodes.len();
                        let id = self.add_const(format!("Const-{}", id), konst.clone())?;
                        self.add_edge(OutletId::new(id, 0), InletId::new(node, ix))?;
                        self.set_fact(OutletId::new(id, 0), konst.into())?;
                    } else {
                        needed.push(source.node);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn compact(&self) -> TfdResult<Model> {
        let mut model = Model::default();
        let mut map = HashMap::new();
        for old_id in self.eval_order()? {
            let old_node = &self.nodes[old_id];
            let new_id = model.add_node(old_node.name.clone(), old_node.op.clone())?;
            map.insert(old_id, new_id);
            for (ix, output) in old_node.outputs.iter().enumerate() {
                model.set_fact(OutletId::new(new_id, ix), output.fact.clone())?;
            }
            for (ix, input) in old_node.inputs.iter().enumerate() {
                model.add_edge(
                    OutletId::new(map[&input.node], input.slot),
                    InletId::new(new_id, ix),
                )?;
            }
        }
        // maintaining order of i/o interface
        model.inputs = self.inputs()?.iter().map(|i| OutletId::new(map[&i.node], i.slot)).collect();
        model.outputs = self.outputs()?.iter().map(|o| OutletId::new(map[&o.node], o.slot)).collect();
        Ok(model)
    }

    pub fn into_optimized(mut self) -> TfdResult<Model> {
        self.reduce()?;
        self.prop_constants()?;
        self.compact()
    }

    pub fn eval_order(&self) -> TfdResult<Vec<usize>> {
        eval_order(&self)
    }

    pub fn node_by_name(&self, name: &str) -> TfdResult<&Node> {
        let id: &usize = self
            .nodes_by_name
            .get(name)
            .ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&self.nodes[*id])
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| &*s.name).collect()
    }

    pub fn nodes(&self) -> &[Node] {
        &*self.nodes
    }

    pub fn fact(&self, outlet: OutletId) -> TfdResult<&TensorFact> {
        let outlets = &self.nodes[outlet.node].outputs;
        Ok(&outlets[outlet.slot].fact)
    }

    pub fn inputs_fact(&self, ix: usize) -> TfdResult<&TensorFact> {
        let input = self.inputs()?[ix];
        self.fact(input)
    }

    pub fn input_fact(&self) -> TfdResult<&TensorFact> {
        self.inputs_fact(0)
    }

    pub fn inputs(&self) -> TfdResult<&[OutletId]> {
        Ok(&self.inputs)
    }

    pub fn outputs_fact(&self, ix: usize) -> TfdResult<&TensorFact> {
        let output = self.outputs()?[ix];
        self.fact(output)
    }

    pub fn output_fact(&self) -> TfdResult<&TensorFact> {
        self.outputs_fact(0)
    }

    pub fn outputs(&self) -> TfdResult<&[OutletId]> {
        Ok(&self.outputs)
    }

    pub fn into_arc(self) -> Arc<Model> {
        Arc::new(self)
    }
}
