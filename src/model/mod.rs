use std::collections::HashMap;
use std::str;
use std::sync::Arc;

mod dsl;
mod order;
pub use self::order::eval_order;
pub use analyser::types::TensorFact;

use {ops, TfdResult};
pub use self::dsl::ModelDsl;

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
    pub(crate) inputs: Vec<OutletId>,
    pub(crate) outputs: Vec<OutletId>,
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

    pub fn into_optimized(mut self) -> TfdResult<Model> {
        use optim::OptimizerPass;
        ::optim::Reduce::pass(&mut self)?;
        ::optim::prop_const(&mut self)?;
        ::optim::compact(&self)
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

    pub fn mut_nodes(&mut self) -> &mut [Node] {
        &mut *self.nodes
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
