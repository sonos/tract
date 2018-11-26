use std::collections::HashMap;
use std::str;
use std::sync::Arc;

pub mod dsl;
mod order;
pub use self::order::eval_order;
pub use analyser::types::TensorFact;
use context::Context;

pub use self::dsl::ModelDsl;
use {ops, TractResult};

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

    pub fn op_as<O: ops::Op>(&self) -> Option<&O> {
        self.op().downcast_ref::<O>()
    }
    pub fn op_is<O: ops::Op>(&self) -> bool {
        self.op_as::<O>().is_some()
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

/// Model is Tract workhouse.
#[derive(Clone, Debug)]
pub struct Model {
    ctx: Arc<::context::Context>,
    nodes: Vec<Node>,
    nodes_by_name: HashMap<String, usize>,
    pub(crate) inputs: Vec<OutletId>,
    pub(crate) outputs: Vec<OutletId>,
}

impl Default for Model {
    fn default() -> Model {
        Model {
            ctx: Arc::new(::context::DefaultContext),
            nodes: vec!(),
            nodes_by_name: HashMap::new(),
            inputs: vec!(),
            outputs: vec!(),
        }
    }
}

impl Model {
    pub fn with_context(self, ctx: Arc<Context>) -> Model {
        Model {
            ctx, ..self
        }
    }

    pub fn add_node(&mut self, name: String, op: Box<ops::Op>) -> TractResult<usize> {
        let id = self.nodes.len();
        self.nodes_by_name.insert(name.clone(), id);
        let is_input = op.name() == "Source";
        let noutputs = op.noutputs();
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
        for o in 0..noutputs {
            self.outputs.push(OutletId::new(id, o));
        }
        self.nodes.push(node);
        Ok(id)
    }

    pub fn add_edge(&mut self, outlet: OutletId, inlet: InletId) -> TractResult<()> {
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
    ) -> TractResult<()> {
        let ids: Vec<OutletId> = inputs
            .into_iter()
            .map(|s| {
                self.node_by_name(s.as_ref())
                    .map(|n| OutletId::new(n.id, 0))
            }).collect::<TractResult<_>>()?;
        self.inputs = ids;
        Ok(())
    }

    pub fn set_outputs(
        &mut self,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> TractResult<()> {
        let ids: Vec<OutletId> = outputs
            .into_iter()
            .map(|s| {
                self.node_by_name(s.as_ref())
                    .map(|n| OutletId::new(n.id, 0))
            }).collect::<TractResult<_>>()?;
        self.outputs = ids;
        Ok(())
    }

    pub fn set_outputs_outlets(&mut self, outputs: &[OutletId]) -> TractResult<()> {
        self.outputs = outputs.to_vec();
        Ok(())
    }

    pub fn set_fact(&mut self, outlet: OutletId, fact: TensorFact) -> TractResult<()> {
        let outlets = &mut self.nodes[outlet.node].outputs;
        if outlets.len() <= outlet.slot {
            outlets.push(OutletFact::default());
        }
        outlets[outlet.slot].fact = fact;
        Ok(())
    }

    pub fn set_input_fact(&mut self, input: usize, fact: TensorFact) -> TractResult<()> {
        let outlet = self.inputs()?[input];
        self.set_fact(outlet, fact)
    }

    pub fn facts(&self, id: usize) -> TractResult<(TVec<&TensorFact>, TVec<&TensorFact>)> {
        let node = &self.nodes[id];

        let inputs: TVec<&TensorFact> = node
            .inputs
            .iter()
            .enumerate()
            .map(|(ix, outlet)| (ix, outlet, self.fact(*outlet).unwrap()))
            .inspect(|(ix, outlet, fact)| {
                trace!("Input {} from {:?}: {:?}", ix, outlet, fact);
            }).map(|(_, _, fact)| fact)
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

    pub fn analyse_one(&mut self, id: usize) -> TractResult<()> {
        let _ = ::analyser::Analyser::new(self)?.analyse_one(id)?;
        Ok(())
    }

    pub fn analyse(&mut self) -> TractResult<()> {
        ::analyser::Analyser::new(self)?.analyse()
    }

    pub fn missing_type_shape(&self) -> TractResult<Vec<OutletId>> {
        use analyser::types::Fact;
        Ok(self
            .eval_order()?
            .iter()
            .flat_map(|&node| {
                self.nodes[node]
                    .outputs
                    .iter()
                    .enumerate()
                    .map(move |(ix, outlet)| (OutletId::new(node, ix), outlet))
            }).filter(|(_, o)| !o.fact.datum_type.is_concrete() || !o.fact.shape.is_concrete())
            .map(|(id, _)| id)
            .collect())
    }

    pub fn into_optimized(mut self) -> TractResult<Model> {
        self.analyse()?;
        for pass in self.ctx.optimizer_passes() {
            info!("Optization pass: {:?}", pass);
            pass.pass(&mut self)?;
        }
        let mut model = ::optim::compact(&self)?;
        model.analyse()?;
        Ok(model)
    }

    pub fn eval_order(&self) -> TractResult<Vec<usize>> {
        eval_order(&self)
    }

    pub fn node_by_name(&self, name: &str) -> TractResult<&Node> {
        let id: &usize = self
            .nodes_by_name
            .get(name)
            .ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&self.nodes[*id])
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| &*s.name).collect()
    }

    pub fn node(&self, id: usize) -> &Node {
        &self.nodes[id]
    }

    pub fn nodes(&self) -> &[Node] {
        &*self.nodes
    }

    pub fn mut_nodes(&mut self) -> &mut [Node] {
        &mut *self.nodes
    }

    pub fn fact(&self, outlet: OutletId) -> TractResult<&TensorFact> {
        let outlets = &self.nodes[outlet.node].outputs;
        Ok(&outlets[outlet.slot].fact)
    }

    pub fn inputs_fact(&self, ix: usize) -> TractResult<&TensorFact> {
        let input = self.inputs()?[ix];
        self.fact(input)
    }

    pub fn input_fact(&self) -> TractResult<&TensorFact> {
        self.inputs_fact(0)
    }

    pub fn inputs(&self) -> TractResult<&[OutletId]> {
        Ok(&self.inputs)
    }

    pub fn outputs_fact(&self, ix: usize) -> TractResult<&TensorFact> {
        let output = self.outputs()?[ix];
        self.fact(output)
    }

    pub fn output_fact(&self) -> TractResult<&TensorFact> {
        self.outputs_fact(0)
    }

    pub fn outputs(&self) -> TractResult<&[OutletId]> {
        Ok(&self.outputs)
    }

    pub fn into_arc(self) -> Arc<Model> {
        Arc::new(self)
    }
}
