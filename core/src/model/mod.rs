use std::collections::HashMap;
use std::str;
use std::sync::Arc;

pub mod dsl;
mod order;
pub mod patch;

pub use self::order::eval_order;
pub use crate::analyser::types::TensorFact;

pub use self::dsl::ModelDsl;
use crate::analyser::types::ShapeFact;
use crate::datum::TryInto;
use crate::dim::ToDim;
pub use crate::framework::*;
use crate::{ops, DatumType, SharedTensor, TDim, TractResult};
use patch::ModelPatch;

pub trait TensorInfo: Clone + std::fmt::Debug {
    fn to_tensor_fact(&self) -> TensorFact;
}

impl<TI: TensorInfo> TryInto<TI> for TI {
    fn try_into(&self) -> TractResult<TI> {
        Ok(self.clone())
    }
}

impl TensorInfo for TensorFact {
    fn to_tensor_fact(&self) -> TensorFact {
        self.clone()
    }
}

impl TryInto<TypedTensorInfo> for TensorFact {
    fn try_into(&self) -> TractResult<TypedTensorInfo> {
        use crate::analyser::types::Fact;
        if let (Some(datum_type), Some(shape)) =
            (self.datum_type.concretize(), self.shape.concretize())
        {
            let stream_info = shape
                .iter()
                .cloned()
                .enumerate()
                .find(|d| d.1.to_integer().is_err())
                .map(|(axis, len)| StreamInfo { axis, len });
            let shape = shape.iter().map(|d| d.to_integer().unwrap_or(0) as usize).collect();
            let shape = ShapeInfo { shape, stream_info };
            Ok(TypedTensorInfo { datum_type, shape, konst: self.value.concretize() })
        } else {
            bail!("Can not make a TypedTensorInfo out of {:?}", self)
        }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct StreamInfo {
    pub axis: usize,
    pub len: TDim,
}

#[derive(Debug, Clone)]
pub struct ShapeInfo {
    shape: TVec<usize>,
    pub stream_info: Option<StreamInfo>,
}

impl PartialEq for ShapeInfo {
    fn eq(&self, other: &ShapeInfo) -> bool {
        self.shape.len() == other.shape.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl ShapeInfo {
    pub fn as_finite(&self) -> Option<&[usize]> {
        match self.stream_info {
            None => Some(&*self.shape),
            _ => None,
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = TDim> + 'a {
        self.shape.clone().into_iter().enumerate().map(move |(ix, d)| {
            if let Some(info) = self.stream_info {
                if ix == info.axis {
                    return info.len;
                }
            }
            (d as i64).to_dim()
        })
    }

    pub fn to_shape_fact(&self) -> ShapeFact {
        ShapeFact::from(self.iter())
    }
}

impl<T: AsRef<[usize]>> From<T> for ShapeInfo {
    fn from(it: T) -> ShapeInfo {
        ShapeInfo { shape: it.as_ref().iter().cloned().collect(), stream_info: None }
    }
}

#[derive(Debug, Clone)]
pub struct TypedTensorInfo {
    pub datum_type: DatumType,
    pub shape: ShapeInfo,
    pub konst: Option<SharedTensor>,
}

impl TensorInfo for TypedTensorInfo {
    fn to_tensor_fact(&self) -> TensorFact {
        match self.konst.clone() {
            Some(k) => k.into(),
            None => TensorFact::dt_shape(self.datum_type, self.shape.to_shape_fact()),
        }
    }
}

impl From<SharedTensor> for TypedTensorInfo {
    fn from(t: SharedTensor) -> TypedTensorInfo {
        TypedTensorInfo {
            datum_type: t.datum_type(),
            shape: ShapeInfo { shape: t.shape().into(), stream_info: None },
            konst: Some(t),
        }
    }
}

impl TryInto<NormalizedTensorInfo> for TypedTensorInfo {
    fn try_into(&self) -> TractResult<NormalizedTensorInfo> {
        match self.konst {
            None => {
                Ok(NormalizedTensorInfo { shape: self.shape.clone(), datum_type: self.datum_type })
            }
            _ => bail!("Constant tensor are excluded from normalized stage: {:?}", self),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedTensorInfo {
    pub datum_type: DatumType,
    pub shape: ShapeInfo,
}

impl TensorInfo for NormalizedTensorInfo {
    fn to_tensor_fact(&self) -> TensorFact {
        TensorFact::dt_shape(self.datum_type, self.shape.to_shape_fact())
    }
}

pub type InferenceModel = Model<TensorFact>;
pub type InferenceNode = Node<TensorFact>;

pub type TypedModel = Model<TypedTensorInfo>;
pub type TypedNode = Node<TypedTensorInfo>;
pub type TypedModelPatch = ModelPatch<TypedTensorInfo>;

pub type NormalizedModel = Model<NormalizedTensorInfo>;
pub type NormalizedNode = Node<NormalizedTensorInfo>;
pub type NormalizedModelPatch = ModelPatch<NormalizedTensorInfo>;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct Node<TI: TensorInfo> {
    pub id: usize,
    pub name: String,
    pub inputs: Vec<OutletId>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: Box<ops::Op>,
    pub outputs: TVec<OutletFact<TI>>,
}

impl<TI: TensorInfo> Node<TI> {
    pub fn op(&self) -> &ops::Op {
        &*self.op
    }

    pub fn op_as<O: ops::Op>(&self) -> Option<&O> {
        self.op().downcast_ref::<O>()
    }

    pub fn op_is<O: ops::Op>(&self) -> bool {
        self.op_as::<O>().is_some()
    }

    pub fn same_as(&self, other: &Node<TI>) -> bool {
        self.inputs == other.inputs && self.op.same_as(other.op.as_ref())
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletFact<TI: TensorInfo> {
    pub fact: TI,
    pub successors: TVec<InletId>,
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
pub struct Model<TI: TensorInfo> {
    //    norm: Option<Arc<Vec<Box<OptimizerPass>>>>,
    nodes: Vec<Node<TI>>,
    nodes_by_name: HashMap<String, usize>,
    pub(crate) inputs: Vec<OutletId>,
    pub(crate) outputs: Vec<OutletId>,
}

impl<TI: TensorInfo> Default for Model<TI> {
    fn default() -> Model<TI> {
        Model {
            /*
            norm: None,
            */
            nodes: vec![],
            nodes_by_name: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
        }
    }
}

impl<TI: TensorInfo> Model<TI> {
    /*
    pub fn with_norm_optims(self, norm: Option<Arc<Vec<Box<OptimizerPass>>>>) -> Model<TI> {
        Model { norm, ..self }
    }

    pub fn norm_optims(&self) -> Option<&Arc<Vec<Box<OptimizerPass>>>> {
        self.norm.as_ref()
    }
    */

    pub fn add_node(
        &mut self,
        name: String,
        op: Box<ops::Op>,
        outputs_fact: TVec<TI>,
    ) -> TractResult<usize> {
        self.add_node_disable_output_guess(name, op, outputs_fact, false)
    }

    pub(crate) fn add_node_disable_output_guess(
        &mut self,
        name: String,
        op: Box<ops::Op>,
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

    /*
    pub fn into_optimized(self) -> TractResult<Model<TI>> {
        self.into_normalized()?.into_codegen()
    }
    */

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

impl InferenceModel {
    pub fn analyse_one(&mut self, id: usize) -> TractResult<()> {
        let _ = crate::analyser::Analyser::new(self)?.analyse_one(id)?;
        Ok(())
    }

    pub fn analyse(&mut self) -> TractResult<()> {
        crate::analyser::Analyser::new(self)?.analyse()
    }

    pub fn missing_type_shape(&self) -> TractResult<Vec<OutletId>> {
        use crate::analyser::types::Fact;
        Ok(self
            .eval_order()?
            .iter()
            .flat_map(|&node| {
                self.nodes[node]
                    .outputs
                    .iter()
                    .enumerate()
                    .map(move |(ix, outlet)| (OutletId::new(node, ix), outlet))
            })
            .filter(|(_, o)| !o.fact.datum_type.is_concrete() || !o.fact.shape.is_concrete())
            .map(|(id, _)| id)
            .collect())
    }

    pub fn into_typed(mut self) -> TractResult<TypedModel> {
        self.analyse()?;
        crate::optim::compact(&mut self)
    }

    pub fn into_normalized(self) -> TractResult<NormalizedModel> {
        self.into_typed()?.into_normalized()
    }

    pub fn into_optimized(self) -> TractResult<NormalizedModel> {
        self.into_normalized()?.into_codegen()
    }
}

impl TypedModel {
    pub fn into_normalized(self) -> TractResult<NormalizedModel> {
        let mut model = self;
        loop {
            let mut done_something = false;
            /*
            if let Some(passes) = model.norm.clone() {
                for p in passes.iter() {
                    done_something = done_something || p.pass(&mut model)?;
                    if cfg!(debug_assertions) {
                        model.check_edges()?;
                    }
                }
            }
            */
            for p in crate::optim::normalization() {
                done_something = done_something || p.pass(&mut model)?;
                if cfg!(debug_assertions) {
                    model.check_edges()?;
                }
            }
            if !done_something {
                break;
            }
            model = crate::optim::compact(&model)?;
        }
        crate::optim::compact(&model)
    }
}

impl NormalizedModel {
    pub fn into_codegen(mut self) -> TractResult<NormalizedModel> {
        loop {
            let mut done_something = false;
            for p in crate::optim::codegen() {
                done_something = done_something || p.pass(&mut self)?;
                if cfg!(debug_assertions) {
                    self.check_edges()?;
                }
            }
            if !done_something {
                break;
            }
        }
        crate::optim::compact(&self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        fn is_sync<T: Sync>() {}
        is_sync::<InferenceModel>();
        is_sync::<TypedModel>();
        is_sync::<NormalizedModel>();
    }
}
