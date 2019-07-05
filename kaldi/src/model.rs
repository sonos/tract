use tract_core::internal::*;

use crate::ops::memory::Memory;
use bit_set::BitSet;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct KaldiProtoModel {
    pub config_lines: ConfigLines,
    pub components: HashMap<String, Component>,
}

#[derive(Clone, Debug)]
pub struct ConfigLines {
    pub input_name: String,
    pub input_dim: usize,
    pub component_nodes: HashMap<String, ComponentNode>,
    pub dim_range_nodes: HashMap<String, DimRangeNode>,
    pub output_name: String,
    pub output_input: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GeneralDescriptor {
    Append(Vec<GeneralDescriptor>),
    IfDefined(Box<GeneralDescriptor>),
    Name(String),
    Offset(Box<GeneralDescriptor>, isize),
}

impl GeneralDescriptor {
    pub fn inputs(&self) -> TVec<&str> {
        match self {
            GeneralDescriptor::Append(ref gds) => gds.iter().fold(tvec!(), |mut acc, gd| {
                gd.inputs().iter().for_each(|i| {
                    if !acc.contains(i) {
                        acc.push(i)
                    }
                });
                acc
            }),
            GeneralDescriptor::IfDefined(ref gd) => gd.inputs(),
            GeneralDescriptor::Name(ref s) => tvec!(&**s),
            GeneralDescriptor::Offset(ref gd, _) => gd.inputs(),
        }
    }

    pub fn as_conv_shape_dilation(&self) -> Option<(usize, usize)> {
        if let GeneralDescriptor::Name(_) = self {
            return Some((1, 1));
        }
        if let GeneralDescriptor::Append(ref appendees) = self {
            let mut offsets = vec![];
            for app in appendees {
                match app {
                    GeneralDescriptor::Name(_) => offsets.push(0),
                    GeneralDescriptor::Offset(_, offset) => offsets.push(*offset),
                    _ => return None,
                }
            }
            let dilation = offsets[1] - offsets[0];
            if offsets.windows(2).all(|pair| pair[1] - pair[0] == dilation) {
                return Some((offsets.len(), dilation as usize));
            }
        }
        return None;
    }

    fn wire<'a>(
        &'a self,
        inlet: InletId,
        name: &str,
        model: &mut InferenceModel,
        deferred: &mut BTreeMap<InletId, String>,
    ) -> TractResult<()> {
        use GeneralDescriptor::*;
        match &self {
            &Name(n) => {
                deferred.insert(inlet, n.to_string());
                return Ok(());
            }
            &Append(appendees) => {
                let name = format!("{}-Append", name);
                let id = model.add_node_default(&*name, tract_core::ops::array::Concat::new(1))?;
                model.add_edge(OutletId::new(id, 0), inlet)?;
                for (ix, appendee) in appendees.iter().enumerate() {
                    let name = format!("{}-{}", name, ix);
                    appendee.wire(InletId::new(id, ix), &*name, model, deferred)?;
                }
                return Ok(());
            }
            &IfDefined(ref o) => {
                if let &Offset(ref n, ref o) = &**o {
                    if let Name(n) = &**n {
                        let name = format!("{}-Memory", name);
                        model.add_node_default(
                            &*name,
                            crate::ops::memory::Memory::new(n.to_string(), *o),
                        )?;
                        deferred.insert(inlet, name);
                        return Ok(());
                    }
                }
            }
            _ => (),
        }
        bail!("Unhandled input descriptor: {:?}", self)
    }
}

#[derive(Clone, Debug)]
pub struct DimRangeNode {
    pub input: GeneralDescriptor,
    pub offset: usize,
    pub dim: usize,
}

#[derive(Clone, Debug)]
pub struct ComponentNode {
    pub input: GeneralDescriptor,
    pub component: String,
}

#[derive(Clone, Debug, Default)]
pub struct Component {
    pub klass: String,
    pub attributes: HashMap<String, Arc<Tensor>>,
}

pub struct ParsingContext<'a> {
    pub proto_model: &'a KaldiProtoModel,
}

#[derive(Clone, Default)]
pub struct KaldiOpRegister(
    pub HashMap<String, fn(&ParsingContext, node: &str) -> TractResult<Box<InferenceOp>>>,
);

impl KaldiOpRegister {
    pub fn insert(
        &mut self,
        s: &'static str,
        builder: fn(&ParsingContext, node: &str) -> TractResult<Box<InferenceOp>>,
    ) {
        self.0.insert(s.into(), builder);
    }
}

#[derive(Clone, Default)]
pub struct Kaldi {
    pub op_register: KaldiOpRegister,
}

impl Framework<KaldiProtoModel> for Kaldi {
    fn proto_model_for_read(&self, r: &mut std::io::Read) -> TractResult<KaldiProtoModel> {
        use crate::parser;
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        parser::nnet3(&*v)
    }

    fn model_for_proto_model(&self, proto_model: &KaldiProtoModel) -> TractResult<InferenceModel> {
        let ctx = ParsingContext { proto_model };
        let mut model = InferenceModel::default();
        model.add_source(
            proto_model.config_lines.input_name.clone(),
            TensorFact::dt_shape(
                f32::datum_type(),
                shapefact!(S, (proto_model.config_lines.input_dim)),
            ),
        )?;
        let mut inputs_to_wire: BTreeMap<InletId, String> = Default::default();
        for (name, node) in &proto_model.config_lines.component_nodes {
            let component = &proto_model.components[&node.component];
            if crate::ops::AFFINE.contains(&&*component.klass)
                && node.input.as_conv_shape_dilation().is_some()
            {
                let op = crate::ops::affine::affine_component(&ctx, name)?;
                let id = model.add_node_default(name.to_string(), op)?;
                inputs_to_wire.insert(InletId::new(id, 0), node.input.inputs()[0].to_owned());
            } else {
                let op = match self.op_register.0.get(&*component.klass) {
                    Some(builder) => (builder)(&ctx, name)?,
                    None => {
                        (Box::new(tract_core::ops::unimpl::UnimplementedOp::new(
                            component.klass.to_string(),
                            format!("{:?}", proto_model.config_lines.component_nodes.get(name)),
                        )))
                    }
                };
                let id = model.add_node_default(name.to_string(), op)?;
                node.input.wire(InletId::new(id, 0), name, &mut model, &mut inputs_to_wire)?
            }
        }
        for (name, node) in &proto_model.config_lines.dim_range_nodes {
            let op = tract_core::ops::array::Slice::new(
                vec![1],
                vec![node.offset as usize],
                vec![(node.offset + node.dim) as usize],
            );
            let id = model.add_node_default(name.to_string(), op)?;
            node.input.wire(InletId::new(id, 0), name, &mut model, &mut inputs_to_wire)?
        }
        for (inlet, name) in inputs_to_wire {
            let src = OutletId::new(model.node_by_name(&*name)?.id, 0);
            model.add_edge(src, inlet)?;
        }
        let output = model.add_node_default(
            proto_model.config_lines.output_name.to_string(),
            tract_core::ops::identity::Identity::default(),
        )?;
        let src = OutletId::new(model.node_by_name(&*proto_model.config_lines.output_input)?.id, 0);
        let dst = InletId::new(output, 0);
        model.add_edge(src, dst)?;
        model.set_output_outlets(&[OutletId::new(output, 0)])?;
        /*
        reinterpret_memory_ops_as_scans(&mut model)?;
        */
        Ok(model)
    }
}

pub fn reinterpret_memory_ops_as_scans(model: &mut InferenceModel) -> TractResult<()> {
    // println!("{:#?}", model);
    for mem_node in model.nodes() {
        if mem_node.op_is::<Memory>() {
            let observed_node_id = model.node_by_name(&mem_node.name)?.id;
            let time_loop = time_loop_nodes_for_memory(model, mem_node.id)?;
            let loop_inputs: Vec<OutletId> = time_loop
                .iter()
                .flat_map(|node_id| model.node(node_id).inputs.iter())
                .filter(|outlet| !time_loop.contains(outlet.node))
                .cloned()
                .collect();
            let loop_outputs: Vec<InletId> = time_loop
                .iter()
                .flat_map(|node_id| model.node(node_id).outputs.iter())
                .flat_map(|outputs| outputs.successors.iter())
                .filter(|outlet| !time_loop.contains(outlet.node))
                .cloned()
                .collect();
            let mut inner_model = InferenceModel::default();
            let mut node_id_old_to_new: HashMap<usize, usize> = HashMap::new();
            let id = inner_model.add_source_default(&*mem_node.name)?;
            node_id_old_to_new.insert(mem_node.id, id);
            for loop_input in loop_inputs.iter() {
                let new_id = inner_model
                    .add_source_default(format!("{}-scan", model.node(loop_input.node).name))?;
                node_id_old_to_new.insert(loop_input.node, new_id);
            }
            for node in time_loop.iter() {
                if node == mem_node.id {
                    continue;
                }
                let node = model.node(node);
                let new_id = inner_model.add_node(
                    &*node.name,
                    node.op.clone(),
                    node.outputs.iter().map(|of| of.fact.clone()).collect(),
                )?;
                node_id_old_to_new.insert(node.id, new_id);
            }
            for node in time_loop.iter() {
                let node = model.node(node);
                for (ix, input) in node.inputs.iter().enumerate() {
                    inner_model.add_edge(
                        OutletId::new(node_id_old_to_new[&input.node], input.slot),
                        InletId::new(node_id_old_to_new[&node.id], ix),
                    )?;
                }
            }
            let mut inner_outputs = vec![OutletId::new(observed_node_id, 0)];
            for output in &loop_outputs {
                let old_outlet = model.node(output.node).inputs[output.slot];
                inner_outputs
                    .push(OutletId::new(node_id_old_to_new[&old_outlet.node], old_outlet.slot));
            }
            inner_model.set_output_outlets(&inner_outputs)?;

            let scan = tract_core::ops::rec::scan::Scan::new(
                inner_model,
                loop_inputs.len(),
                0,
                vec![1; loop_inputs.len()],
                vec![1; loop_outputs.len()],
            );

            let mut patch = InferenceModelPatch::default();
            patch.add_node_default("scan", scan)?;
            unimplemented!();
        }
    }
    Ok(())
}

pub fn time_loop_nodes_for_memory(
    model: &InferenceModel,
    memory_node_id: usize,
) -> TractResult<BitSet> {
    let memory_name = if let Some(mem) = &model.node(memory_node_id).op_as::<Memory>() {
        &*mem.name
    } else {
        bail!("Should only be called for a memory name")
    };
    let observed_node_id = model.node_by_name(&memory_name)?.id;
    let mut time_loop = all_successors(model, memory_node_id)?;
    let precursors = all_precursors(model, observed_node_id)?;
    time_loop.intersect_with(&precursors);
    Ok(time_loop)
}

pub fn all_successors(model: &InferenceModel, id: usize) -> TractResult<BitSet> {
    let mut queue = vec![id];
    let mut visited = BitSet::with_capacity(model.nodes().len());
    visited.insert(id);
    while let Some(next) = queue.pop() {
        let node = model.node(next);
        for out in &node.outputs {
            for suc in &out.successors {
                if !visited.contains(suc.node) {
                    queue.push(suc.node);
                    visited.insert(suc.node);
                }
            }
        }
    }
    Ok(visited)
}

pub fn all_precursors(model: &InferenceModel, id: usize) -> TractResult<BitSet> {
    let mut queue = vec![id];
    let mut visited = BitSet::with_capacity(model.nodes().len());
    visited.insert(id);
    while let Some(next) = queue.pop() {
        let node = model.node(next);
        for prec in &node.inputs {
            if !visited.contains(prec.node) {
                queue.push(prec.node);
                visited.insert(prec.node);
            }
        }
    }
    Ok(visited)
}
