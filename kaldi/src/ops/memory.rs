use std::collections::BTreeMap;
use bit_set::BitSet;
use itertools::Itertools;

use tract_core::internal::*;

#[derive(Clone, Debug, new)]
pub struct Memory {
    pub name: String,
    pub offset: isize,
}

impl Op for Memory {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Memory".into()
    }

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        Ok(Some(incorporate_memory_ops_as_scans(model, node)?))
    }
}

impl StatefullOp for Memory {
    fn state(&self, _session: &mut SessionState, _id: usize) -> TractResult<Option<Box<OpState>>> {
        unimplemented!()
    }
}

impl InferenceOp for Memory {
    fn infer_facts(
        &mut self,
        _inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
        observed: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>, TVec<TensorFact>)> {
        let unified = outputs[0].unify(observed[0])?;
        Ok((tvec!(), tvec!(unified.clone()), tvec!(unified.clone())))
    }

    fn observe_outlets(
        &self,
        model: &InferenceModel,
        _node: &InferenceNode,
    ) -> TractResult<Vec<OutletId>> {
        Ok(vec![OutletId::new(model.node_by_name(&self.name)?.id, 0)])
    }

    inference_op_as_op!();
}

fn incorporate_memory_ops_as_scans(
    model: &InferenceModel,
    _: &InferenceNode,
) -> TractResult<InferenceModelPatch> {
    let memory_node_ids: Vec<usize> =
        model.nodes().iter().filter(|n| n.op_is::<Memory>()).map(|n| n.id).collect();
    let mut loops: BTreeMap<usize, BitSet> = memory_node_ids
        .iter()
        .map(|id| Ok((*id, time_loop_nodes_for_memory(model, *id)?)))
        .collect::<TractResult<_>>()?;

    let mut patch = InferenceModelPatch::default();
    while loops.len() > 0 {
        let (_, time_loop) = loops.iter().next().unwrap();
        let coupled_mem_ops: Vec<usize> = loops
            .iter()
            .filter_map(|other| if !other.1.is_disjoint(time_loop) { Some(*other.0) } else { None })
            .collect();
        let mut time_loop = BitSet::new();
        coupled_mem_ops.iter().for_each(|i| time_loop.union_with(&loops[i]));
        coupled_mem_ops.iter().for_each(|i| {
            loops.remove(i);
        });
        let scan_inputs: Vec<OutletId> = time_loop
            .iter()
            .flat_map(|node_id| model.node(node_id).inputs.iter())
            .filter(|outlet| !time_loop.contains(outlet.node))
            .cloned()
            .collect();
        let scan_outputs: Vec<InletId> = time_loop
            .iter()
            .flat_map(|node_id| model.node(node_id).outputs.iter())
            .flat_map(|outputs| outputs.successors.iter())
            .filter(|outlet| !time_loop.contains(outlet.node))
            .cloned()
            .collect();
        let mut inner_model = InferenceModel::default();
        let mut node_id_old_to_new: HashMap<usize, usize> = HashMap::new();
        for &mem in &coupled_mem_ops {
            let mem_node = model.node(mem);
            let channel = mem_node.outputs[0].fact.shape.dim(1).unwrap().concretize().unwrap().to_integer()? as usize;
            let id = inner_model.add_source(&*mem_node.name,  TensorFact::dt_shape(f32::datum_type(), ShapeFact::from(&[1, channel])))?;
            node_id_old_to_new.insert(mem, id);
        }
        for scan_input in scan_inputs.iter() {
            let old_node = model.node(scan_input.node);
            let channel = old_node.outputs[0].fact.shape.dim(1).unwrap().concretize().unwrap().to_integer()? as usize;
            let new_id = inner_model
                .add_source(format!("{}-scan", old_node.name), TensorFact::dt_shape(f32::datum_type(), ShapeFact::from(&[1, channel])))?;
            node_id_old_to_new.insert(scan_input.node, new_id);
        }
        for old_node_id in time_loop.iter() {
            if coupled_mem_ops.contains(&old_node_id) {
                continue;
            }
            let node = model.node(old_node_id);
            let new_id = inner_model.add_node(
                &*node.name,
                node.op.clone(),
                (0..node.outputs.len()).map(|_| TensorFact::default()).collect()
//                model.node_output_facts(old_node_id)?.into_iter().map(|ti| ti.to_tensor_fact()).collect(),
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
        let mut inner_outputs: Vec<OutletId> = coupled_mem_ops
            .iter()
            .map(|node| {
                let op = model.node(*node).op_as::<Memory>().unwrap();
                let observed_id = model.node_by_name(&op.name)?.id;
                Ok(OutletId::new(node_id_old_to_new[&observed_id], 0))
            })
            .collect::<TractResult<_>>()?;
        for output in &scan_outputs {
            let old_outlet = model.node(output.node).inputs[output.slot];
            inner_outputs
                .push(OutletId::new(node_id_old_to_new[&old_outlet.node], old_outlet.slot));
        }
        inner_model.set_output_outlets(&inner_outputs)?;

        // prepare patch
        let scan = tract_core::ops::rec::scan::Scan::new(
            inner_model,
            scan_inputs.len(),
            0,
            vec![0; scan_inputs.len()],
            vec![0; scan_outputs.len()],
            false,
        );

        let mut output_facts = tvec!();
        for memory in coupled_mem_ops.iter() {
            let channels = model.node(*memory).outputs[0].fact.shape.dim(1).unwrap().concretize().unwrap().to_integer()? as usize;
            let op = model.node(*memory).op_as::<Memory>().unwrap();
            let delay = (-op.offset) as usize;
            output_facts.push(TensorFact::dt_shape(f32::datum_type(), tvec![delay, channels]));
        }

        for output in &scan_outputs {
            let old_outlet = model.node(output.node).inputs[output.slot];
            output_facts.push(model.outlet_fact(old_outlet)?.clone());
        }

        let name = format!("scan-{}", scan_inputs.iter().map(|li| &model.node(li.node).name).join("-"));
        let scan_id = patch.add_node(name, scan, output_facts.iter().map(|ti| ti.to_tensor_fact()).collect())?;

        for (ix, memory) in coupled_mem_ops.iter().enumerate() {
            let op = model.node(*memory).op_as::<Memory>().unwrap();
            let zeroes = Tensor::from(tract_core::ndarray::ArrayD::<f32>::zeros(&*output_facts[ix].shape.as_concrete_finite().unwrap().unwrap()));
            patch.plug_const(InletId::new(scan_id, ix), format!("initial-zeros-for-{}", op.name), zeroes)?;
        }

        for (ix, input) in scan_inputs.iter().enumerate() {
            let tapped = patch.tap_model(model, *input)?;
            patch.add_edge(tapped, InletId::new(scan_id, ix + coupled_mem_ops.len()))?;
        }

        for (ix, output) in scan_outputs.iter().enumerate() {
            let old_outlet = model.node(output.node).inputs[output.slot];
            patch.shunt_outside(old_outlet, OutletId::new(scan_id, ix + coupled_mem_ops.len()))?;
        }
    }
    Ok(patch)
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
