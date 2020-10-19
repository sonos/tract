use bit_set::BitSet;
use std::collections::BTreeMap;
use tract_itertools::Itertools;

use tract_hir::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct Memory {
    pub name: String,
    pub offset: isize,
}

tract_data::impl_dyn_hash!(Memory);

impl Op for Memory {
    fn name(&self) -> Cow<str> {
        "Memory".into()
    }

    op_kaldi!();
    not_a_typed_op!();
}

impl EvalOp for Memory {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        unimplemented!()
    }
}

impl InferenceOp for Memory {
    fn infer_facts(
        &mut self,
        _inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        observed: TVec<&InferenceFact>,
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>, TVec<InferenceFact>)> {
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

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        Ok(Some(incorporate_memory_ops_as_scans(model, node)?))
    }

    as_op!();
}

fn incorporate_memory_ops_as_scans(
    model: &InferenceModel,
    _: &InferenceNode,
) -> TractResult<InferenceModelPatch> {
    let memory_node_ids: Vec<usize> =
        model.nodes().iter().filter(|n| n.op_is::<Memory>()).map(|n| n.id).collect();

    trace!("Identified memory nodes: {:?}", memory_node_ids);

    let mut loops: BTreeMap<usize, BitSet> = memory_node_ids
        .iter()
        .map(|id| Ok((*id, time_loop_nodes_for_memory(model, *id)?)))
        .collect::<TractResult<_>>()?;

    trace!("Loops: {:?}", loops);

    let mut patch = InferenceModelPatch::default();
    while loops.len() > 0 {
        let (mem, time_loop) = loops.iter().next().unwrap();

        trace!("Dealing with node {} / loop: {:?}", model.node(*mem), time_loop);

        let coupled_mem_ops: Vec<usize> = loops
            .iter()
            .filter_map(|other| if !other.1.is_disjoint(time_loop) { Some(*other.0) } else { None })
            .collect();
        let mut time_loop = BitSet::new();
        coupled_mem_ops.iter().for_each(|i| time_loop.union_with(&loops[i]));
        coupled_mem_ops.iter().for_each(|i| {
            loops.remove(i);
        });
        trace!("Loops still in queue: {:?}. Processing: {:?}", loops, coupled_mem_ops);

        let scan_inputs: Vec<OutletId> = time_loop
            .iter()
            .flat_map(|node_id| model.node(node_id).inputs.iter())
            .filter(|outlet| !time_loop.contains(outlet.node))
            .cloned()
            .collect();
        let scan_outputs: Vec<OutletId> = time_loop
            .iter()
            .flat_map(|node_id| {
                model
                    .node(node_id)
                    .outputs
                    .iter()
                    .enumerate()
                    .map(move |(ix, outlet_fact)| (OutletId::new(node_id, ix), outlet_fact))
            })
            .filter(|(_, outlet_fact)| {
                outlet_fact.successors.iter().any(|inlet| !time_loop.contains(inlet.node))
            })
            .map(|(id, _fact)| id)
            .collect();
        let mut inner_model = InferenceModel::default();
        let mut mapped_inputs = vec![];
        let mut mapped_outputs = vec![];
        let mut node_id_old_to_new: HashMap<usize, usize> = HashMap::new();
        for &mem in &coupled_mem_ops {
            let mem_node = model.node(mem);
            let op = mem_node.op_as::<Memory>().unwrap();
            let channel =
                mem_node.outputs[0].fact.shape.dim(1).unwrap().concretize().unwrap().to_usize()?;
            let chunk = op.offset.abs();
            let id = inner_model.add_source(
                &*mem_node.name,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    ShapeFactoid::from(&[(-op.offset) as usize, channel]),
                ),
            )?;
            node_id_old_to_new.insert(mem, id.node);

            let zeroes =
                Tensor::from(tract_ndarray::Array2::<f32>::zeros(((-op.offset) as usize, channel)));
            mapped_inputs.push(tract_hir::ops::scan::InputMapping::State {
                initializer: tract_hir::ops::scan::StateInitializer::Value(zeroes.into()),
            });
            mapped_outputs.push(tract_hir::ops::scan::OutputMapping {
                state: true,
                axis: 0,
                chunk,
                full_dim_hint: None,
                full_slot: None,
                last_value_slot: None,
            });
        }
        for scan_input in &scan_inputs {
            let old_node = model.node(scan_input.node);
            let channel =
                old_node.outputs[0].fact.shape.dim(1).unwrap().concretize().unwrap().to_usize()?;
            let new_id = inner_model.add_source(
                format!("{}-scan", old_node.name),
                InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(_, channel)),
            )?;
            node_id_old_to_new.insert(scan_input.node, new_id.node);
        }

        for old_node_id in time_loop.iter() {
            if coupled_mem_ops.contains(&old_node_id) {
                continue;
            }
            let node = model.node(old_node_id);
            let new_id = inner_model.add_node(
                &*node.name,
                node.op.clone(),
                (0..node.outputs.len()).map(|_| InferenceFact::default()).collect(),
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
            inner_outputs.push(OutletId::new(node_id_old_to_new[&output.node], output.slot));
        }

        inner_model.set_output_outlets(&inner_outputs)?;

        inner_model.analyse(false)?;

        for (ix, scan_input) in scan_inputs.iter().enumerate() {
            let old_node = model.node(scan_input.node);
            let fact = inner_model.input_fact(coupled_mem_ops.len() + ix)?;
            let chunk = fact.shape.dim(0).unwrap().concretize().unwrap().to_isize()?;
            mapped_inputs.push(tract_hir::ops::scan::InputMapping::Scan {
                axis: 0,
                chunk,
                slot: ix,
            });
            mapped_outputs.push(tract_hir::ops::scan::OutputMapping {
                state: false,
                axis: 0,
                chunk,
                full_slot: Some(ix),
                last_value_slot: None,
                full_dim_hint: old_node.outputs[0].fact.shape.dim(0).unwrap().concretize(),
            });
        }

        // prepare patch
        let scan = tract_hir::ops::scan::InferenceScan::new(
            inner_model,
            mapped_inputs,
            mapped_outputs,
            None,
            false,
            GenericFactoid::default(),
        );

        let mut output_facts = tvec!();

        for output in &scan_outputs {
            output_facts.push(model.outlet_fact(*output)?.clone());
        }

        let name =
            format!("scan-{}", scan_inputs.iter().map(|li| &model.node(li.node).name).join("-"));
        let scan_id = patch.add_node(name, scan, output_facts)?;

        for (ix, input) in scan_inputs.iter().enumerate() {
            let tapped = patch.tap_model(model, *input)?;
            patch.add_edge(tapped, InletId::new(scan_id, ix))?;
        }

        for (ix, output) in scan_outputs.iter().enumerate() {
            patch.shunt_outside(model, *output, OutletId::new(scan_id, ix))?;
        }

        for mem in coupled_mem_ops {
            patch.obliterate(mem)?
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
