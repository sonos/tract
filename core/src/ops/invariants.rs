use crate::internal::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Clone, Default)]
pub struct Invariants {
    element_wise: bool,
    pub axes: TVec<AxisInfo>,
}

impl Invariants {
    pub fn none() -> Invariants {
        Invariants { element_wise: false, axes: tvec!() }
    }

    pub fn new_element_wise(model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let (input_facts, output_facts) = model.node_facts(node.id)?;
        let all_facts = input_facts.iter().chain(output_facts.iter()).collect::<Vec<_>>();
        let shape = &all_facts[0].shape;
        if all_facts.iter().any(|s| shape != &s.shape) {
            bail!("Inconsistent element wise operation: {:?} {:?}", input_facts, output_facts);
        }
        let axes = (0..shape.rank())
            .map(|axis| {
                Ok(AxisInfo::for_node(model, node, axis)?.disposable(shape.dim(axis) == 1.into()))
            })
            .collect::<TractResult<_>>()?;
        Ok(Invariants { element_wise: true, axes })
    }

    pub fn element_wise(&self) -> bool {
        self.element_wise
    }
}

impl fmt::Debug for Invariants {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.axes.len() > 0 {
            if self.element_wise {
                write!(fmt, "Element wise. ")?;
            }
            write!(fmt, "Axes: {}", self.axes.iter().map(|axis| format!("{:?}", axis)).join(", "))?;
        } else {
            write!(fmt, "No invariants")?;
        }
        Ok(())
    }
}

impl From<TVec<AxisInfo>> for Invariants {
    fn from(axes: TVec<AxisInfo>) -> Invariants {
        Invariants { element_wise: false, axes }
    }
}

impl std::iter::FromIterator<AxisInfo> for Invariants {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = AxisInfo>,
    {
        Invariants { element_wise: false, axes: iter.into_iter().collect() }
    }
}

/// Translation invariance property.
#[derive(Clone, Default, Eq, Hash, PartialEq)]
pub struct AxisInfo {
    pub inputs: TVec<Option<usize>>,
    pub outputs: TVec<Option<usize>>,
    pub period: usize,
    pub disposable: bool,
}

impl fmt::Debug for AxisInfo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "{}->{}",
            self.inputs
                .iter()
                .map(|i| i.map(|a| a.to_string()).unwrap_or("_".to_string()))
                .join(","),
            self.outputs
                .iter()
                .map(|i| i.map(|a| a.to_string()).unwrap_or("_".to_string()))
                .join(",")
        )?;
        if !self.disposable {
            write!(fmt, " not disposable")?;
        }
        if self.period != 1 {
            write!(fmt, " period: {}", self.period)?;
        }
        Ok(())
    }
}

impl AxisInfo {
    pub fn simple(axis: usize) -> AxisInfo {
        AxisInfo {
            inputs: tvec!(Some(axis)),
            outputs: tvec!(Some(axis)),
            period: 1,
            disposable: true,
        }
    }

    pub fn with_period(self, period: usize) -> AxisInfo {
        AxisInfo { period, ..self }
    }

    pub fn disposable(self, disposable: bool) -> AxisInfo {
        AxisInfo { disposable, ..self }
    }

    pub fn for_node(_model: &TypedModel, node: &TypedNode, axis: usize) -> TractResult<AxisInfo> {
        Ok(AxisInfo {
            inputs: node.inputs.iter().map(|_| Some(axis)).collect(),
            outputs: node.outputs.iter().map(|_| Some(axis)).collect(),
            disposable: true,
            period: 1,
        })
    }
}

impl Invariants {
    pub fn track_input_axis(&self, input: usize, axis: usize) -> Option<&AxisInfo> {
        self.axes.iter().find(|conn| conn.inputs.get(input) == Some(&Some(axis)))
    }

    pub fn track_output_axis(&self, output: usize, axis: usize) -> Option<&AxisInfo> {
        self.axes.iter().find(|conn| conn.outputs.get(output) == Some(&Some(axis)))
    }

    pub fn unary_track_axis_up(&self, axis: usize, only_disposable: bool) -> Option<usize> {
        // TODO use track_input_axis
        if self.element_wise {
            Some(axis)
        } else {
            self.axes
                .iter()
                .find(|connection| {
                    connection.outputs.get(0) == Some(&Some(axis)) && connection.period == 1
                })
                .filter(|conn| conn.disposable || !only_disposable)
                .and_then(|connection| connection.inputs.get(0))
                .and_then(|d| *d)
        }
    }

    pub fn unary_track_axis_down(&self, axis: usize, only_disposable: bool) -> Option<usize> {
        // TODO use track_input_axis
        if self.element_wise {
            Some(axis)
        } else {
            self.axes
                .iter()
                .find(|connection| {
                    connection.inputs.get(0) == Some(&Some(axis)) && connection.period == 1
                })
                .filter(|conn| conn.disposable || !only_disposable)
                .and_then(|connection| connection.outputs.get(0))
                .and_then(|d| *d)
        }
    }
}
#[derive(Debug, Clone)]
pub struct AxisTracking {
    pub creators: TVec<OutletId>,
    pub destructors: TVec<InletId>,
    pub outlets: HashMap<OutletId, usize>,
    pub disposable: bool,
}

impl AxisTracking {
    pub fn for_outlet_and_axis(
        model: &TypedModel,
        outlet: OutletId,
        axis: usize,
    ) -> TractResult<AxisTracking> {
        let mut mapped_outlets = HashMap::<OutletId, usize>::new();
        let mut todo = HashSet::<OutletId>::new();
        let mut disposable = true;
        let mut creators = tvec!();
        let mut destructors = tvec!();
        mapped_outlets.insert(outlet, axis);
        todo.insert(outlet);
        while let Some(wire) = todo.iter().cloned().next() {
            todo.remove(&wire);
            let axis = mapped_outlets[&wire];
            let emiter_node = model.node(wire.node);
            let mut nodes = vec![];
            let invs = emiter_node
                .op
                .invariants(&model, emiter_node)
                .chain_err(|| format!("Computing invariants for {}", emiter_node))?;
            assert!(invs.axes.iter().all(|axis| axis.inputs.len() == emiter_node.inputs.len()));
            assert!(invs.axes.iter().all(|axis| axis.outputs.len() == emiter_node.outputs.len()));
            if let Some(info) = invs.track_output_axis(wire.slot, axis) {
                nodes.push((wire.node, info.clone()));
            } else {
                creators.push(wire);
            };
            for succ in &emiter_node.outputs[wire.slot].successors {
                let succ_node = model.node(succ.node);
                let invs = succ_node.op.invariants(&model, succ_node)?;
                assert!(invs.axes.iter().all(|axis| axis.inputs.len() == succ_node.inputs.len()));
                assert!(invs.axes.iter().all(|axis| axis.outputs.len() == succ_node.outputs.len()));
                if let Some(info) = invs.track_input_axis(succ.slot, axis) {
                    nodes.push((succ_node.id, info.clone()));
                } else {
                    destructors.push(*succ);
                };
            }
            let mut new_outlets = vec![];
            for (n, axes) in nodes {
                disposable = disposable && axes.disposable;
                let node = model.node(n);
                for slot in 0..node.outputs.len() {
                    if let Some(axis) = axes.outputs[slot] {
                        new_outlets.push((OutletId::new(n, slot), axis));
                    }
                }
                for slot in 0..node.inputs.len() {
                    if let Some(axis) = axes.inputs[slot] {
                        new_outlets.push((node.inputs[slot], axis));
                    }
                }
            }
            for (outlet, axis) in new_outlets {
                if let Some(prev) = mapped_outlets.get(&outlet) {
                    if *prev != axis {
                        bail!("Inconsistent network");
                    }
                } else {
                    mapped_outlets.insert(outlet, axis);
                    todo.insert(outlet);
                }
            }
        }
        Ok(AxisTracking { creators, destructors, outlets: mapped_outlets, disposable })
    }
}

pub fn full_axis_tracking(model: &TypedModel) -> TractResult<Vec<AxisTracking>> {
    let mut axes: Vec<AxisTracking> = vec![];
    for node in model.eval_order()? {
        for slot in 0..model.node(node).outputs.len() {
            let outlet = OutletId::new(node, slot);
            let input_fact = model.outlet_fact(outlet)?;
            'axis: for axis in 0..input_fact.rank() {
                if axes.iter().any(|tracking| tracking.outlets.get(&outlet) == Some(&axis)) {
                    continue 'axis;
                }
                axes.push(AxisTracking::for_outlet_and_axis(model, outlet, axis)?);
            }
        }
    }
    Ok(axes)
}

pub fn for_model(model: &TypedModel) -> TractResult<Invariants> {
    full_axis_tracking(model)?
        .into_iter()
        .map(|tracking| {
            let inputs =
                model.input_outlets()?.iter().map(|i| tracking.outlets.get(i).cloned()).collect();
            let outputs =
                model.input_outlets()?.iter().map(|i| tracking.outlets.get(i).cloned()).collect();
            Ok(AxisInfo { inputs, outputs, disposable: tracking.disposable, period: 1 })
        })
        .collect()
}
