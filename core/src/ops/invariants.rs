use crate::internal::*;
use itertools::Itertools;
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
                Ok(AxisInfo::for_node(model, node, axis)?.disposable(shape[axis] == 1.into()))
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
/*
#[derive(Debug, Clone, Default)]
pub struct OutletMap<T>(std::collections::BTreeMap<OutletId, T>);

impl<T> OutletMap<T> {
fn insert(&mut self, outlet: OutletId, t: T) {
self.0.insert(outlet, t);
}

fn remove(&mut self, outlet: &OutletId) -> Option<T> {
self.0.remove(outlet)
}

fn get(&self, outlet: &OutletId) -> Option<&T> {
self.0.get(outlet)
}
}

impl<'a, T> std::ops::Index<&'a OutletId> for OutletMap<T> {
type Output = T;
fn index(&self, index: &'a OutletId) -> &Self::Output {
&self.0[index]
}
}
*/

#[derive(Debug, Clone, Default)]
pub struct OutletMap<T>(Vec<TVec<Option<T>>>);

impl<T: Clone> OutletMap<T> {
    fn insert(&mut self, outlet: OutletId, t: T) {
        if outlet.node >= self.0.len() {
            self.0.resize_with(outlet.node + 1, || tvec!());
        }
        let node = &mut self.0[outlet.node];
        if outlet.slot >= node.len() {
            node.resize(outlet.slot + 1, None);
        }
        node[outlet.slot] = Some(t)
    }
}

impl<T> OutletMap<T> {
    fn remove(&mut self, outlet: &OutletId) -> Option<T> {
        if let Some(node) = self.0.get_mut(outlet.node) {
            if let Some(slot) = node.get_mut(outlet.slot) {
                return slot.take();
            }
        }
        None
    }

    fn get(&self, outlet: &OutletId) -> Option<&T> {
        if let Some(node) = self.0.get(outlet.node) {
            if let Some(slot) = node.get(outlet.slot) {
                return slot.as_ref();
            }
        }
        None
    }

    fn keys(&self) -> OutletMapKeysIter<T> {
        OutletMapKeysIter(self, (0, 0).into())
    }

    /*
    fn iter(&self) -> OutletMapIter<T> {
        OutletMapIter(self, 0, 0)
    }
    */
}

impl<'a, T: Clone> std::ops::Index<&'a OutletId> for OutletMap<T> {
    type Output = T;
    fn index(&self, index: &'a OutletId) -> &Self::Output {
        self.get(index).unwrap()
    }
}

struct OutletMapKeysIter<'a, T>(&'a OutletMap<T>, OutletId);

impl<'a, T> std::iter::Iterator for OutletMapKeysIter<'a, T> {
    type Item = OutletId;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.1.node >= (self.0).0.len() {
                return None;
            }
            if self.1.slot >= (self.0).0[self.1.node].len() {
                self.1.slot = 0;
                self.1.node += 1;
                continue;
            }
            let current = self.1.clone();
            self.1.slot += 1;
            if self.0.get(&current).is_some() {
                return Some(current);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AxisTracking {
    pub creators: TVec<OutletId>,
    pub destructors: TVec<InletId>,
    pub outlets: OutletMap<usize>,
    pub disposable: bool,
}

impl AxisTracking {
    pub fn for_outlet_and_axis(
        model: &TypedModel,
        outlet: OutletId,
        axis: usize,
    ) -> TractResult<AxisTracking> {
        let mut mapped_outlets = OutletMap::default();
        let mut todo = OutletMap::default();
        let mut disposable = true;
        let mut creators = tvec!();
        let mut destructors = tvec!();
        mapped_outlets.insert(outlet, axis);
        todo.insert(outlet, ());
        while let Some(wire) = todo.keys().next() {
            todo.remove(&wire);
            let axis = mapped_outlets[&wire];
            let emiter_node = model.node(wire.node);
            let mut nodes = vec![];
            let invs = emiter_node
                .op
                .invariants(&model, emiter_node)
                .with_context(|| format!("Computing invariants for {}", emiter_node))?;
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
                    todo.insert(outlet, ());
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
