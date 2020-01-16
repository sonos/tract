use crate::model::{TypedModel, TypedNode};
use crate::prelude::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ops::konst::Const;
use crate::ops::source::TypedSource;

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
            .map(|axis| Ok(AxisInfo::for_node(model, node, axis)?.disposable(shape.dim(axis) == 1.into())))
            .collect::<TractResult::<_>>()?;
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

    pub fn for_node(
        _model: &TypedModel,
        node: &TypedNode,
        axis: usize,
    ) -> TractResult<AxisInfo> {
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

pub fn full_axis_tracking(
    model: &TypedModel,
) -> TractResult<Vec<(HashMap<OutletId, usize>, bool)>> {
    let mut axes: Vec<(HashMap<OutletId, usize>, bool)> = vec![];
    for &input in model.input_outlets()? {
        let input_fact = model.outlet_fact(input)?;
        'axis: for axis in 0..input_fact.rank() {
            if axes.iter().any(|(group, _)| group.get(&input) == Some(&axis)) {
                continue 'axis;
            }
            let mut mapped = HashMap::<OutletId, usize>::new();
            let mut todo = HashSet::<OutletId>::new();
            let mut disposable = true;
            mapped.insert(input, axis);
            todo.insert(input);
            while let Some(wire) = todo.iter().cloned().next() {
                todo.remove(&wire);
                let axis = mapped[&wire];
                let emiter_node = model.node(wire.node);
                let mut nodes = vec![];
                if !emiter_node.op().is::<TypedSource>() && !emiter_node.op().is::<Const>() {
                    let invs = emiter_node.op.invariants(&model, emiter_node)?;
                    let axis_info = if let Some(info) = invs.track_output_axis(wire.slot, axis) {
                        info
                    } else {
                        continue 'axis;
                    };
                    nodes.push((wire.node, axis_info.clone()));
                }
                for succ in &emiter_node.outputs[wire.slot].successors {
                    let succ_node = model.node(succ.node);
                    let invs = succ_node.op.invariants(&model, succ_node)?;
                    let axis_info = if let Some(info) = invs.track_input_axis(succ.slot, axis) {
                        info
                    } else {
                        continue 'axis;
                    };
                    nodes.push((succ_node.id, axis_info.clone()));
                }
                let mut outlets = vec![];
                for (n, axes) in nodes {
                    disposable = disposable && axes.disposable;
                    let node = model.node(n);
                    for slot in 0..node.outputs.len() {
                        if let Some(axis) = axes.outputs[slot] {
                            outlets.push((OutletId::new(n, slot), axis));
                        }
                    }
                    for slot in 0..node.inputs.len() {
                        if let Some(axis) = axes.inputs[slot] {
                            outlets.push((node.inputs[slot], axis));
                        }
                    }
                }
                for (outlet, axis) in outlets {
                    if let Some(prev) = mapped.get(&outlet) {
                        if *prev != axis {
                            bail!("Inconsistent network");
                        }
                    } else {
                        mapped.insert(outlet, axis);
                        todo.insert(outlet);
                    }
                }
            }
            axes.push((mapped, disposable));
        }
    }
    Ok(axes)
}

pub fn for_model(model: &TypedModel) -> TractResult<Invariants> {
    full_axis_tracking(model)?
        .into_iter()
        .map(|(axes, disposable)| {
            let inputs = model.input_outlets()?.iter().map(|i| axes.get(i).cloned()).collect();
            let outputs = model.input_outlets()?.iter().map(|i| axes.get(i).cloned()).collect();
            Ok(AxisInfo { inputs, outputs, disposable, period: 1 })
        })
        .collect()
}

/*
#[derive(Debug)]
struct DisposeDummyAxisTranslator {
    tracked: HashMap<OutletId, usize>,
}

impl crate::model::compact::Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>>
    for DisposeDummyAxisTranslator
{
    fn translate_op(
        &self,
        source: &ModelImpl<TypedFact, Box<dyn TypedOp>>,
        node: &BaseNode<TypedFact, Box<dyn TypedOp>>,
        _target: &mut ModelImpl<TypedFact, Box<dyn TypedOp>>,
        _mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<Box<dyn TypedOp>> {
        if let Some((input, axis)) = self.tracked.iter().find(|(k, _)| k.node == node.id) {
            Ok(node
                .op
                .dispose_dummy_axis(source, node, input.slot, *axis)?
                .unwrap_or_else(|| node.op.clone()))
        } else {
            Ok(node.op.clone())
        }
    }

    fn node_output_facts(
        &self,
        _source: &ModelImpl<TypedFact, Box<dyn TypedOp>>,
        node: &BaseNode<TypedFact, Box<dyn TypedOp>>,
        target: &mut ModelImpl<TypedFact, Box<dyn TypedOp>>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<TypedFact>> {
        let inputs = node
            .inputs
            .iter()
            .map(|i| target.outlet_fact(mapping[i]))
            .collect::<TractResult<TVec<_>>>()?;
        node.op.output_facts(&*inputs)
    }
}

pub fn dispose_dummy_axis(
    model: &TypedModel,
    input: usize,
    axis: usize,
) -> TractResult<TypedModel> {
    let input = model.input_outlets()?[input];
    let tracked = full_axis_tracking(model)?;
    let (tracked, disposable) =
        tracked.into_iter().find(|ax| ax.0.get(&input) == Some(&axis)).unwrap();
    let translator = DisposeDummyAxisTranslator { tracked };
    translator.translate_model(model)
}
*/
