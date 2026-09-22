//! Give a model's dataflow a batch axis, on axis 0 of every input and output, so
//! that a turn can seat several callers in one run.
//!
//! Every op the axis reaches is asked to host it, through the same
//! `TypedOp::change_axes` protocol declutter propagates axis changes with, and
//! the answer is accepted only if the op **carries the axis through** to all of
//! its outputs. That is a stricter contract than the protocol's own, which only
//! owes semantics at extent 1: an `EinSum` offered an extra leading axis on an
//! input takes it as a fresh contraction label, summing an axis of extent one
//! away, which is exact there and sums the seats together once the extent is the
//! batch. So the batched `EinSum` is built here instead, with the label on the
//! output as well, and an op that answers with anything but the axis carried
//! through is refused by name.
//!
//! An input the caller lists as **shared** is one every seat reads whole -- a
//! table, a language id. It keeps its own shape, and the ops that need their
//! operands to agree on rank read it through an `AxisOp::Add(0)`, so it
//! broadcasts across the turn.
//!
//! An input already sized by the batch symbol has that axis **moved** to the
//! front, as an interface change plus the inverse `AxisOp::Move` on the wire
//! behind it: an export putting the batch inside (`[T, B, H]` being the common
//! shape) becomes a graph edit declutter can absorb rather than a transpose the
//! runtime would pay per turn. Only axis 0 makes a seat's values a contiguous
//! run, which is what the laned runtime addresses and what stacking a turn's
//! answer back relies on.
//!
//! What this does not touch is state: a seat's own history is the laned
//! runtime's lane addressing, and a batch axis on the dataflow does not imply
//! one. Batchify gives you seats, lane addressing gives each seat its own past.

use std::collections::HashSet;

use crate::internal::*;
use crate::ops::change_axes::AxisOp;
use crate::ops::einsum::EinSum;
use crate::ops::source::TypedSource;
use crate::transform::ModelTransform;

#[derive(Debug, Default, serde::Deserialize)]
pub struct BatchifyConfig {
    /// Symbol sizing the batch axis. Defaults to "B".
    pub symbol: Option<String>,
    /// Inputs every seat shares, by node name.
    #[serde(default)]
    pub shared: Option<Vec<String>>,
}

#[derive(Debug)]
pub struct Batchify(pub BatchifyConfig);

impl ModelTransform for Batchify {
    fn name(&self) -> StaticName {
        "batchify".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let name = self.0.symbol.as_deref().unwrap_or("B");
        let symbol = model.symbols.sym(name);
        let shared = self.0.shared.as_deref().unwrap_or(&[]);
        *model = batchify(model, &symbol, shared)?;
        Ok(())
    }
}

/// What an op does with the batch axis arriving on some of its inputs: the op to
/// wire in its place, and the input slots that must read through an
/// `AxisOp::Add(0)` to agree on rank with the batched ones.
struct Hosted {
    op: Option<Box<dyn TypedOp>>,
    pad: TVec<usize>,
}

/// Batch axis on axis 0 of every input but the `shared` ones, sized by `batch`.
pub fn batchify(model: &TypedModel, batch: &Symbol, shared: &[String]) -> TractResult<TypedModel> {
    for name in shared {
        ensure!(
            model.input_outlets()?.iter().any(|o| &model.node(o.node).name == name),
            "{name} is listed as a shared input but is not a model input"
        );
    }
    let mut carried: HashSet<OutletId> = Default::default();
    let mut hosts: HashMap<usize, Hosted> = Default::default();
    for id in model.eval_order()? {
        let node = model.node(id);
        if let Some(source) = node.op_as::<TypedSource>() {
            if !shared.contains(&node.name) && carrier(&source.fact, batch).is_none() {
                carried.insert(id.into());
            }
            continue;
        }
        let batched: TVec<usize> = node
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| carried.contains(input))
            .map(|(slot, _)| slot)
            .collect();
        if batched.is_empty() {
            continue;
        }
        let hosted = host(model, node, &batched, &carried)
            .with_context(|| format!("Giving {node} a batch axis"))?;
        hosts.insert(id, hosted);
        carried.extend((0..node.outputs.len()).map(|slot| OutletId::new(id, slot)));
    }
    let wired = wire(model, batch, shared, &hosts)?;
    check(wired, batch)
}

/// The batch axis an input carries, if any: the first axis sized by an
/// expression the batch symbol is in.
fn carrier(fact: &TypedFact, batch: &Symbol) -> Option<usize> {
    fact.shape.iter().position(|dim| dim.symbols().contains(batch))
}

/// Asks an op to host the batch axis on the inputs that carry it, and refuses an
/// answer that does not carry it through to every output.
fn host(
    model: &TypedModel,
    node: &TypedNode,
    batched: &[usize],
    carried: &HashSet<OutletId>,
) -> TractResult<Hosted> {
    if let Some(einsum) = node.op_as::<EinSum>() {
        return batched_einsum(einsum, batched);
    }
    let change = AxisOp::Add(0);
    let consequence = node
        .op
        .change_axes(model, node, InOut::In(batched[0]), &change)?
        .context("the op takes no extra leading axis at all")?;
    for slot in 0..node.outputs.len() {
        ensure!(
            consequence.wire_changes.contains(&(InOut::Out(slot), change.clone())),
            "the op takes an extra leading axis on an input but does not carry it to output {slot}, \
             so the seats would be folded into one another"
        );
    }
    let pad = consequence
        .wire_changes
        .iter()
        .filter_map(|(io, _)| match io {
            InOut::In(slot) if !carried.contains(&node.inputs[*slot]) => Some(*slot),
            _ => None,
        })
        .collect();
    Ok(Hosted { op: consequence.substitute_op, pad })
}

/// An `EinSum` batched by one label of its own, on axis 0 of the batched inputs
/// and of every output. Its `change_axes` would rather contract the axis away,
/// which is the same thing only while the extent is one.
fn batched_einsum(einsum: &EinSum, batched: &[usize]) -> TractResult<Hosted> {
    let label = einsum.axes.available_label();
    let mut axes = einsum.axes.clone().with_extra_axis(label, InOut::In(batched[0]), 0)?;
    for slot in &batched[1..] {
        axes = axes.with_extra_axis_occurency(label, InOut::In(*slot), 0)?;
    }
    for slot in 0..axes.output_count() {
        axes = axes.with_extra_axis_occurency(label, InOut::Out(slot), 0)?;
    }
    let op = EinSum { axes, ..einsum.clone() };
    Ok(Hosted { op: Some(Box::new(op)), pad: tvec!() })
}

/// The batchified model, rebuilt through its ops so that every fact comes from
/// `output_facts` with the batch extent in place rather than from the facts the
/// unbatched model stored.
fn wire(
    model: &TypedModel,
    batch: &Symbol,
    shared: &[String],
    hosts: &HashMap<usize, Hosted>,
) -> TractResult<TypedModel> {
    let mut target = TypedModel { symbols: model.symbols.clone(), ..TypedModel::default() };
    let mut mapping: HashMap<OutletId, OutletId> = Default::default();
    let mut interface: HashMap<OutletId, OutletId> = Default::default();
    for id in model.eval_order()? {
        let node = model.node(id);
        let wires = if let Some(source) = node.op_as::<TypedSource>() {
            let (source, interior) =
                batched_source(&mut target, node, source, batch, !shared.contains(&node.name))?;
            interface.insert(id.into(), source);
            tvec!(interior)
        } else {
            let hosted = hosts.get(&id);
            let mut inputs: TVec<OutletId> =
                node.inputs.iter().map(|input| mapping[input]).collect();
            for slot in hosted.map(|h| &*h.pad).unwrap_or(&[]) {
                inputs[*slot] = target.wire_node(
                    format!("{}.batchify.rank.{slot}", node.name),
                    AxisOp::Add(0),
                    &[inputs[*slot]],
                )?[0];
            }
            let op = hosted.and_then(|h| h.op.clone()).unwrap_or_else(|| node.op.clone());
            target
                .wire_node(&node.name, op, &inputs)
                .with_context(|| format!("Wiring {node} with a batch axis"))?
        };
        for (slot, wire) in wires.into_iter().enumerate() {
            let outlet = OutletId::new(id, slot);
            if let Some(label) = model.outlet_label(outlet) {
                target.set_outlet_label(wire, label.to_string())?;
            }
            mapping.insert(outlet, wire);
        }
    }
    let inputs: Vec<OutletId> = model.input_outlets()?.iter().map(|i| interface[i]).collect();
    let mut outputs: Vec<OutletId> = model.output_outlets()?.iter().map(|o| mapping[o]).collect();
    for (ix, output) in outputs.iter_mut().enumerate() {
        let fact = target.outlet_fact(*output)?.clone();
        if let Some(axis) = carrier(&fact, batch).filter(|axis| *axis > 0) {
            let label = target.outlet_label(*output).map(|l| l.to_string());
            *output = target.wire_node(
                format!("batchify.move.output.{ix}"),
                AxisOp::Move(axis, 0),
                &[*output],
            )?[0];
            if let Some(label) = label {
                target.set_outlet_label(*output, label)?;
            }
        }
    }
    target.set_input_outlets(&inputs)?;
    target.select_output_outlets(&outputs)?;
    Ok(target)
}

/// A source of the batchified model: a batched one carries the batch on axis 0,
/// by gaining the axis or by having the one it already carries moved there, and
/// the wire behind it moves it back so that the graph sees the layout it was
/// built for.
fn batched_source(
    target: &mut TypedModel,
    node: &TypedNode,
    source: &TypedSource,
    batch: &Symbol,
    batched: bool,
) -> TractResult<(OutletId, OutletId)> {
    let carrier = carrier(&source.fact, batch);
    if !batched || carrier == Some(0) {
        let wire = target.wire_node(&node.name, source.clone(), &[])?[0];
        return Ok((wire, wire));
    }
    let mut shape = source.fact.shape.to_tvec();
    let mut restore = None;
    match carrier {
        Some(axis) => {
            let dim = shape.remove(axis);
            shape.insert(0, dim);
            restore = Some(AxisOp::Move(0, axis));
        }
        None => shape.insert(0, batch.to_dim()),
    }
    let mut fact = source.fact.clone();
    fact.shape = shape.into();
    let wire = target.wire_node(&node.name, TypedSource::new(fact), &[])?[0];
    match restore {
        Some(restore) => {
            let interior =
                target.wire_node(format!("{}.batchify.move", node.name), restore, &[wire])?[0];
            Ok((wire, interior))
        }
        None => Ok((wire, wire)),
    }
}

/// Fails on a model the laned runtime would refuse anyway, naming what it found:
/// a model answering nothing per seat has no turn to run.
fn check(model: TypedModel, batch: &Symbol) -> TractResult<TypedModel> {
    let batch = batch.to_dim();
    let mut shared = vec![];
    for (ix, outlet) in model.output_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*outlet)?;
        if fact.shape.first() == Some(&batch) {
            return Ok(model);
        }
        shared.push(format!("output {ix} is {:?}", fact.shape));
    }
    bail!("No output carries the batch axis on axis 0: {}", shared.join(", "))
}

register_model_transform!("batchify", BatchifyConfig, |config| Ok(Box::new(Batchify(config))));
