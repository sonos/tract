//! Give the batch axis to the wires a model computes without any data input,
//! before pulsification turns their windows into state.
//!
//! A **data-free** wire is one no `Source` feeds: a `Range` and the shape
//! arithmetic over it, computed from positions alone. Nothing batched flows in,
//! so no batch axis reaches it either, and the extent-1 axis an export leaves in
//! front of such a wire is a **placeholder** — broadcasting makes it free to
//! narrow away. Pulsification then windows those wires into `Delay` /
//! `PulsePad`, whose buffers are session state, and a state whose axis 0 is not
//! the batch axis cannot be lane-addressed: one buffer would serve every stream
//! of a batched turn at once.
//!
//! Only the data-free wires pulsification turns into state are **widened**, and
//! what marks them is carrying one symbol on two axes: a wire quadratic in the
//! stream is one pulsification windows, and a window is a buffer. The rel-pos
//! embedding table of a conformer is data-free too, and linear in the stream —
//! it is read, never buffered, and is left alone.
//!
//! The rewrite prepends the batch axis where the widened wires start
//! (`AxisOp::Add(0)` then a `MultiBroadcastTo`), moves every `AxisOp` axis right
//! by one on the way down, and drops the placeholder — whose rank slot is what
//! pays for the new axis. Every other op rides the extra axis through
//! elementwise broadcasting. What checks that is the assertion that each widened
//! wire's fact is exactly the old fact with the batch dim prepended: an op
//! carrying an axis in a field of its own fails here instead of pulsifying into
//! a silently shared buffer.
//!
//! The narrow name is deliberate: `Batchify` is reserved for the transform that
//! gives a model's inputs and outputs a batch axis, a different job at a
//! different level. This one only widens wires that have no batch axis to begin
//! with, and only where the state is.

use std::collections::HashSet;

use crate::internal::*;
use tract_core::model::translator::Translate;
use tract_core::ops::array::MultiBroadcastTo;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::wire_rank_broadcast;
use tract_core::ops::konst::Const;
use tract_core::ops::source::TypedSource;
use tract_core::transform::ModelTransform;

#[derive(Debug, Default, serde::Deserialize)]
pub struct BatchifyDataFreeConfig {
    /// Symbol sizing the model's batch axis. Defaults to "BATCH".
    pub symbol: Option<String>,
}

#[derive(Debug)]
pub struct BatchifyDataFree(pub BatchifyDataFreeConfig);

impl ModelTransform for BatchifyDataFree {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "batchify_data_free".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let name = self.0.symbol.as_deref().unwrap_or("BATCH");
        let symbol = model.symbols.sym(name);
        // A symbol no input is sized by would give the rewritten wires a dim
        // nothing ever binds, which only fails at run time.
        let mut carried = false;
        for outlet in model.input_outlets()? {
            let fact = model.outlet_fact(*outlet)?;
            carried |= fact.shape.iter().any(|d| d.symbols().contains(&symbol));
        }
        ensure!(carried, "No model input is sized by {name}, so there is no batch axis to join");
        *model = batchify_data_free(model, &symbol.to_dim())?;
        Ok(())
    }
}

/// Nodes computing a value no `Source` feeds. Constants are left out: they carry
/// no batch axis and need none, broadcasting covers them.
fn data_free_nodes(model: &TypedModel) -> TractResult<HashSet<usize>> {
    let mut fed: HashSet<usize> = Default::default();
    let order = model.eval_order()?;
    for &id in &order {
        let node = model.node(id);
        if node.op_is::<TypedSource>() || node.inputs.iter().any(|i| fed.contains(&i.node)) {
            fed.insert(id);
        }
    }
    Ok(order
        .into_iter()
        .filter(|id| !fed.contains(id) && !model.node(*id).op_is::<Const>())
        .collect())
}

/// Facts recomputed from the ops. A stored fact can outlive the op that produced
/// it, and this is the subgraph where that shows: narrowing an export's batched
/// mask down to a data-free one leaves the batch symbol in the stored fact of a
/// wire whose extent is now 1, which would read here as "already batched".
fn data_free_facts(
    model: &TypedModel,
    data_free: &HashSet<usize>,
) -> TractResult<HashMap<OutletId, TypedFact>> {
    let mut facts: HashMap<OutletId, TypedFact> = Default::default();
    for id in model.eval_order()? {
        if !data_free.contains(&id) {
            continue;
        }
        let node = model.node(id);
        let mut inputs: TVec<TypedFact> = tvec!();
        for input in &node.inputs {
            inputs.push(match facts.get(input) {
                Some(fact) => fact.clone(),
                None => model.outlet_fact(*input)?.clone(),
            });
        }
        let inputs: TVec<&TypedFact> = inputs.iter().collect();
        for (slot, fact) in node.op.output_facts(&inputs)?.into_iter().enumerate() {
            facts.insert(OutletId::new(id, slot), fact);
        }
    }
    Ok(facts)
}

/// A shape carrying one symbol on two axes, which is what pulsification windows
/// into a buffer.
fn quadratic(fact: &TypedFact) -> bool {
    let mut per_symbol: HashMap<Symbol, usize> = Default::default();
    for dim in fact.shape.iter() {
        for symbol in dim.symbols() {
            *per_symbol.entry(symbol).or_default() += 1;
        }
    }
    per_symbol.values().any(|&axes| axes >= 2)
}

/// Outlets leaving the data-free subgraph towards state: quadratic in a symbol,
/// read by a node that has a data input, and not batched already.
fn exits(
    model: &TypedModel,
    data_free: &HashSet<usize>,
    facts: &HashMap<OutletId, TypedFact>,
    batch: &TDim,
) -> TractResult<Vec<OutletId>> {
    let outputs = model.output_outlets()?;
    let mut exits = vec![];
    for &id in data_free {
        for slot in 0..model.node(id).outputs.len() {
            let outlet = OutletId::new(id, slot);
            let fact = &facts[&outlet];
            if !quadratic(fact)
                || fact.shape.first().is_some_and(|d| d == batch)
                || !model.node(id).outputs[slot]
                    .successors
                    .iter()
                    .any(|s| !data_free.contains(&s.node))
            {
                continue;
            }
            ensure!(
                !outputs.contains(&outlet),
                "{} is a model output and has no data input, batchifying it would change the model interface",
                model.node(id)
            );
            exits.push(outlet);
        }
    }
    Ok(exits)
}

/// The `AxisOp::Add(0)` each exit reads its leading axis from: the placeholder
/// standing where the batch axis belongs. One is expected, so that what the
/// batch axis replaces is unambiguous.
fn find_placeholder(
    model: &TypedModel,
    data_free: &HashSet<usize>,
    exits: &[OutletId],
) -> TractResult<usize> {
    let mut todo: Vec<usize> = exits.iter().map(|o| o.node).collect();
    let mut seen: HashSet<usize> = Default::default();
    let mut found: HashSet<usize> = Default::default();
    while let Some(id) = todo.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = model.node(id);
        if matches!(node.op_as::<AxisOp>(), Some(AxisOp::Add(0))) {
            found.insert(id);
            continue;
        }
        let upstream: Vec<usize> =
            node.inputs.iter().map(|i| i.node).filter(|n| data_free.contains(n)).collect();
        ensure!(
            !upstream.is_empty(),
            "No batch axis placeholder between {node} and the data-free subgraph's exits"
        );
        todo.extend(upstream);
    }
    ensure!(
        found.len() == 1,
        "Expected one batch axis placeholder, found {:?}",
        found.iter().map(|&id| model.node(id).name.as_str()).collect::<Vec<_>>()
    );
    Ok(found.into_iter().next().unwrap())
}

/// The data-free nodes feeding `node`: the ones whose wires gain the batch axis.
fn nodes_to_widen(model: &TypedModel, data_free: &HashSet<usize>, node: usize) -> HashSet<usize> {
    let mut widened: HashSet<usize> = Default::default();
    let mut todo: Vec<usize> =
        model.node(node).inputs.iter().map(|i| i.node).filter(|n| data_free.contains(n)).collect();
    while let Some(id) = todo.pop() {
        if !widened.insert(id) {
            continue;
        }
        todo.extend(model.node(id).inputs.iter().map(|i| i.node).filter(|n| data_free.contains(n)));
    }
    widened
}

pub fn batchify_data_free(model: &TypedModel, batch: &TDim) -> TractResult<TypedModel> {
    let data_free = data_free_nodes(model)?;
    let facts = data_free_facts(model, &data_free)?;
    let exits = exits(model, &data_free, &facts, batch)?;
    if exits.is_empty() {
        return Ok(model.clone());
    }
    let placeholder = find_placeholder(model, &data_free, &exits)?;
    let widened = nodes_to_widen(model, &data_free, placeholder);
    Widen { batch: batch.clone(), nodes: widened, placeholder, facts }.translate_model(model)
}

/// Prepends the batch axis to the wires of `nodes`, and drops `placeholder`.
struct Widen {
    batch: TDim,
    nodes: HashSet<usize>,
    placeholder: usize,
    facts: HashMap<OutletId, TypedFact>,
}

impl std::fmt::Debug for Widen {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Widen({}, {} wires)", self.batch, self.nodes.len())
    }
}

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for Widen {
    fn translate_node(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
        if node.id == self.placeholder {
            return Ok(inputs);
        }
        if !self.nodes.contains(&node.id) {
            return target.wire_node(&node.name, node.op.clone(), &inputs);
        }
        ensure!(node.outputs.len() == 1, "{node} has several outputs, which batchify cannot widen");
        // `merge_incoming_change` answers the same question, and the same way
        // except where the op names axis 0 itself: there it reads the arriving
        // axis as the second of the two, which would leave the batch axis at 1
        // on that wire and at 0 on every other.
        let op: Box<dyn TypedOp> = match node.op_as::<AxisOp>() {
            Some(axis_op) => Box::new(axis_op.pad_left(1)),
            None => node.op.clone(),
        };
        // The extra axis breaks the rank match typed binary ops require, so the
        // operands that did not gain it -- constants, mostly -- are bumped the
        // way the model's own `bump_rank` wires already are.
        let inputs = if node.op_is::<TypedBinOp>() {
            wire_rank_broadcast(&node.name, target, &inputs)?
        } else {
            inputs
        };
        let mut wire = target.wire_node(&node.name, op, &inputs)?;
        if node.inputs.iter().all(|i| !self.nodes.contains(&i.node)) {
            wire =
                target.wire_node(format!("{}.batchify.axis", node.name), AxisOp::Add(0), &wire)?;
            let mut shape = target.outlet_fact(wire[0])?.shape.to_tvec();
            shape[0] = self.batch.clone();
            wire = target.wire_node(
                format!("{}.batchify.extent", node.name),
                MultiBroadcastTo { shape: shape.into() },
                &wire,
            )?;
        }
        let expected: TVec<TDim> = std::iter::once(self.batch.clone())
            .chain(self.facts[&node.id.into()].shape.iter().cloned())
            .collect();
        let got = target.outlet_fact(wire[0])?.shape.to_tvec();
        ensure!(got == expected, "Batchifying {node} gave shape {got:?}, expected {expected:?}");
        Ok(wire)
    }
}
