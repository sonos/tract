use crate::internal::*;
use crate::model::{TypedModel, TypedNode};
use crate::ops::identity::Identity;
use itertools::Itertools;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InOut {
    Out(usize),
    In(usize),
}

impl InOut {
    pub fn as_outlet<F: Clone + Fact + Hash, O: Clone + Hash>(
        &self,
        node: &Node<F, O>,
    ) -> OutletId {
        match self {
            InOut::In(ix) => node.inputs[*ix],
            InOut::Out(ix) => OutletId::new(node.id, *ix),
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub enum AxisOp {
    Add(usize),
    Rm(usize),
    Move(usize, usize),
    Reshape(usize, TVec<TDim>, TVec<TDim>),
}
use AxisOp::*;

impl PartialEq for AxisOp {
    fn eq(&self, other: &AxisOp) -> bool {
        if self.is_noop() && other.is_noop() {
            true
        } else if self.is_noop() != other.is_noop() {
            false
        } else {
            match (self, other) {
                (Add(a), Add(b)) | (Rm(a), Rm(b)) => a == b,
                (Move(f1, t1), Move(f2, t2)) => {
                    (f1 == f2 && t1 == t2)
                        || ((*t1 == f1 + 1 || *f1 == t1 + 1) && t2 == f1 && t1 == f2)
                }
                (Reshape(at1, f1, t1), Reshape(at2, f2, t2)) => at1 == at2 && f1 == f2 && t1 == t2,
                _ => false,
            }
        }
    }
}

impl AxisOp {
    fn canonical(&self) -> Cow<AxisOp> {
        match self {
            Move(from, to) if *from == to + 1 => Cow::Owned(Move(*to, *from)),
            other => Cow::Borrowed(other),
        }
    }

    pub fn transform_axis(&self, axis: usize) -> Option<usize> {
        match self.canonical().as_ref() {
            Add(ix) => Some(axis + (axis >= *ix) as usize),
            Rm(ix) => {
                if axis == *ix {
                    None
                } else {
                    Some(axis - (axis > *ix) as usize)
                }
            }
            Move(from, to) if from < to => {
                if axis < *from || axis > *to {
                    Some(axis)
                } else if axis == *from {
                    Some(*to)
                } else {
                    Some(axis - 1)
                }
            }
            Move(from, to) => {
                if axis < *to || axis > *from {
                    Some(axis)
                } else if axis == *from {
                    Some(*to)
                } else {
                    Some(axis + 1)
                }
            }
            Reshape(at, _, _) if axis < *at => Some(axis),
            Reshape(at, from, to) if axis >= at + from.len() => Some(axis + to.len() - from.len()),
            Reshape(_, _, _) => None,
        }
    }

    // if sucessful return Some()
    // first item is the Op we want to be replaced by. if none, we are now identity.
    // second item is the change to propagate. if none, the output is not
    // changed
    pub fn merge_incoming_change(
        &self,
        change: &AxisOp,
    ) -> Option<(Option<AxisOp>, Option<AxisOp>)> {
        match (self.canonical().as_ref(), change.canonical().as_ref()) {
            (Add(op), Add(c)) => {
                Some((Some(Add(op + (c < op) as usize)), Some(Add(c + (c >= op) as usize))))
            }
            (Add(op), Rm(c)) => {
                Some((Some(Add(op - (c < op) as usize)), Some(Rm(c + (c >= op) as usize))))
            }
            (Rm(op), Add(c)) => {
                Some((Some(Rm(op + (c <= op) as usize)), Some(Add(c - (op < c) as usize))))
            }
            (Rm(op), Rm(c)) => {
                Some((Some(Rm(op - (c < op) as usize)), Some(Rm(c - (op <= c) as usize))))
            }

            (Add(x), Move(from, to)) => {
                if x <= from.min(to) {
                    Some((Some(self.clone()), Some(Move(from + 1, to + 1))))
                } else if x > from.max(to) {
                    Some((Some(self.clone()), Some(change.clone())))
                } else {
                    None
                }
            }

            (Move(from, to), Add(x)) => {
                if x <= from.min(to) {
                    Some((Some(Move(from + 1, to + 1)), Some(Add(*x))))
                } else if x > from.max(to) {
                    Some((Some(Move(*from, *to)), Some(Add(*x))))
                } else {
                    None
                }
            }

            (Rm(x), Move(from, to)) => {
                // disabled these two as they kinda break axis tracking
                // semantics
                if x == from {
                    None
                // Some((Some(Rm(*to)), None))
                } else if x < from.min(to) {
                    Some((Some(self.clone()), Some(Move(from - 1, to - 1))))
                } else if x > from.max(to) {
                    Some((Some(self.clone()), Some(change.clone())))
                } else if from + 1 == *to && x == to {
                    // Some((Some(Rm(*from)), None))
                    None
                } else if from < to && x <= to {
                    Some((Some(Rm(x - 1)), Some(Move(*from, *to - 1))))
                } else {
                    Some((Some(Rm(x + 1)), Some(Move(*from - 1, *to))))
                }
            }

            (Move(from, to), Rm(x)) => {
                if x < from.min(to) {
                    Some((Some(Move(from - 1, to - 1)), Some(Rm(*x))))
                } else if x > from.max(to) {
                    Some((Some(Move(*from, *to)), Some(Rm(*x))))
                } else {
                    None
                }
            }

            (Add(op), Reshape(at, from, to)) => {
                if op <= at {
                    Some((Some(Add(*op)), Some(Reshape(at + 1, from.clone(), to.clone()))))
                } else if *op > at + from.len() {
                    Some((
                        Some(Add(*op + to.len() - from.len())),
                        Some(Reshape(*at, from.clone(), to.clone())),
                    ))
                } else {
                    None
                }
            }
            (Rm(op), Reshape(at, from, to)) => {
                if op < at {
                    Some((Some(Rm(*op)), Some(Reshape(at - 1, from.clone(), to.clone()))))
                } else if *op > at + from.len() {
                    Some((
                        Some(Rm(*op + to.len() - from.len())),
                        Some(Reshape(*at, from.clone(), to.clone())),
                    ))
                } else {
                    None
                }
            }
            (Reshape(at, from, to), Add(change)) => {
                if change < at {
                    Some((Some(Reshape(at + 1, from.clone(), to.clone())), Some(Add(*change))))
                } else if *change > *at + from.len() {
                    Some((
                        Some(Reshape(*at, from.clone(), to.clone())),
                        Some(Add(change + to.len() - from.len())),
                    ))
                } else {
                    None
                }
            }
            (Reshape(at, from, to), Rm(change)) => {
                if change < at {
                    Some((Some(Reshape(at - 1, from.clone(), to.clone())), Some(Rm(*change))))
                } else if *change > *at + from.len() {
                    Some((
                        Some(Reshape(*at, from.clone(), to.clone())),
                        Some(Rm(change + to.len() - from.len())),
                    ))
                } else {
                    None
                }
            }
            (Reshape(_, _, _), Move(_, _)) => None, // todo, some are manageable
            (Move(_, _), Reshape(_, _, _)) => None, // todo, some are manageable
            (Reshape(_, _, _), Reshape(_, _, _)) => None, // todo, some are manageable
            _ => None,
        }
    }

    pub fn change_shape_array<D: DimLike>(&self, shape: &mut TVec<D>) -> TractResult<()> {
        use std::convert::TryInto;
        match self.canonical().as_ref() {
            Add(ix) => shape.insert(*ix, D::one()),
            Rm(ix) => {
                shape.remove(*ix);
            }
            Move(from, to) => {
                let axis = shape.remove(*from);
                shape.insert(*to, axis);
            }
            Reshape(at, from, to) => {
                for _ in from {
                    shape.remove(*at);
                }
                for d in to.iter().rev() {
                    shape.insert(*at, d.try_into()?);
                }
            }
        }
        Ok(())
    }

    pub fn change_shape(&self, shape: &mut ShapeFact) -> TractResult<()> {
        match self.canonical().as_ref() {
            Add(ix) => shape.insert_axis(*ix),
            Rm(ix) => {
                if shape.rank() <= *ix {
                    bail!("Attempt to remove {} axis on shape {:?}", ix, shape);
                }
                if shape[*ix] != 1.to_dim() {
                    bail!("Removing a non-trivial axis.");
                }
                shape.remove_axis(*ix)
            }
            _ => {
                let mut array = shape.to_tvec();
                self.change_shape_array(&mut array)?;
                let mut new_shape = ShapeFact::from_dims(array);
                std::mem::swap(shape, &mut new_shape);
                Ok(())
            }
        }
    }

    pub fn change_tensor(&self, tensor: &mut Tensor) -> TractResult<()> {
        self.change_tensor_broadcast(tensor, false)
    }

    pub fn change_tensor_broadcast_aware(&self, tensor: &mut Tensor) -> TractResult<()> {
        self.change_tensor_broadcast(tensor, true)
    }

    fn change_tensor_broadcast(&self, tensor: &mut Tensor, broadcasting: bool) -> TractResult<()> {
        match self.canonical().as_ref() {
            Add(ix) => tensor.insert_axis(*ix),
            Rm(ix) => tensor.remove_axis(*ix),
            Move(from, to) => {
                let mut permutation: Vec<usize> = (0..tensor.rank()).collect();
                permutation.remove(*from);
                permutation.insert(*to, *from);
                let mut tmp = tensor.clone().permute_axes(&permutation)?;
                std::mem::swap(tensor, &mut tmp);
                Ok(())
            }
            Reshape(at, from, to) => {
                let mut shape: TVec<usize> = tensor.shape().into();
                self.change_shape_array(&mut shape)?;
                if tensor.set_shape(&shape).is_ok() {
                    Ok(())
                } else if broadcasting
                    && tensor.shape().iter().skip(*at).take(from.len()).all(|d| *d == 1)
                {
                    if from.len() > to.len() {
                        for _ in to.len()..from.len() {
                            tensor.insert_axis(*at)?;
                        }
                    }
                    if to.len() > from.len() {
                        for _ in from.len()..to.len() {
                            tensor.remove_axis(*at)?;
                        }
                    }
                    Ok(())
                } else {
                    bail!(
                        "Invalid reshaping: {:?} on tensor {:?} (broadcasting allowed: {:?})",
                        self,
                        tensor,
                        broadcasting
                    )
                }
            }
        }
    }

    pub fn recip(&self) -> AxisOp {
        match self.canonical().as_ref() {
            Add(ix) => Rm(*ix),
            Rm(ix) => Add(*ix),
            Move(from, to) if from == to => self.clone(),
            Move(from, to) if *from + 1 == *to => self.clone(),
            Move(from, to) if *from == *to + 1 => {
                unreachable!();
            }
            Move(from, to) => Move(*to, *from),
            Reshape(at, from, to) => Reshape(*at, to.clone(), from.clone()),
        }
    }

    pub fn is_noop(&self) -> bool {
        match self {
            Move(f, t) if f == t => true,
            Reshape(_, f, t) if f == t => true,
            _ => false,
        }
    }

    pub fn only_shape(&self) -> bool {
        if self.is_noop() {
            return true;
        }
        if let Move(_, _) = self {
            false
        } else {
            true
        }
    }
}

#[derive(Clone, Debug)]
pub struct AxisChange {
    pub outlet: OutletId,
    pub op: AxisOp,
}

#[derive(Clone, Default, Debug)]
pub struct AxisChangeConsequence {
    pub substitute_op: Option<Box<dyn TypedOp>>,
    pub wire_changes: TVec<(InOut, AxisOp)>,
}

impl AxisChangeConsequence {
    pub fn new(
        _model: &TypedModel,
        node: &TypedNode,
        op: Option<Box<dyn TypedOp>>,
        axis_op: &AxisOp,
    ) -> AxisChangeConsequence {
        let mut wire_changes = tvec!();
        for i in 0..node.inputs.len() {
            wire_changes.push((InOut::In(i), axis_op.clone()));
        }
        for i in 0..node.outputs.len() {
            wire_changes.push((InOut::Out(i), axis_op.clone()));
        }
        AxisChangeConsequence { wire_changes, substitute_op: op }
    }
}

impl Op for AxisOp {
    fn name(&self) -> Cow<str> {
        match self {
            Add(_) => "AddAxis".into(),
            Rm(_) => "RmAxis".into(),
            Move(_, _) => "MoveAxis".into(),
            Reshape(_, _, _) => "Reshape".into(),
        }
    }

    fn info(&self) -> TractResult<Vec<String>> {
        match self {
            Add(axis) | Rm(axis) => Ok(vec![format!("Axis: {}", axis)]),
            Move(from, to) => Ok(vec![format!("Axis {} to {}", from, to)]),
            Reshape(at, from, to) => Ok(vec![format!(
                "Axes starting at {}: {:?} to {:?}",
                at,
                from.iter().join(","),
                to.iter().join(",")
            )]),
        }
    }

    op_core_lir_mir!();
    op_as_typed_op!();
}

impl_dyn_hash!(AxisOp);

impl EvalOp for AxisOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut input = args_1!(inputs).into_tensor();
        self.change_tensor(&mut input)?;
        Ok(tvec!(input.into_arc_tensor()))
    }
}

impl TypedOp for AxisOp {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.clone();
        self.change_shape(&mut shape)?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, shape)))
    }

    fn invariants(&self, _model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let mut axes = vec![];
        for i in 0..node.outputs[0].fact.shape.rank() {
            if let Some(out) = self.transform_axis(i) {
                axes.push(AxisInfo {
                    inputs: tvec!(Some(i)),
                    outputs: tvec!(Some(out)),
                    period: 1,
                    disposable: true,
                });
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.is_noop() {
            Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
        } else {
            Ok(None)
        }
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        Ok(tvec!((InOut::Out(0), self.recip()), (InOut::In(0), self.clone())))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let op = if let InOut::Out(0) = io {
            let more =
                if let Some(more) = self.recip().change_axes(model, node, InOut::In(0), &change)? {
                    more
                } else {
                    return Ok(None);
                };
            AxisChangeConsequence {
                substitute_op: more.substitute_op.map(|op| {
                    if let Some(op) = op.as_op().downcast_ref::<AxisOp>() {
                        Box::new(op.recip())
                    } else {
                        op // have to be identity
                    }
                }),
                wire_changes: more
                    .wire_changes
                    .into_iter()
                    .map(|wc| {
                        (if wc.0 == InOut::In(0) { InOut::Out(0) } else { InOut::In(0) }, wc.1)
                    })
                    .collect(),
            }
        } else if change == self {
            AxisChangeConsequence { substitute_op: Some(Box::new(Identity)), wire_changes: tvec!() }
        } else {
            let (new_op, new_change) = if let Some(pair) = self.merge_incoming_change(change) {
                pair
            } else {
                return Ok(None);
            };
            trace!(
                "  Change:{:?} self:{:?} -> change:{:?} op:{:?}",
                change,
                self,
                new_change,
                new_op
            );
            let substitute_op: Box<dyn TypedOp> =
                if let Some(o) = new_op { Box::new(o) as _ } else { Box::new(Identity) };
            let mut wire_changes = tvec!();
            if !change.is_noop() {
                wire_changes.push((InOut::In(0), change.clone()))
            }
            if let Some(new_change) = new_change {
                wire_changes.push((InOut::Out(0), new_change))
            }
            AxisChangeConsequence { substitute_op: Some(substitute_op), wire_changes }
        };
        Ok(Some(op))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = if let AxisOp::Reshape(axis, from, to) = self {
            AxisOp::Reshape(
                *axis,
                from.iter().map(|d| d.eval(&values)).collect(),
                to.iter().map(|d| d.eval(&values)).collect(),
            )
        } else {
            self.clone()
        };
        target.wire_node(&node.name, op, &[mapping[&node.inputs[0]]])
    }
}

pub fn change_axes(
    model: &TypedModel,
    change: &AxisChange,
    locked: &[OutletId],
    bounds: &[TVec<OutletId>],
) -> TractResult<Option<(TypedModelPatch, TVec<(InOut, AxisOp)>)>> {
    trace!("Considering change {:?}", change);
    let mut todo_changes = vec![(change.clone(), None)];
    let mut changed_wires = HashMap::new();
    changed_wires.insert(change.outlet, change.op.clone());
    let mut changed_ops: HashMap<usize, Box<dyn TypedOp>> = HashMap::new();
    while let Some((c, emitter)) = todo_changes.pop() {
        let outlets = if let Some(group) = bounds.iter().find(|b| b.contains(&c.outlet)) {
            group.clone()
        } else {
            tvec![c.outlet]
        };
        for outlet in outlets {
            if locked.contains(&outlet) {
                trace!("  Change {:?} blocked by locked interface {:?}", change, outlet);
                return Ok(None);
            }
            let mut nodes = vec![(outlet.node, InOut::Out(outlet.slot))];
            for inlet in model.outlet_successors(outlet) {
                nodes.push((inlet.node, InOut::In(inlet.slot)));
            }
            for (node_id, io) in nodes {
                if Some(node_id) == emitter {
                    continue;
                }
                let node = model.node(node_id);
                let more = node
                    .op
                    .change_axes(model, node, io, &c.op)
                    .with_context(|| format!("Propagating {:?} to node {}", change, node))?;
                if more.is_none() {
                    trace!("    Propagation of {:?} blocked by {}", change, node);
                    return Ok(None);
                }
                let AxisChangeConsequence { substitute_op, wire_changes } = more.unwrap();
                if let Some(op) = substitute_op {
                    trace!(
                        "    Change {:?} enters {} from {:?} would replace {:?} by {:?}",
                        c.op,
                        node,
                        io,
                        node.op,
                        op
                    );
                    changed_ops.insert(node.id, op);
                } else {
                    trace!(
                        "    Change {:?} enters {} from {:?} leaves it unchanged",
                        c.op,
                        node,
                        io,
                    );
                }
                for (wire, op) in wire_changes.into_iter() {
                    let outlet = wire.as_outlet(node);
                    if !changed_wires.contains_key(&outlet) {
                        changed_wires.insert(outlet, op.clone());
                        todo_changes.push((AxisChange { outlet, op }, Some(node_id)));
                    }
                }
            }
        }
    }
    trace!("Translating {:?} to patch", change);
    let mut patch = TypedModelPatch::new(format!("{:?}", change));
    let mut replaced_wires: HashMap<OutletId, OutletId> = HashMap::default();
    let nodes_to_replace = changed_wires
        .keys()
        .map(|o| o.node)
        .chain(changed_ops.keys().copied())
        .collect::<std::collections::HashSet<usize>>();
    for node_id in model.eval_order()? {
        let node = model.node(node_id);
        if nodes_to_replace.contains(&node_id) {
            let mut inputs = tvec!();
            for orig in &node.inputs {
                let tgt = replaced_wires
                    .entry(*orig)
                    .or_insert_with(|| patch.tap_model(model, *orig).unwrap());
                inputs.push(*tgt);
            }
            let op: Box<dyn TypedOp> =
                changed_ops.get(&node_id).cloned().unwrap_or_else(|| node.op.clone());
            let new_wires = patch.wire_node(&node.name, op, &inputs)?;
            if new_wires.len() == 1
                && patch.node(new_wires[0].node).op_is::<crate::ops::source::TypedSource>()
            {
                patch.inputs.insert(new_wires[0].node, node_id);
            }
            for (ix, w) in new_wires.iter().enumerate() {
                replaced_wires.insert((node_id, ix).into(), *w);
            }
        } else {
            for orig in &node.inputs {
                if let Some(replacement) = replaced_wires.get(&orig) {
                    patch.shunt_outside(model, *orig, *replacement)?;
                }
            }
        }
    }
    for output in model.output_outlets()? {
        if let Some(replacement) = replaced_wires.get(output) {
            unsafe {
                patch.shunt_outside_unchecked(*output, *replacement)?;
            }
        }
    }
    let mut interface_change = tvec!();
    for (ix, input) in model.input_outlets()?.iter().enumerate() {
        if let Some(change) = changed_wires.get(&input) {
            interface_change.push((InOut::In(ix), change.clone()));
        }
    }
    for (ix, output) in model.output_outlets()?.iter().enumerate() {
        if let Some(change) = changed_wires.get(&output) {
            interface_change.push((InOut::Out(ix), change.clone()));
        }
    }
    debug_assert!(
        patch.model.nodes.iter().map(|n| &n.name).collect::<std::collections::HashSet<_>>().len()
            == patch.model.nodes.len()
    );
    Ok(Some((patch, interface_change)))
}

// a, b, c is a <- b, b <- c, c <- a
fn perm_to_cycles(perm: &[usize]) -> TVec<TVec<usize>> {
    let mut cycles: TVec<TVec<usize>> = tvec!();
    let mut done = 0;
    while done < perm.len() {
        if perm[done] == done || cycles.iter().any(|c| c.contains(&done)) {
            done += 1;
            continue;
        }
        let mut cycle = tvec!();
        let mut current = done;
        loop {
            cycle.push(current);
            current = perm[current];
            if current == done {
                break;
            }
        }
        cycles.push(cycle)
    }
    cycles
}

fn is_rotation_cycle(cycle: &[usize]) -> Option<(usize, usize)> {
    if cycle.windows(2).all(|w| w[0] + 1 == w[1]) {
        Some((cycle[0], cycle[cycle.len() - 1]))
    } else if cycle[1..cycle.len()].windows(2).all(|w| w[0] - 1 == w[1])
        && cycle[cycle.len() - 1] - 1 == cycle[0]
    {
        Some((cycle[1], cycle[0]))
    } else {
        None
    }
}

fn perm_to_atoms(input: &[usize]) -> TVec<(usize, usize)> {
    let mut changes: TVec<(usize, usize)> = tvec!();
    'top: loop {
        let mut reached: TVec<usize> = (0..input.len()).collect();
        changes.iter().for_each(|(f, t)| {
            let axis = reached.remove(*f);
            reached.insert(*t, axis);
        });
        if &*reached == input {
            return changes;
        }
        let remaining: TVec<usize> =
            input.iter().map(|x| reached.iter().position(|y| y == x).unwrap()).collect();
        let cycles = perm_to_cycles(&remaining);
        for cycle in &cycles {
            if let Some(rot) = is_rotation_cycle(&cycle) {
                changes.push(rot);
                continue 'top;
            }
        }
        changes.push((cycles[0][1], cycles[0][0]));
    }
}

pub fn perm_to_ops(input: &[usize]) -> TVec<AxisOp> {
    perm_to_atoms(input).into_iter().map(|pair| AxisOp::Move(pair.0, pair.1)).collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_perm_to_cycles() {
        assert_eq!(perm_to_cycles(&[1, 2, 0]), tvec!(tvec!(0, 1, 2)));
        assert_eq!(perm_to_cycles(&[2, 0, 1]), tvec!(tvec!(0, 2, 1)));
        assert_eq!(perm_to_cycles(&[1, 2, 3, 0]), tvec!(tvec!(0, 1, 2, 3)));
        assert_eq!(perm_to_cycles(&[3, 0, 1, 2]), tvec!(tvec!(0, 3, 2, 1)));
        assert_eq!(perm_to_cycles(&[3, 1, 2, 0, 4]), tvec!(tvec!(0, 3)));
    }

    #[test]
    fn is_rotation() {
        assert_eq!(is_rotation_cycle(&[0, 1, 2]), Some((0, 2)));
        assert_eq!(is_rotation_cycle(&[0, 2, 1]), Some((2, 0)));
    }

    #[test]
    fn test_perm_one_rotation() {
        assert_eq!(perm_to_atoms(&[1, 2, 0, 3, 4]), tvec!((0, 2)));
    }

    #[test]
    fn test_perm_two_rotations() {
        assert_eq!(perm_to_atoms(&[1, 2, 0, 4, 3]), tvec!((0, 2), (3, 4)));
    }

    #[test]
    fn test_perm_complex() {
        assert_eq!(perm_to_atoms(&[3, 1, 2, 0, 4]), tvec!((3, 0), (1, 3)));
    }

    // ADD-ADD

    //                          Op
    //           b,c   ------|Add(0)|----->        n,b,c
    //   Add(0)                                            Add(1)
    //         a,b,c   ------|Add(0)|----->        a,n,b,c
    #[test]
    pub fn transform_op_add_0_add_0() {
        let change = Add(0);
        let op = Add(0);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Add(0)), Some(Add(1)))));
    }

    //                          Op
    //           b,c   ------|Add(1)|----->        b,n,c
    //   Add(0)                                                 Add(0)
    //         a,b,c   ------|Add(2)|----->        a,b,n,c
    #[test]
    pub fn transform_op_add_0_add_1() {
        let change = Add(0);
        let op = Add(1);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Add(2)), Some(Add(0)))));
    }

    //                          Op
    //           a,c   ------|Add(0)|----->        n,a,c
    //   Add(1)                                                 Add(2)
    //         a,b,c   ------|Add(0)|----->        n,a,b,c
    #[test]
    pub fn transform_op_add_1_add_0() {
        let change = Add(1);
        let op = Add(0);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Add(0)), Some(Add(2)))));
    }

    //                          Op
    //         a,b,c   ------|Rm(1)|----->         a,c
    //   Rm(0)                                             Rm(0)
    //           b,c   ------|Rm(0)|----->         c
    #[test]
    pub fn transform_op_rm_0_rm_1() {
        let change = Rm(0);
        let op = Rm(1);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(0)), Some(Rm(0)))));
    }

    //                          Op
    //         a,b,c   ------|Rm(0)|----->         b,c
    //   Rm(1)                                             Rm(0)
    //           a,c   ------|Rm(0)|----->         c
    #[test]
    pub fn transform_op_rm_1_rm_0() {
        let change = Rm(1);
        let op = Rm(0);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(0)), Some(Rm(0)))));
    }

    // ADD - RM

    //                          Op
    //          b,c     ------|Rm(0)|------>        c
    //   Add(0)                                                 Add(0)
    //          a,b,c   ------|Rm(1)|----->         a,c
    #[test]
    pub fn transform_op_add_0_rm_0() {
        let change = Add(0);
        let op = Rm(0);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(1)), Some(Add(0)))));
    }

    //                          Op
    //          b,c     ------|Rm(1)|------>        b
    //   Add(0)                                                 Add(0)
    //          a,b,c   ------|Rm(2)|----->         a,b
    #[test]
    pub fn transform_op_add_0_rm_1() {
        let change = Add(0);
        let op = Rm(1);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(2)), Some(Add(0)))));
    }

    //                          Op
    //          a,c     ------|Rm(0)|------>        c
    //   Add(1)                                                 Add(0)
    //          a,b,c   ------|Rm(0)|----->         b,c
    #[test]
    pub fn transform_op_add_1_rm_0() {
        let change = Add(1);
        let op = Rm(0);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(0)), Some(Add(0)))));
    }

    // RM - ADD

    //                          Op
    //         a,b,c   ------|Add(0)|----->        X,a,b,c
    //   Rm(1)                                                 Rm(2)
    //           a,c   ------|Add(0)|----->        X,a,c
    #[test]
    pub fn transform_op_rm_1_add_0() {
        let change = Rm(1);
        let op = Add(0);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Add(0)), Some(Rm(2)))));
    }

    //                          Op
    //         a,b,c   ------|Add(1)|----->        a,X,b,c
    //   Rm(0)                                                 Rm(0)
    //           b,c   ------|Add(0)|----->        X,b,c
    #[test]
    pub fn transform_op_rm_0_add_1() {
        let change = Rm(0);
        let op = Add(1);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Add(0)), Some(Rm(0)))));
    }

    //                          Op
    //         a,b,c   ------|Rm(2)|----->        a,b
    //   Move(0, 2)                                           Move(0,1)
    //         b,c,a   ------|Rm(1)|----->        b,a
    #[test]
    pub fn transform_op_mv_02_rm_2() {
        let change = Move(0, 2);
        let op = Rm(2);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(1)), Some(Move(0, 1)))));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug)]
    struct ComposeProblem {
        input: TVec<usize>,
        ops: TVec<AxisOp>,
    }

    impl Arbitrary for AxisOp {
        type Parameters = TVec<usize>;
        type Strategy = BoxedStrategy<AxisOp>;
        fn arbitrary_with(shape: TVec<usize>) -> Self::Strategy {
            let mut ops: BoxedStrategy<AxisOp> =
                (0usize..shape.len() + 1).prop_map(|ax| Add(ax)).boxed();
            if shape.len() > 1 {
                ops = ops
                    .prop_union(
                        (0..shape.len(), 0..shape.len()).prop_map(|(a, b)| Move(a, b)).boxed(),
                    )
                    .boxed()
            }
            let rms =
                (0..shape.len()).filter(|&ax| shape[ax] == 1).map(|ax| Rm(ax)).collect::<Vec<_>>();
            if rms.len() > 0 {
                ops = ops
                    .prop_union((0..rms.len()).prop_map(move |rm| rms[rm].clone()).boxed())
                    .boxed()
            }
            let mergeable: Vec<AxisOp> = shape
                .windows(2)
                .enumerate()
                .filter(|(_, w)| w[0] > 1 && w[1] > 1)
                .map(|(ix, w)| {
                    Reshape(ix, tvec!(w[0].to_dim(), w[1].to_dim()), tvec!((w[0] * w[1]).to_dim()))
                })
                .collect();
            if mergeable.len() > 1 {
                ops = ops
                    .prop_union(
                        (0..mergeable.len()).prop_map(move |ix| mergeable[ix].clone()).boxed(),
                    )
                    .boxed()
            }
            ops
        }
    }

    impl Arbitrary for ComposeProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<ComposeProblem>;
        fn arbitrary_with(_args: ()) -> Self::Strategy {
            let input = proptest::collection::vec(1usize..4, 1usize..4);
            fn tail(len: usize, shape: TVec<usize>) -> BoxedStrategy<TVec<AxisOp>> {
                if len == 0 {
                    Just(tvec!()).boxed()
                } else {
                    AxisOp::arbitrary_with(shape.clone().into())
                        .prop_flat_map(move |op| {
                            let mut shape = shape.clone();
                            op.change_shape_array(&mut shape).unwrap();
                            tail(len - 1, shape.clone()).prop_map(move |mut t| {
                                t.insert(0, op.clone());
                                t
                            })
                        })
                        .boxed()
                }
            }
            (input, 1usize..=5)
                .prop_flat_map(|(input, len)| (Just(input.clone()), tail(len, input.into())))
                .prop_map(|(input, ops)| ComposeProblem { input: input.into(), ops })
                .boxed()
        }
    }

    impl ComposeProblem {
        pub fn model(&self) -> TractResult<TypedModel> {
            let mut model = TypedModel::default();
            let mut wire =
                model.add_source("source", TypedFact::dt_shape(i64::datum_type(), &self.input))?;
            for (ix, op) in self.ops.iter().enumerate() {
                wire = model.wire_node(format!("op_{}", ix), op.clone(), &[wire])?[0];
            }
            model.set_output_outlets(&[wire])?;
            Ok(model)
        }

        fn input(&self) -> TractResult<Tensor> {
            unsafe {
                let mut t = Tensor::uninitialized::<i64>(&*self.input)?;
                for i in 0..t.len() {
                    t.as_slice_mut().unwrap()[i] = i as i64;
                }
                Ok(t)
            }
        }

        fn check(&self) -> TractResult<()> {
            crate::setup_test_logger();
            let input = self.input()?;
            let model = self.model()?;
            let raw = model.into_runnable()?.run(tvec!(input.clone()))?;
            let optimized = self.model()?.declutter()?;
            let opt = optimized.into_runnable()?.run(tvec!(input))?;
            opt[0].close_enough(&raw[0], false)
        }
    }

    proptest! {
        #[test]
        fn recip(pb in any::<AxisOp>()) {
            assert_eq!(pb.recip().recip(), pb);
        }

        #[test]
        fn axis_ops(pb in any::<ComposeProblem>()) {
            pb.check().unwrap()
        }
    }

    #[test]
    fn add_0_rm_0() {
        let pb = ComposeProblem { input: tvec![1], ops: tvec![Add(0), Rm(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_move_01() {
        let pb = ComposeProblem { input: tvec![2], ops: tvec![Add(0), Move(0, 1)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_move_01_add_1() {
        let pb = ComposeProblem { input: tvec![2], ops: tvec![Add(0), Move(0, 1), Add(1)] };
        pb.check().unwrap();
    }

    #[test]
    fn recip_move_01() {
        let op = Move(1, 0);
        assert_eq!(op.recip().recip(), op);
    }

    #[test]
    fn recip_move_20() {
        let op = Move(2, 0);
        assert_eq!(op.recip().recip(), op);
    }

    #[test]
    fn recip_move_02() {
        let op = Move(0, 2);
        assert_eq!(op.recip().recip(), op);
    }

    #[test]
    fn add_0_add_1_move_02() {
        let pb = ComposeProblem { input: tvec![2], ops: tvec![Add(0), Add(1), Move(0, 2)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0() {
        let pb = ComposeProblem { input: tvec![1], ops: tvec![Add(0), Add(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0_move_02() {
        let pb = ComposeProblem { input: tvec![2], ops: tvec![Add(0), Add(0), Move(0, 2)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_2_move_12() {
        let pb = ComposeProblem { input: tvec![2], ops: tvec![Add(0), Add(2), Move(1, 2)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0_move_02_rm_0() {
        let pb = ComposeProblem { input: tvec![1], ops: tvec![Add(0), Add(0), Move(0, 2), Rm(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0_move_20_move_20() {
        let pb =
            ComposeProblem { input: tvec![2], ops: tvec![Add(0), Add(0), Move(2, 0), Move(2, 0)] };
        pb.check().unwrap();
    }

    #[test]
    fn move_01_add_0() {
        let pb = ComposeProblem { input: tvec![1, 1], ops: tvec![Move(0, 1), Add(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_move_02_move_02() {
        let pb = ComposeProblem { input: tvec![1, 1], ops: tvec![Add(0), Move(0, 2), Move(0, 2),] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_2_move_20_move_12_rm_2() {
        let pb = ComposeProblem {
            input: tvec![3],
            ops: tvec![Add(0), Add(2), Move(2, 0), Move(1, 2), Rm(2)],
        };
        pb.check().unwrap();
    }

    #[test]
    fn move_02_move_02() {
        let pb = ComposeProblem { input: tvec![2, 1, 1], ops: tvec![Move(0, 2), Move(0, 2)] };
        pb.check().unwrap();
    }

    #[test]
    fn rm_1_perm_10_add_0() {
        let pb = ComposeProblem { input: tvec![1, 1, 2], ops: tvec![Rm(1), Move(0, 1), Add(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_2_move_02_move_02() {
        let pb = ComposeProblem { input: tvec![3, 2], ops: tvec![Add(2), Move(0, 2), Move(0, 2)] };
        pb.check().unwrap();
    }

    #[test]
    fn move_01_move_20_move_20() {
        let pb = ComposeProblem {
            input: tvec![2, 3, 2],
            ops: tvec![Move(0, 1), Move(2, 0), Move(2, 0)],
        };
        pb.check().unwrap();
    }
}
