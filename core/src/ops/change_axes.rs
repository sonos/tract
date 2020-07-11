use crate::internal::*;
use crate::model::{TypedModel, TypedNode};
use crate::ops::identity::Identity;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InOut {
    Out(usize),
    In(usize),
}

impl InOut {
    pub fn as_outlet<F: Clone + Fact + Hash, O: Clone + Hash>(
        &self,
        node: &BaseNode<F, O>,
    ) -> OutletId {
        match self {
            InOut::In(ix) => node.inputs[*ix],
            InOut::Out(ix) => OutletId::new(node.id, *ix),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq)]
pub enum AxisOp {
    Add(usize),
    Rm(usize),
    Move(usize, usize),
    Reshape(usize, TVec<TDim>, TVec<TDim>),
}
use AxisOp::*;

impl AxisOp {
    fn check(&self) {
        assert!(!self.is_noop());
        if let Move(from, to) = self {
            assert!(*from != *to + 1, "Swap must be \"in left to right order\"");
        }
    }

    pub fn checked(self) -> Option<AxisOp> {
        if self.is_noop() {
            None
        } else if let Move(from, to) = self {
            if from == to + 1 {
                return Some(Move(to, from));
            }
            Some(Move(from, to))
        } else {
            Some(self)
        }
    }

    pub fn transform_axis(&self, axis: usize) -> Option<usize> {
        self.check();
        match self {
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
        self.check();
        change.check();
        dbg!(self, change);
        let r = match (self, change) {
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
                if x == from {
                    Some((Some(Rm(*to)), None))
                } else if x < from.min(to) {
                    Some((Some(self.clone()), Some(Move(from - 1, to - 1))))
                } else if x > from.max(to) {
                    Some((Some(self.clone()), Some(change.clone())))
                } else if from + 1 == *to && x == to {
                    Some((Some(Rm(*from)), None))
                } else if from < to && x <= to {
                    Some((Some(Rm(x - 1)), Move(*from, *to - 1).checked()))
                } else {
                    Some((Some(Rm(x + 1)), Move(*from - 1, *to).checked()))
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
                if op <= at {
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
        };
        eprintln!("op:{:?} c:{:?} -> {:?}", self, change, r);
        r
    }

    pub fn change_shape_array<D: DimLike>(&self, shape: &mut TVec<D>) {
        self.check();
        use std::convert::TryInto;
        match self {
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
                    shape.insert(*at, d.try_into().unwrap());
                }
            }
        }
    }

    pub fn change_shape(&self, shape: &mut ShapeFact) -> TractResult<()> {
        self.check();
        match self {
            Add(ix) => shape.insert_axis(*ix),
            Rm(ix) => {
                debug_assert_eq!(shape.dim(*ix), 1.to_dim(), "Removing a non-trivial axis.");
                shape.remove_axis(*ix)
            }
            _ => {
                let mut array = shape.to_tvec();
                self.change_shape_array(&mut array);
                let mut new_shape = ShapeFact::from_dims(array).unwrap();
                std::mem::swap(shape, &mut new_shape);
                Ok(())
            }
        }
    }

    pub fn change_tensor(&self, tensor: &mut Tensor) -> TractResult<()> {
        self.check();
        fn permute<T: Datum>(axes: &[usize], input: Tensor) -> TractResult<Tensor> {
            Ok(input.into_array::<T>()?.permuted_axes(axes).into_tensor())
        }
        match self {
            Add(ix) => tensor.insert_axis(*ix),
            Rm(ix) => tensor.remove_axis(*ix),
            Move(from, to) => {
                let mut permutation: Vec<usize> = (0..tensor.rank()).collect();
                permutation.remove(*from);
                permutation.insert(*to, *from);
                let mut tmp =
                    dispatch_datum!(permute(tensor.datum_type())(&permutation, tensor.clone()))?;
                std::mem::swap(tensor, &mut tmp);
                Ok(())
            }
            Reshape(_, _, _) => {
                let mut shape: TVec<usize> = tensor.shape().into();
                self.change_shape_array(&mut shape);
                unsafe { tensor.set_shape(&shape) }
                Ok(())
            }
        }
    }

    pub fn recip(&self) -> AxisOp {
        self.check();
        match self {
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
        if let Move(f, t) = self {
            f == t
        } else if let Reshape(_, f, t) = self {
            f == t
        } else {
            false
        }
    }

    pub fn only_shape(&self) -> bool {
        self.check();
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
        use crate::itertools::Itertools;
        match self {
            Add(axis) | Rm(axis) => Ok(vec![format!("Axis: {}", axis)]),
            Move(from, to) => Ok(vec![format!("Axis {} to {}", from, to)]),
            Reshape(at, from, to) => Ok(vec![format!(
                "Axes starting at {}: {:?} to {:?}",
                at,
                from.iter().join("x"),
                to.iter().join("x")
            )]),
        }
    }

    canonic!();
    op_core_lir_mir!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

tract_linalg::impl_dyn_hash!(AxisOp);

impl StatelessOp for AxisOp {
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
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, shape)?))
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

    fn pulsify(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        target.wire_node(&*node.name, self.clone(), &[input])
    }
}

impl PulsedOp for AxisOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape = inputs[0].shape.clone();
        self.change_shape_array(&mut fact.shape);
        fact.axis = self.transform_axis(fact.axis).ok_or("Invalid axis for pulsification")?;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

pub fn change_axes(
    model: &mut TypedModel,
    change: &AxisChange,
    locked: &[OutletId],
    bounds: &[TVec<OutletId>],
) -> TractResult<Option<HashMap<OutletId, AxisOp>>> {
    trace!("Trying to apply change {:?}", change);
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
                debug!("  Change {:?} blocked by locked interface {:?}", change, outlet);
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
                    .chain_err(|| format!("Propagating {:?} to node {}", change, node))?;
                if more.is_none() {
                    debug!("    Propagation of {:?} blocked by {}", change, node);
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
    debug!("Applying {:?}: {:?}", change, changed_ops);
    for node_id in model.eval_order()? {
        if let Some(new_op) = changed_ops.remove(&node_id) {
            model.node_mut(node_id).op = new_op;
        }
        let output_facts =
            model.node(node_id).op.output_facts(&model.node_input_facts(node_id)?)?;
        for (ix, f) in output_facts.into_iter().enumerate() {
            model.set_outlet_fact(OutletId::new(node_id, ix), f)?;
        }
    }
    Ok(Some(changed_wires))
}

#[cfg(test)]
mod test {
    use super::*;

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
                        (0..shape.len(), 0..shape.len())
                            .prop_filter_map("trivial", |(a, b)| Move(a, b).checked())
                            .boxed(),
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
                            op.change_shape_array(&mut shape);
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
            let mut wire = model
                .add_source("source", TypedFact::dt_shape(i64::datum_type(), &*self.input)?)?;
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
            // dbg!(&model);
            let raw = model.into_runnable()?.run(tvec!(input.clone()))?;
            let optimized = self.model()?.declutter()?;
            // dbg!(&optimized);
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
            pb.check()?
        }
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
        let op = Move(1, 0).checked().unwrap();
        assert_eq!(op.recip().recip(), op);
    }

    #[test]
    fn recip_move_20() {
        let op = Move(2, 0).checked().unwrap();
        assert_eq!(op.recip().recip(), op);
    }

    #[test]
    fn recip_move_02() {
        let op = Move(0, 2).checked().unwrap();
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
