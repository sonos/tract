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

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum AxisOp {
    Add(usize),
    Rm(usize),
    Permute(TVec<usize>),
}
use AxisOp::*;

impl AxisOp {
    pub fn transform_axis(&self, axis: usize) -> Option<usize> {
        match self {
            Add(ix) => Some(axis + (axis >= *ix) as usize),
            Rm(ix) => {
                if axis == *ix {
                    None
                } else {
                    Some(axis - (axis > *ix) as usize)
                }
            }
            Permute(perm) => perm.get(axis).cloned(),
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
        match (self, change) {
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
            (Add(_), Permute(_)) => None,
            (Permute(_), Add(_)) => None,
            (Rm(op), Permute(c)) => Some((
                Some(Rm(c.iter().position(|it| it == op).unwrap())),
                Some(Permute(
                    c.iter()
                        .filter_map(|d| if d == op { None } else { Some(d - (d > op) as usize) })
                        .collect(),
                )),
            )),
            (Permute(op), Rm(c)) => Some((
                Some(Permute(
                    op.iter()
                        .filter_map(|d| if d == c { None } else { Some(d - (d > c) as usize) })
                        .collect(),
                )),
                Some(Rm(op.iter().position(|d| d == c).unwrap())),
            )),
            (Permute(op), Permute(c)) => {
                let c_r = Self::recip_perm(c);
                let new_op = Permute(op.iter().map(|d| c_r[*d]).collect());
                Some((Some(new_op), None))
            }
        }
    }

    pub fn change_shape_array<D: DimLike>(&self, shape: &mut TVec<D>) {
        match self {
            Add(ix) => shape.insert(*ix, D::one()),
            Rm(ix) => {
                shape.remove(*ix);
            }
            Permute(perm) => {
                let mut new_shape: TVec<D> = tvec!(D::default(); shape.len());
                for (ix, &from) in perm.iter().enumerate() {
                    new_shape[ix] = shape[from].clone();
                }
                shape.as_mut().clone_from_slice(&*new_shape);
            }
        }
    }

    pub fn change_shape(&self, shape: &mut ShapeFact) -> TractResult<()> {
        match self {
            Add(ix) => shape.insert_axis(*ix),
            Rm(ix) => {
                debug_assert_eq!(shape.dim(*ix), 1.to_dim(), "Removing a non-trivial axis.");
                shape.remove_axis(*ix)
            }
            Permute(perm) => {
                assert_eq!(perm.len(), shape.rank());
                let orig = shape.clone();
                for (ix, &from) in perm.iter().enumerate() {
                    shape.set_dim(ix, orig.dim(from).to_integer().unwrap_or(1).to_dim())?;
                }
                if let Some(info) = orig.stream_info {
                    shape.set_dim(perm.iter().position(|&i| i == info.axis).unwrap(), info.len)?;
                }
                Ok(())
            }
        }
    }

    pub fn change_tensor(&self, tensor: &mut Tensor) -> TractResult<()> {
        fn permute<T: Datum>(axes: &[usize], input: Tensor) -> TractResult<Tensor> {
            Ok(input.into_array::<T>()?.permuted_axes(axes).into_tensor())
        }
        match self {
            Add(ix) => tensor.insert_axis(*ix),
            Rm(ix) => tensor.remove_axis(*ix),
            Permute(axes) => {
                let mut tmp = dispatch_datum!(permute(tensor.datum_type())(axes, tensor.clone()))?;
                std::mem::swap(tensor, &mut tmp);
                Ok(())
            }
        }
    }

    fn recip_perm(axes: &[usize]) -> TVec<usize> {
        (0..axes.len()).map(|axis| axes.iter().position(|i| axis == *i).unwrap()).collect()
    }

    pub fn recip(&self) -> AxisOp {
        match self {
            Add(ix) => Rm(*ix),
            Rm(ix) => Add(*ix),
            Permute(axes) => Permute(Self::recip_perm(axes)),
        }
    }

    pub fn is_noop(&self) -> bool {
        if let Permute(axes) = self {
            axes.iter().enumerate().all(|(ix, &ax)| ix == ax)
        } else {
            false
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
            Permute(_) => "Permute".into(),
        }
    }

    fn info(&self) -> TractResult<Vec<String>> {
        match self {
            Add(axis) | Rm(axis) => Ok(vec![format!("Axis: {}", axis)]),
            Permute(axes) => Ok(vec![format!("Axes: {:?}", axes)]),
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
            trace!("  reciprocating {:?} as {:?}", self, self.recip());
            let more =
                if let Some(more) = self.recip().change_axes(model, node, InOut::In(0), &change)? {
                    more
                } else {
                    return Ok(None);
                };
            trace!("  returned {:?}", more);
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
            trace!("  Change {:?} from {:?} cancelling {:?}", &change, io, &self);
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
        _source: &NormalizedModel,
        node: &NormalizedNode,
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
    debug!("Trying to apply change {:?}", change);
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
                trace!("  node: {:?}", (node_id, io));
                if Some(node_id) == emitter {
                    continue;
                }
                trace!("  yep");
                let node = model.node(node_id);
                let more = node
                    .op
                    .change_axes(model, node, io, &c.op)
                    .chain_err(|| format!("Propagating {:?} to node {}", change, node))?;
                trace!("    more: {:?}", more);
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
    trace!("Applying {:?}", change);
    trace!("  changed_ops: {:?}", changed_ops);
    for node_id in model.eval_order()? {
        if let Some(new_op) = changed_ops.remove(&node_id) {
            trace!("{} -> {:?}", node_id, new_op);
            model.node_mut(node_id).op = new_op;
        }
        let output_facts =
            model.node(node_id).op.output_facts(&model.node_input_facts(node_id)?)?;
        for (ix, f) in output_facts.into_iter().enumerate() {
            model.set_outlet_fact(OutletId::new(node_id, ix), f)?;
        }
    }
    debug!("Applied change {:?}", change);
    debug!("{:#?}", &model);
    Ok(Some(changed_wires))
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! p {
        ($($d:expr),*) => { Permute(tvec!($($d),*)) }
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

    // PERMUTE ADD

    //                           Op
    //         a,b,c     ------|Rm(1)|----->        a,c
    //   Perm(1,0,2)                                        Perm(0,1)
    //         b,a,c     ------|Rm(0)|----->        a,c
    #[test]
    pub fn transform_permute_102_rm_1() {
        let change = Permute(tvec!(1, 0, 2));
        let op = Rm(1);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(0)), Some(p!(0, 1)))));
    }

    //                           Op
    //         a,b,c     ------|Rm(0)|----->        b,c
    //   Perm(1,2,0)                                        Perm(0,1)
    //         b,c,a     ------|Rm(2)|----->        b,c
    #[test]
    pub fn transform_permute_120_rm_0() {
        let change = Permute(tvec!(1, 2, 0));
        let op = Rm(0);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(2)), Some(p!(0, 1)))));
    }

    //                           Op
    //         a,b,c     ------|Rm(1)|----->        b,c
    //      Perm(1,2,0)                                 Perm(1, 0)
    //         b,c,a     ------|Rm(2)|------->      c,a
    #[test]
    pub fn transform_permute_120_rm_1() {
        let change = Permute(tvec!(1, 2, 0));
        let op = Rm(1);
        assert_eq!(op.merge_incoming_change(&change), Some((Some(Rm(0)), Some(p!(1, 0)))));
    }

    // RM PERMUTE

    //                           Op
    //         a,b,c   ------|Perm(2,0,1)|----->       c,b,a
    //      Rm(0)                                           Rm(1)
    //         b,c     ------|Perm(1,0)|------->       c,a
    #[test]
    pub fn transform_rm_0_permute_201() {
        let change = Rm(0);
        let op = Permute(tvec!(2, 0, 1));
        assert_eq!(op.merge_incoming_change(&change), Some((Some(p!(1, 0)), Some(Rm(1)))));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug)]
    struct Problem {
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
                        Just((0..shape.len()).collect::<Vec<usize>>())
                            .prop_shuffle()
                            .prop_filter("trivial permutation", |p| {
                                !p.windows(2).all(|w| w[0] < w[1])
                            })
                            .prop_map(|p| Permute(p.into()))
                            .boxed(),
                    )
                    .boxed();
            }
            let rms =
                (0..shape.len()).filter(|&ax| shape[ax] == 1).map(|ax| Rm(ax)).collect::<Vec<_>>();
            if rms.len() > 0 {
                ops = ops
                    .prop_union((0..rms.len()).prop_map(move |rm| rms[rm].clone()).boxed())
                    .boxed()
            }
            ops
        }
    }

    impl Arbitrary for Problem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Problem>;
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
                .prop_map(|(input, ops)| Problem { input: input.into(), ops })
                .boxed()
        }
    }

    impl Problem {
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
            let raw = model.into_runnable()?.run(tvec!(input.clone()))?;
            let optimized = self.model()?.declutter()?;
            let opt = optimized.into_runnable()?.run(tvec!(input))?;
            opt[0].close_enough(&raw[0], false)
        }
    }

    proptest! {
        #[test]
        fn axis_ops(pb in any::<Problem>()) {
            pb.check()?
        }
    }

    #[test]
    fn add_0_perm_10() {
        let pb = Problem { input: tvec![2], ops: tvec![Add(0), Permute(tvec![1, 0])] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_1_perm_120() {
        let pb = Problem { input: tvec![2], ops: tvec![Add(0), Add(1), Permute(tvec![1, 2, 0])] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0() {
        let pb = Problem { input: tvec![1], ops: tvec![Add(0), Add(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0_perm_120() {
        let pb = Problem { input: tvec![2], ops: tvec![Add(0), Add(0), Permute(tvec!(1, 2, 0))] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0_perm_120_rm_0() {
        let pb =
            Problem { input: tvec![1], ops: tvec![Add(0), Add(0), Permute(tvec!(1, 2, 0)), Rm(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_0_perm_201_perm_201() {
        let pb = Problem {
            input: tvec![2],
            ops: tvec![Add(0), Add(0), Permute(tvec!(2, 0, 1)), Permute(tvec!(2, 0, 1))],
        };
        pb.check().unwrap();
    }

    #[test]
    fn perm_1_0_add_0() {
        let pb = Problem { input: tvec![1, 1], ops: tvec![Permute(tvec!(1, 0)), Add(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_perm_120_perm_120() {
        let pb = Problem {
            input: tvec![1, 1],
            ops: tvec![Add(0), Permute(tvec!(1, 2, 0)), Permute(tvec!(1, 2, 0)),],
        };
        pb.check().unwrap();
    }

    #[test]
    fn add_0_add_2_perm_201_perm_021_rm_2() {
        let pb = Problem {
            input: tvec![3],
            ops: tvec![Add(0), Add(2), Permute(tvec!(2, 0, 1)), Permute(tvec!(0, 2, 1)), Rm(2)],
        };
        pb.check().unwrap();
    }

    #[test]
    fn perm_120_perm_120() {
        let pb = Problem {
            input: tvec![2, 1, 1],
            ops: tvec![Permute(tvec!(1, 2, 0)), Permute(tvec!(1, 2, 0))],
        };
        pb.check().unwrap();
    }

    #[test]
    fn rm_1_perm_10_add_0() {
        let pb = Problem { input: tvec![1, 1, 2], ops: tvec![Rm(1), Permute(tvec![1, 0]), Add(0)] };
        pb.check().unwrap();
    }

    #[test]
    fn add_2_perm_120_perm_120() {
        let pb = Problem {
            input: tvec![3, 2],
            ops: tvec![Add(2), Permute(tvec!(1, 2, 0)), Permute(tvec!(1, 2, 0))],
        };
        pb.check().unwrap();
    }
}
