use crate::internal::*;
use crate::model::{TypedModel, TypedNode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InOut {
    Out(usize),
    In(usize),
}

impl InOut {
    pub fn as_outlet<F: Clone + Fact + Hash, O: Clone + Hash>(&self, node: &BaseNode<F, O>) -> OutletId {
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

impl AxisOp {
    pub fn transform_axis(&self, axis: usize) -> Option<usize> {
        match self {
            AxisOp::Add(ix) => Some(axis + (axis >= *ix) as usize),
            AxisOp::Rm(ix) => {
                if axis == *ix {
                    None
                } else {
                    Some(axis - (axis > *ix) as usize)
                }
            }
            AxisOp::Permute(perm) => perm.get(axis).cloned(),
        }
    }

    pub fn transform_change(&self, change: &AxisOp) -> Option<AxisOp> {
        match change {
            AxisOp::Add(other) => self.transform_axis(*other).map(|o| AxisOp::Add(o)),
            AxisOp::Rm(other) => self.transform_axis(*other).map(|o| AxisOp::Rm(o)),
            AxisOp::Permute(axes) => {
                let mut axes: TVec<usize> =
                    axes.iter().flat_map(|a| self.transform_axis(*a)).collect();
                match self {
                    AxisOp::Add(add) => {
                        axes.insert(*add, *add);
                    }
                    _ => (),
                }
                Some(AxisOp::Permute(axes))
            }
        }
    }

    pub fn transform_op(&self, op: &AxisOp) -> Option<AxisOp> {
        Some(match op {
            AxisOp::Add(other) => match self {
                AxisOp::Rm(me) => AxisOp::Add(other - (me < other) as usize),
                AxisOp::Add(me) => AxisOp::Add(other + (me < other) as usize),
                AxisOp::Permute(_) => AxisOp::Add(*other),
            },
            AxisOp::Rm(other) => match self {
                AxisOp::Rm(me) => AxisOp::Rm(other - (me < other) as usize),
                AxisOp::Add(me) => AxisOp::Rm(other + (me < other) as usize),
                _ => self.transform_change(op).unwrap(),
            },
            _ => self.transform_change(op).unwrap(),
        })
    }

    pub fn change_shape_array<D: DimLike>(&self, shape: &mut TVec<D>) {
        match self {
            AxisOp::Add(ix) => shape.insert(*ix, D::one()),
            AxisOp::Rm(ix) => {
                shape.remove(*ix);
            }
            AxisOp::Permute(perm) => {
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
            AxisOp::Add(ix) => shape.insert_axis(*ix),
            AxisOp::Rm(ix) => {
                debug_assert_eq!(shape.dim(*ix), 1.to_dim());
                shape.remove_axis(*ix)
            }
            AxisOp::Permute(perm) => {
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
            AxisOp::Add(ix) => tensor.insert_axis(*ix),
            AxisOp::Rm(ix) => tensor.remove_axis(*ix),
            AxisOp::Permute(axes) => {
                let mut tmp = dispatch_datum!(permute(tensor.datum_type())(axes, tensor.clone()))?;
                std::mem::swap(tensor, &mut tmp);
                Ok(())
            }
        }
    }

    pub fn recip(&self) -> AxisOp {
        match self {
            AxisOp::Add(ix) => AxisOp::Rm(*ix),
            AxisOp::Rm(ix) => AxisOp::Add(*ix),
            AxisOp::Permute(axes) => {
                let perm = (0..axes.len())
                    .map(|axis| axes.iter().position(|i| axis == *i).unwrap())
                    .collect();
                AxisOp::Permute(perm)
            }
        }
    }

    pub fn is_noop(&self) -> bool {
        if let AxisOp::Permute(axes) = self {
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
            AxisOp::Add(_) => "AddAxis".into(),
            AxisOp::Rm(_) => "RmAxis".into(),
            AxisOp::Permute(_) => "Permute".into(),
        }
    }

    fn info(&self) -> TractResult<Vec<String>> {
        match self {
            AxisOp::Add(axis) | AxisOp::Rm(axis) => Ok(vec![format!("Axis: {}", axis)]),
            AxisOp::Permute(axes) => Ok(vec![format!("Axes: {:?}", axes)]),
        }
    }

    canonic!();
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
        _model: &TypedModel,
        _node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        debug_assert!(!change.is_noop());
        if (io == InOut::In(0) && change == self)
            || (io == InOut::Out(0) && change == &self.recip())
        {
            return Ok(Some(AxisChangeConsequence {
                substitute_op: Some(Box::new(crate::ops::identity::Identity)),
                wire_changes: tvec!(),
            }));
        }
        let incoming_change = match io {
            InOut::In(_) => change.clone(),
            InOut::Out(_) => {
                if let Some(change) = self.recip().transform_change(change) {
                    change
                } else {
                    return Ok(None);
                }
            }
        };
        let outgoing_change = if let Some(oc) = self.transform_change(&incoming_change) {
            oc
        } else {
            return Ok(None);
        };
        let new_me = incoming_change.transform_op(&self).unwrap();
        let substitute_op = if &new_me != self { Some(Box::new(new_me) as _) } else { None };
        let mut wire_changes = tvec!();
        if !incoming_change.is_noop() {
            wire_changes.push((InOut::In(0), incoming_change))
        }
        if !outgoing_change.is_noop() {
            wire_changes.push((InOut::Out(0), outgoing_change))
        }
        Ok(Some(AxisChangeConsequence { substitute_op, wire_changes }))
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
                debug!("Change {:?} blocked by locked interface {:?}", change, outlet);
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
                    debug!("Propagation of {:?} blocked by {}", change, node);
                    return Ok(None);
                }
                let AxisChangeConsequence { substitute_op, wire_changes } = more.unwrap();
                if let Some(op) = substitute_op {
                    trace!(
                        "Change {:?} enters {} from {:?} -> replace {:?} by {:?}",
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
    for node_id in model.eval_order()? {
        if let Some(new_op) = changed_ops.remove(&node_id) {
            model.node_mut(node_id).op = new_op;
        }
        let output_facts = model.node(node_id).op.output_facts(&model.node_input_facts(node_id)?)?;
        for (ix, f) in output_facts.into_iter().enumerate() {
            model.set_outlet_fact(OutletId::new(node_id, ix), f)?;
        }
    }
    debug!("Applied change {:?}", change);
    Ok(Some(changed_wires))
}

#[cfg(test)]
mod test {
    use super::*;
    use AxisOp::*;

    // ADD-ADD

    //           b,c   ------|Add(0)|----->        n,b,c
    //   Add(0)                                            Add(1)
    //         a,b,c   ------|Add(0)|----->        a,n,b,c
    #[test]
    pub fn transform_op_add_0_add_0() {
        let change = Add(0);
        let op = Add(0);
        assert_eq!(op.transform_change(&change), Some(Add(1)));
        assert_eq!(change.transform_op(&op), Some(Add(0)));
    }

    //           b,c   ------|Add(1)|----->        b,n,c
    //   Add(0)                                                 Add(0)
    //         a,b,c   ------|Add(2)|----->        a,b,n,c
    #[test]
    pub fn transform_op_add_0_add_1() {
        let change = Add(0);
        let op = Add(1);
        assert_eq!(op.transform_change(&change).unwrap(), Add(0));
        assert_eq!(change.transform_op(&op).unwrap(), Add(2));
    }

    //           a,c   ------|Add(0)|----->        n,a,c
    //   Add(1)                                                 Add(2)
    //         a,b,c   ------|Add(0)|----->        n,a,b,c
    #[test]
    pub fn transform_op_add_1_add_0() {
        let change = Add(1);
        let op = Add(0);
        assert_eq!(op.transform_change(&change).unwrap(), Add(2));
        assert_eq!(change.transform_op(&op).unwrap(), Add(0));
    }

    // RM-RM

    //         a,b,c   ------|Rm(0)|----->        b,c
    //   Rm(0)
    //           b,c
    #[test]
    pub fn transform_op_rm_0_rm_0() {
        let change = Rm(0);
        let op = Rm(0);
        assert_eq!(op.transform_change(&change), None);
    }

    //         a,b,c   ------|Rm(1)|----->         a,c
    //   Rm(0)                                             Rm(0)
    //           b,c   ------|Rm(0)|----->         c
    #[test]
    pub fn transform_op_rm_0_rm_1() {
        let change = Rm(0);
        let op = Rm(1);
        assert_eq!(op.transform_change(&change).unwrap(), Rm(0));
        assert_eq!(change.transform_op(&op).unwrap(), Rm(0));
    }

    //         a,b,c   ------|Rm(0)|----->         b,c
    //   Rm(1)                                             Rm(0)
    //           a,c   ------|Rm(0)|----->         c
    #[test]
    pub fn transform_op_rm_1_rm_0() {
        let change = Rm(1);
        let op = Rm(0);
        assert_eq!(op.transform_change(&change).unwrap(), Rm(0));
        assert_eq!(change.transform_op(&op).unwrap(), Rm(0));
    }

    // ADD - RM

    //
    //          b,c     ------|Rm(1)|------>        b
    //   Add(0)                                                 Add(0)
    //          a,b,c   ------|Rm(2)|----->         a,b
    #[test]
    pub fn transform_op_add_0_rm_1() {
        let change = Add(0);
        let op = Rm(1);
        assert_eq!(op.transform_change(&change).unwrap(), Add(0));
        assert_eq!(change.transform_op(&op).unwrap(), Rm(2));
    }

    //
    //          a,c     ------|Rm(0)|------>        c
    //   Add(1)                                                 Add(0)
    //          a,b,c   ------|Rm(0)|----->         b,c
    #[test]
    pub fn transform_op_add_1_rm_0() {
        let change = Add(1);
        let op = Rm(0);
        assert_eq!(op.transform_change(&change).unwrap(), Add(0));
        assert_eq!(change.transform_op(&op).unwrap(), Rm(0));
    }

    // RM - ADD

    //         a,b,c   ------|Add(0)|----->        X,a,b,c
    //   Rm(1)                                                 Rm(2)
    //           a,c   ------|Add(0)|----->        X,a,c
    #[test]
    pub fn transform_op_rm_1_add_0() {
        let change = Rm(1);
        let op = Add(0);
        assert_eq!(op.transform_change(&change).unwrap(), Rm(2));
        assert_eq!(change.transform_op(&op).unwrap(), Add(0));
    }

    //         a,b,c   ------|Add(1)|----->        a,X,b,c
    //   Rm(0)                                                 Rm(0)
    //           b,c   ------|Add(0)|----->        X,b,c
    #[test]
    pub fn transform_op_rm_0_add_1() {
        let change = Rm(0);
        let op = Add(1);
        assert_eq!(op.transform_change(&change).unwrap(), Rm(0));
        assert_eq!(change.transform_op(&op).unwrap(), Add(0));
    }

    // PERMUTE ADD

    //         a     ------|Add(1)|----->        a,b
    //   Perm(0)                                          Perm(0,1)
    //         a     ------|Add(1)|----->        a,b
    #[test]
    pub fn transform_permute_0_add_1() {
        let change = Permute(tvec!(0));
        let op = Add(1);
        assert_eq!(op.transform_change(&change).unwrap(), Permute(tvec!(0, 1)));
        assert_eq!(change.transform_op(&op).unwrap(), Add(1));
    }

    //         a,b     ------|Add(1)|----->        a,X,b
    //   Perm(1,0)                                          Perm(2,1,0)
    //         b,a     ------|Add(1)|----->        b,X,a
    #[test]
    pub fn transform_permute_10_add_1() {
        let change = Permute(tvec!(1, 0));
        let op = Add(1);
        assert_eq!(op.transform_change(&change).unwrap(), Permute(tvec!(2, 1, 0)));
        assert_eq!(change.transform_op(&op).unwrap(), Add(1));
    }

    // PERMUTE RM

    //         a,b,c     ------|Rm(1)|----->        a,c
    //   Perm(1,0,2)                                        Perm(0,1)
    //         b,a,c     ------|Rm(0)|----->        a,c
    #[test]
    pub fn transform_permute_102_rm_1() {
        let change = Permute(tvec!(1, 0, 2));
        let op = Rm(1);
        assert_eq!(op.transform_change(&change).unwrap(), Permute(tvec!(0, 1)));
        assert_eq!(change.transform_op(&op).unwrap(), Rm(0));
    }
}
