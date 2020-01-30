use crate::internal::*;
use crate::model::{TypedModel, TypedNode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InOut {
    Out(usize),
    In(usize),
}

impl InOut {
    pub fn as_outlet<TI: Clone + Fact, O: Clone>(&self, node: &BaseNode<TI, O>) -> OutletId {
        match self {
            InOut::In(ix) => node.inputs[*ix],
            InOut::Out(ix) => OutletId::new(node.id, *ix),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AxisOp {
    Add(usize),
    Rm(usize),
    /*
    Permute(TVec<usize>),
    */
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
        }
    }

    pub fn transform_op(&self, op: &AxisOp) -> TractResult<AxisOp> {
        match op {
            AxisOp::Add(other) => {
                Ok(AxisOp::Add(self.transform_axis(*other).ok_or("Invalid axis tranformation")?))
            }
            AxisOp::Rm(other) => {
                Ok(AxisOp::Rm(self.transform_axis(*other).ok_or("Invalid axis transformation")?))
            }
        }
    }

    pub fn change_shape_array(&self, shape: &mut TVec<usize>) {
        match self {
            AxisOp::Add(ix) => shape.insert(*ix, 1),
            AxisOp::Rm(ix) => {
                shape.remove(*ix);
            }
        }
    }

    pub fn change_shape(&self, shape: &mut ShapeInfo) -> TractResult<()> {
        match self {
            AxisOp::Add(ix) => shape.insert_axis(*ix),
            AxisOp::Rm(ix) => shape.remove_axis(*ix),
        }
    }

    pub fn change_tensor(&self, tensor: &mut Tensor) -> TractResult<()> {
        match self {
            AxisOp::Add(ix) => tensor.insert_axis(*ix),
            AxisOp::Rm(ix) => tensor.remove_axis(*ix),
        }
    }

    pub fn recip(&self) -> AxisOp {
        match self {
            AxisOp::Add(ix) => AxisOp::Rm(*ix),
            AxisOp::Rm(ix) => AxisOp::Add(*ix),
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
        }
    }

    fn info(&self) -> TractResult<Vec<String>> {
        match self {
            AxisOp::Add(axis) | AxisOp::Rm(axis) => Ok(vec![format!("Axis: {}", axis)]),
        }
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for AxisOp {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut input = args_1!(inputs).into_tensor();
        self.change_tensor(&mut input)?;
        Ok(tvec!(input.into_arc_tensor()))
    }
}

impl TypedOp for AxisOp {
    typed_op_as_op!();

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
        Ok(tvec!((InOut::Out(0), self.recip())))
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
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
            InOut::Out(_) => self.recip().transform_op(change)?,
        };
        let outgoing_change = self.transform_op(&incoming_change)?;
        let new_me = incoming_change.transform_op(&self)?;
        Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(new_me)),
            wire_changes: tvec!((InOut::In(0), incoming_change), (InOut::Out(0), outgoing_change),),
        }))
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

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}

pub fn change_axes(
    model: &mut TypedModel,
    change: &AxisChange,
    lock_interfaces: bool,
) -> TractResult<Option<HashMap<OutletId, AxisOp>>> {
    let mut todo_changes = vec![change.clone()];
    let mut changed_wires = HashMap::new();
    changed_wires.insert(change.outlet, change.op.clone());
    let mut changed_ops: HashMap<usize, Box<dyn TypedOp>> = HashMap::new();
    while let Some(change) = todo_changes.pop() {
        if lock_interfaces
            && (model.output_outlets()?.contains(&change.outlet)
                || model.input_outlets()?.contains(&change.outlet))
        {
            return Ok(None);
        }
        let mut nodes = vec![(change.outlet.node, InOut::Out(change.outlet.slot))];
        for inlet in model.outlet_successors(change.outlet) {
            nodes.push((inlet.node, InOut::In(inlet.slot)));
        }
        for (node_id, io) in nodes {
            let node = model.node(node_id);
            let more = node
                .op
                .change_axes(model, node, io, &change.op)
                .chain_err(|| format!("Propagating {:?} to node {}", change, node))?;
            if more.is_none() {
                return Ok(None);
            }
            let AxisChangeConsequence { substitute_op, wire_changes } = more.unwrap();
            if let Some(op) = substitute_op {
                changed_ops.insert(node.id, op);
            }
            for (wire, op) in wire_changes.into_iter() {
                let outlet = wire.as_outlet(node);
                if !changed_wires.contains_key(&outlet) {
                    changed_wires.insert(outlet, op.clone());
                    todo_changes.push(AxisChange { outlet, op });
                }
            }
        }
    }
    for (node_id, op) in changed_ops.into_iter() {
        model.node_mut(node_id).op = op;
    }
    for (outlet, axis_op) in &changed_wires {
        let node = model.node_mut(outlet.node);
        axis_op.change_shape(&mut node.outputs[outlet.slot].fact.shape)?;
    }
    Ok(Some(changed_wires))
}
