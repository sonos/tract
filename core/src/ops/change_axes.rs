use std::borrow::Borrow;
use std::fmt::Debug;

use crate::internal::*;
use crate::model::{TypedModel, TypedNode};
use crate::ops::identity::Identity;
use num_traits::One;
use tract_itertools::Itertools;
use tract_linalg::block_quant::{BlockQuantFact, BlockQuantValue};
use tract_ndarray::{ArrayViewD, ArrayViewMutD};
use AxisOp::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InOut {
    Out(usize),
    In(usize),
}

impl InOut {
    pub fn as_outlet<F: Clone + Fact, O: Clone>(&self, node: &Node<F, O>) -> OutletId {
        match self {
            InOut::In(ix) => node.inputs[*ix],
            InOut::Out(ix) => OutletId::new(node.id, *ix),
        }
    }

    pub fn is_input(&self) -> bool {
        matches!(self, InOut::In(_))
    }

    pub fn is_output(&self) -> bool {
        matches!(self, InOut::Out(_))
    }

    pub fn slot(&self) -> usize {
        match self {
            InOut::Out(o) => *o,
            InOut::In(i) => *i,
        }
    }
}

#[derive(Clone, Hash, Eq)]
#[allow(clippy::large_enum_variant)] // FIXME ?
#[allow(clippy::derived_hash_with_manual_eq)] // FIXME. this one may be pretty bad. how about a.canonical() == b.canonical() ? need proper canonicalizeation of Reshape
pub enum AxisOp {
    Add(usize),
    Rm(usize),
    Move(usize, usize),
    Reshape(usize, TVec<TDim>, TVec<TDim>),
}

impl Debug for AxisOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AxisOp::Add(a) => write!(f, "Add({a})"),
            AxisOp::Rm(a) => write!(f, "Rm({a})"),
            AxisOp::Move(from, to) => write!(f, "Move({from},{to})"),
            AxisOp::Reshape(at, from, to) => {
                write!(f, "Reshape({at}, [{}], [{}])", from.iter().join(","), to.iter().join(","))
            }
        }
    }
}

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
    pub fn canonical(&self) -> Cow<AxisOp> {
        match self {
            Move(from, to) if *from == to + 1 => Cow::Owned(Move(*to, *from)),
            Reshape(at, from, to) if from.len() == 1 && to.len() == 2 && from[0] == to[0] => {
                Cow::Owned(Add(*at + 1))
            }
            Reshape(at, from, to) if from.len() == 1 && to.len() == 2 && from[0] == to[1] => {
                Cow::Owned(Add(*at))
            }
            Reshape(at, from, to) if from.len() == 2 && to.len() == 1 && from[0] == to[0] => {
                Cow::Owned(Rm(*at + 1))
            }
            Reshape(at, from, to) if from.len() == 2 && to.len() == 1 && from[1] == to[0] => {
                Cow::Owned(Rm(*at))
            }
            other => Cow::Borrowed(other),
        }
    }

    pub fn simplify(&self) -> TVec<AxisOp> {
        match self.canonical().borrow() {
            Reshape(_, from, to) if from == to => tvec!(),
            Reshape(at, from, to) if to.len() == 0 => tvec!(Rm(*at); from.len()),
            Reshape(at, from, to) if from.len() == 0 => tvec!(Add(*at); to.len()),
            Reshape(at, from, to) if from[0] == to[0] => {
                Reshape(at + 1, from[1..].into(), to[1..].into()).simplify()
            }
            Reshape(at, from, to) if from[from.len() - 1] == to[to.len() - 1] => {
                Reshape(*at, from[..from.len() - 1].into(), to[..to.len() - 1].into()).simplify()
            }
            Reshape(at, from, to) if from[0] == 1.to_dim() => std::iter::once(Rm(*at))
                .chain(Reshape(*at, from[1..].into(), to.clone()).simplify())
                .collect(),
            Reshape(at, from, to) if to[0] == 1.to_dim() => {
                Reshape(*at, from.clone(), to[1..].into())
                    .simplify()
                    .into_iter()
                    .chain(std::iter::once(Add(*at)))
                    .collect()
            }
            Reshape(at, from, to) if from[from.len() - 1] == 1.to_dim() => {
                std::iter::once(Rm(at + from.len() - 1))
                    .chain(Reshape(*at, from[..from.len() - 1].into(), to.clone()).simplify())
                    .collect()
            }
            Reshape(at, from, to) if to[to.len() - 1] == 1.to_dim() => {
                std::iter::once(Add(at + from.len()))
                    .chain(Reshape(*at, from.clone(), to[..to.len() - 1].into()).simplify())
                    .collect()
            }
            other => tvec!(other.clone()),
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
                if x == from {
                    Some((Some(Rm(*to)), None))
                } else if x < from.min(to) {
                    Some((Some(self.clone()), Some(Move(from - 1, to - 1))))
                } else if x > from.max(to) {
                    Some((Some(self.clone()), Some(change.clone())))
                } else if from + 1 == *to && x == to {
                    Some((Some(Rm(*from)), None))
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

    pub fn change_shape_array<D: DimLike>(
        &self,
        shape: &mut TVec<D>,
        broadcasting: bool,
    ) -> TractResult<()> {
        match self.canonical().as_ref() {
            Add(ix) => {
                ensure!(*ix <= shape.len());
                shape.insert(*ix, D::one());
            }
            Rm(ix) => {
                ensure!(*ix < shape.len());
                shape.remove(*ix);
            }
            Move(from, to) => {
                ensure!(*from < shape.len());
                ensure!(*to < shape.len());
                let axis = shape.remove(*from);
                shape.insert(*to, axis);
            }
            Reshape(at, from, to) => {
                let from_volume = from.iter().product::<TDim>();
                let to_volume = to.iter().product::<TDim>();
                ensure!(from_volume == to_volume, "{from_volume} should be equal to {to_volume}");
                ensure!(*at + from.len() <= shape.len());
                if shape.len() >= from.len() + *at
                    && tract_itertools::izip!(shape.iter().skip(*at), from)
                        .all(|(shape, spec)| shape.to_dim() == *spec)
                {
                    for _ in from {
                        shape.remove(*at);
                    }
                    for d in to.iter().rev() {
                        shape.insert(*at, d.try_into()?);
                    }
                } else if broadcasting
                    && shape.iter().skip(*at).take(from.len()).all(|d| d.to_dim() == 1.to_dim())
                {
                    for _ in from {
                        shape.remove(*at);
                    }
                    for _ in to.iter().rev() {
                        shape.insert(*at, 1.into());
                    }
                } else {
                    bail!("Incompatible reshape for shape {:?} and {:?}", shape, self);
                }
            }
        }
        Ok(())
    }

    pub fn change_shape(&self, shape: &mut ShapeFact, broadcasting: bool) -> TractResult<()> {
        match self.canonical().as_ref() {
            Add(ix) => shape.insert_axis(*ix),
            Rm(ix) => {
                if shape.rank() <= *ix {
                    bail!("Attempt to remove axis #{} on shape {:?}", ix, shape);
                }
                if shape[*ix] != 1.to_dim() {
                    bail!("Removing non-trivial axis #{} of dim: {:?}", ix, shape);
                }
                shape.remove_axis(*ix)
            }
            _ => {
                let mut array = shape.to_tvec();
                self.change_shape_array(&mut array, broadcasting)?;
                let mut new_shape = ShapeFact::from_dims(array);
                std::mem::swap(shape, &mut new_shape);
                Ok(())
            }
        }
    }

    pub fn change_tensor(&self, tensor: &mut Tensor, broadcasting: bool) -> TractResult<()> {
        if self.required_rank() > tensor.rank() && tensor.datum_type().is_opaque() {
            let inner_change = self.trim_left(tensor.rank())?;
            for opaque in tensor.as_slice_mut::<Opaque>()? {
                if let Some(bqv) = opaque.downcast_ref::<BlockQuantValue>() {
                    let mut new_shape: TVec<usize> = bqv.fact.shape().into();
                    inner_change.change_shape_array(&mut new_shape, false)?;
                    let new_bqv = BlockQuantValue {
                        value: Arc::clone(&bqv.value),
                        fact: BlockQuantFact::new(bqv.fact.format.clone(), new_shape),
                    };
                    *opaque = Opaque(Arc::new(new_bqv));
                } else {
                    bail!("Can't apply {self:?} to opaque tensor {tensor:?}");
                }
            }
            return Ok(());
        }
        ensure!(self.required_rank() <= tensor.rank());
        match self.canonical().as_ref() {
            Add(ix) => tensor.insert_axis(*ix),
            Rm(ix) => tensor.remove_axis(*ix),
            Move(from, to) => {
                let mut tmp = tensor.clone().move_axis(*from, *to)?;
                std::mem::swap(tensor, &mut tmp);
                Ok(())
            }
            Reshape(at, from, to) => {
                let mut shape: TVec<usize> = tensor.shape().into();
                self.change_shape_array(&mut shape, false)?;
                if tensor.set_shape(&shape).is_ok() {
                    Ok(())
                } else if broadcasting
                    && tensor.shape().iter().skip(*at).take(from.len()).all(|d| *d == 1)
                {
                    if from.len() > to.len() {
                        for _ in to.len()..from.len() {
                            tensor.remove_axis(*at)?;
                        }
                    }
                    if to.len() > from.len() {
                        for _ in from.len()..to.len() {
                            tensor.insert_axis(*at)?;
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

    pub fn change_view<D>(&self, view: &mut ArrayViewD<D>) -> TractResult<()> {
        use tract_ndarray::Axis;
        match *self {
            AxisOp::Rm(axis) => view.index_axis_inplace(Axis(axis), 0),
            AxisOp::Add(axis) => view.insert_axis_inplace(Axis(axis)),
            AxisOp::Move(from, to) if from < to => {
                for left in from..to {
                    view.swap_axes(left, left + 1);
                }
            }
            AxisOp::Move(from, to) => {
                for left in (to..from).rev() {
                    view.swap_axes(left, left + 1);
                }
            }
            AxisOp::Reshape(_, _, _) => bail!("Reshape can not change views in place"),
        }
        Ok(())
    }

    pub fn change_view_mut<D>(&self, view: &mut ArrayViewMutD<D>) -> TractResult<()> {
        use tract_ndarray::Axis;
        match *self {
            AxisOp::Rm(axis) => view.index_axis_inplace(Axis(axis), 0),
            AxisOp::Add(axis) => view.insert_axis_inplace(Axis(axis)),
            AxisOp::Move(from, to) if from < to => {
                for left in from..to {
                    view.swap_axes(left, left + 1);
                }
            }
            AxisOp::Move(from, to) => {
                for left in (to..from).rev() {
                    view.swap_axes(left, left + 1);
                }
            }
            AxisOp::Reshape(_, _, _) => bail!("Reshape can not change views in place"),
        }
        Ok(())
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
        !matches!(self, Move(_, _))
    }

    pub fn wire_split_axis(
        model: &mut TypedModel,
        name: impl ToString,
        outlet: OutletId,
        axis: usize,
        outer_dim: usize,
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(outlet)?;
        let dim: TDim = fact.shape[axis].clone();
        let inner_dim = dim.clone() / outer_dim;
        let op = Self::Reshape(axis, tvec!(dim.clone()), tvec!(outer_dim.to_dim(), inner_dim));
        model.wire_node(name.to_string(), op, &[outlet])
    }

    pub fn wire_collapse_axis(
        model: &mut TypedModel,
        name: impl ToString,
        outlet: OutletId,
        axis: usize,
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(outlet)?;
        let dim: TDim = fact.shape[axis].clone();
        let next_dim: TDim = fact.shape[axis + 1].clone();
        let op = Self::Reshape(axis, tvec!(dim.clone(), next_dim.clone()), tvec!(dim * next_dim));
        model.wire_node(name.to_string(), op, &[outlet])
    }

    #[inline]
    pub fn required_rank(&self) -> usize {
        match self {
            Rm(r) => r + 1,
            Add(a) => *a,
            Reshape(at, from, _to) => at + from.len(),
            Move(from, to) => *from.max(to),
        }
    }

    pub fn trim_left(&self, prefix: usize) -> TractResult<AxisOp> {
        Ok(match self {
            Rm(r) if *r >= prefix => Rm(r - prefix),
            Add(a) if *a >= prefix => Add(a - prefix),
            Reshape(at, from, to) if *at >= prefix => {
                Reshape(at - prefix, from.clone(), to.clone())
            }
            Move(from, to) if *from >= prefix && *to >= prefix => Move(from - prefix, to - prefix),
            _ => bail!("Can no trim left {self:?} by {prefix}"),
        })
    }
}

pub fn wire_rank_broadcast(
    prefix: impl AsRef<str>,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let facts =
        inputs.iter().map(|o| target.outlet_fact(*o).cloned()).collect::<TractResult<TVec<_>>>()?;
    let max_rank = facts.iter().map(|f| f.rank()).max().unwrap();
    let mut wires = tvec!();
    let prefix = prefix.as_ref();
    for i in 0..inputs.len() {
        let mut wire = inputs[i];
        for j in facts[i].rank()..max_rank {
            wire =
                target.wire_node(format!("{prefix}.fix-rank-{i}-{j}"), AxisOp::Add(0), &[wire])?[0];
        }
        wires.push(wire);
    }
    Ok(wires)
}

pub fn wire_with_rank_broadcast(
    prefix: impl AsRef<str>,
    target: &mut TypedModel,
    op: impl Into<Box<dyn TypedOp>>,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let prefix = prefix.as_ref();
    let wires = wire_rank_broadcast(prefix, target, inputs)?;
    target.wire_node(prefix, op.into(), &wires)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
            Add(axis) | Rm(axis) => Ok(vec![format!("Axis: {axis}")]),
            Move(from, to) => Ok(vec![format!("Axis {from} to {to}")]),
            Reshape(at, from, to) => Ok(vec![format!(
                "Axes starting at {}: {:?} to {:?}",
                at,
                from.iter().join(","),
                to.iter().join(",")
            )]),
        }
    }

    op_as_typed_op!();
}

impl EvalOp for AxisOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let mut input = args_1!(inputs).into_tensor();
        match self {
            AxisOp::Reshape(skip, from, to) => {
                let from = from.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                let to = to.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                AxisOp::Reshape(*skip, from, to).change_tensor(&mut input, false)?
            }
            _ => self.change_tensor(&mut input, false)?,
        }
        Ok(tvec!(input.into_tvalue()))
    }
}

impl TypedOp for AxisOp {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if self.required_rank() > inputs[0].rank() {
            if let Some(bqf) =
                inputs[0].opaque_fact().and_then(|of| of.downcast_ref::<BlockQuantFact>())
            {
                let mut new_inner_shape: TVec<usize> = bqf.shape().into();
                self.trim_left(inputs[0].rank())?
                    .change_shape_array(&mut new_inner_shape, false)?;
                let new_bqf = BlockQuantFact::new(bqf.format.clone(), new_inner_shape);
                let mut new_fact = Opaque::fact(inputs[0].shape.clone()).with_opaque_fact(new_bqf);
                if let Some(k) = &inputs[0].konst {
                    let mut new = k.clone().into_tensor(); // cloning bqv is cheap
                    self.change_tensor(&mut new, false)?;
                    new_fact.konst = Some(new.into());
                }
                return Ok(tvec!(new_fact));
            }
        }
        let mut shape = inputs[0].shape.clone();
        self.change_shape(&mut shape, false)?;
        let mut fact = inputs[0].datum_type.fact(shape);
        fact.opaque_fact.clone_from(&inputs[0].opaque_fact);
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes: Vec<Axis> = (0..inputs[0].rank())
            .zip('a'..)
            .map(|(axis_id, repr)| {
                let mut axis = Axis::new(repr, inputs.len(), outputs.len()).input(0, axis_id);
                if let Some(out) = self.transform_axis(axis_id) {
                    axis = axis.output(0, out);
                }
                axis
            })
            .collect();
        for (axis, letter) in (0..outputs[0].rank()).zip('A'..) {
            if self.recip().transform_axis(axis).is_none() {
                axes.push(Axis::new(letter, inputs.len(), outputs.len()).output(0, axis));
            }
        }
        AxesMapping::new(inputs.len(), outputs.len(), axes)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.is_noop() {
            if let Some(p) = TypedModelPatch::shunt_one_op(model, node)? {
                return Ok(Some(p));
            }
        }
        let simplified = self.simplify();
        if simplified.len() != 1 || &simplified[0] != self {
            let mut patch = TypedModelPatch::default();
            let mut wire = patch.tap_model(model, node.inputs[0])?;
            for (ix, op) in simplified.into_iter().enumerate() {
                wire = patch.wire_node(format!("{}.{}", node.name, ix), op, &[wire])?[0];
            }
            patch.shunt_outside(model, node.id.into(), wire)?;
            Ok(Some(patch))
        } else {
            Ok(None)
        }
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
        let op = if let InOut::Out(0) = io {
            let more = if let Some(more) =
                self.recip().change_axes(_model, _node, InOut::In(0), change)?
            {
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
                "  Change:{change:?} self:{self:?} -> change:{new_change:?} op:{new_op:?}"
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
                from.iter().map(|d| d.eval(values)).collect(),
                to.iter().map(|d| d.eval(values)).collect(),
            )
        } else {
            self.clone()
        };
        target.wire_node(&node.name, op, &[mapping[&node.inputs[0]]])
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        // is this test really useful ? or axis mapping preempt this ?
        if let Reshape(pos, _from, to) = self {
            if output_axis >= *pos && output_axis < pos + to.len() {
                return Ok(None);
            }
        }
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if node.outputs[0].fact.opaque_fact.is_some() {
            return Ok(None);
        }
        if let Some(shape) = node.outputs[0].fact.shape.as_concrete() {
            if !matches!(self, AxisOp::Move(_, _)) {
                let (inputs, outputs) = model.node_facts(node.id)?;
                let mapping = self.axes_mapping(&inputs, &outputs)?;
                let op = IntoShape {
                    mapping,
                    len: shape.iter().product(),
                    strides: Tensor::natural_strides(shape),
                    dims: shape.into(),
                };
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &node.inputs,
                    op,
                )?));
            }
        }
        Ok(None)
    }
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
            if let Some(rot) = is_rotation_cycle(cycle) {
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

pub fn compute_shape_with_tf_rules(input: &[TDim], shape_spec: &[TDim]) -> TractResult<TVec<TDim>> {
    let mut shape: TVec<TDim> = shape_spec.into();
    fn deal_with_zero<'a>(
        mut input_dims: std::iter::Peekable<impl Iterator<Item = &'a TDim>>,
        shape: &mut [TDim],
    ) -> TractResult<()> {
        let mut remaining_dim_input = 1.to_dim();
        for slot in shape.iter_mut() {
            if *slot == (-1).into() {
                break;
            }
            if *slot == 0.into() {
                if remaining_dim_input != TDim::one() {
                    bail!("Invalid remaining dim");
                }
                *slot = (*input_dims.peek().context("Invalid")?).clone();
            }
            loop {
                let quotient = remaining_dim_input.maybe_div(slot);
                if quotient.is_err() || quotient.as_ref().unwrap().1 != 1 {
                    remaining_dim_input *= input_dims.next().context("Invalid")?;
                } else {
                    break;
                }
            }
            remaining_dim_input = remaining_dim_input.maybe_div(slot)?.0;
        }
        Ok(())
    }

    deal_with_zero(input.iter().peekable(), &mut shape)?;
    shape.reverse();
    deal_with_zero(input.iter().rev().peekable(), &mut shape)?;
    shape.reverse();

    if let Some(pos) = shape.iter().position(|d| *d == (-1).into()) {
        let input_vol: TDim = input.iter().product();
        let shape_vol: TDim = shape.iter().filter(|d| **d != (-1).into()).product();
        let div = input_vol.maybe_div(&shape_vol)?;
        if div.1 != 1 {
            bail!("invalid")
        }
        shape[pos] = div.0;
    }
    Ok(shape)
}

pub fn to_axis_ops_with_tf_rules(
    input_orig: &[TDim],
    output_spec: &[TDim],
) -> TractResult<TVec<AxisOp>> {
    let final_output = compute_shape_with_tf_rules(input_orig, output_spec)?;
    let mut stack: TVec<AxisOp> = tvec!();
    'top: loop {
        let current_input =
            stack.iter().try_fold(TVec::from(input_orig), |mut shape, op| -> TractResult<_> {
                op.change_shape_array(&mut shape, false)?;
                Ok(shape)
            })?;
        if current_input == final_output {
            return Ok(stack);
        }
        if let Some(common) =
            current_input.iter().zip(final_output.iter()).position(|(a, b)| a != b)
        {
            if current_input[common].is_one() {
                stack.push(AxisOp::Rm(common));
            } else if final_output[common].is_one() {
                stack.push(AxisOp::Add(common));
            } else {
                // actual regrouping. search for a match. this is quadratic, but
                // rank is expected to be somewhat reasonable
                for i in common..current_input.len() {
                    let i_group = &current_input[common..i + 1];
                    let i_volume: TDim = i_group.iter().product();
                    for o in common..final_output.len() {
                        let o_group = &final_output[common..o + 1];
                        let o_volume: TDim = o_group.iter().product();
                        if i_volume == o_volume {
                            stack.push(AxisOp::Reshape(common, i_group.into(), o_group.into()));
                            continue 'top;
                        }
                    }
                }
                todo!()
            }
        } else if final_output.len() > current_input.len() {
            stack.push(AxisOp::Add(current_input.len()));
        } else {
            stack.push(AxisOp::Rm(current_input.len() - 1));
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntoShape {
    pub mapping: AxesMapping,
    pub len: usize,
    pub dims: TVec<usize>,
    pub strides: TVec<isize>,
}

impl Op for IntoShape {
    fn name(&self) -> Cow<str> {
        "IntoShape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{}", self.mapping)])
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for IntoShape {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut input = args_1!(inputs).into_tensor();
        ensure!(input.len() == self.len);
        unsafe { input.set_geometry_unchecked(&self.dims, &self.strides) };
        Ok(tvec!(input.into_tvalue()))
    }
}

impl TypedOp for IntoShape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].datum_type.fact(&self.dims);
        if let Some(of) = &inputs[0].opaque_fact {
            fact = fact.with_opaque_fact(of.clone());
        }
        Ok(tvec!(fact))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input = model.outlet_fact(node.inputs[0])?;
        if input.shape.as_concrete().is_some_and(|shape| shape == &*self.dims) {
            return TypedModelPatch::shunt_one_op(model, node);
        }
        if let Some(succ) = model.single_succ(node.id)? {
            if let Some(into_shape) = succ.op_as::<IntoShape>() {
                let op = Self {
                    mapping: self.mapping.compose(&into_shape.mapping)?,
                    ..into_shape.clone()
                };
                return Ok(Some(TypedModelPatch::fuse_with_next(model, node, op)?));
            }
        }
        Ok(None)
    }

    as_op!();
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
            let mut ops: BoxedStrategy<AxisOp> = (0usize..shape.len() + 1).prop_map(Add).boxed();
            if shape.len() > 1 {
                ops = ops
                    .prop_union(
                        (0..shape.len(), 0..shape.len() - 1)
                            .prop_map(|(a, b)| Move(a, b + (b >= a) as usize))
                            .boxed(),
                    )
                    .boxed()
            }
            let rms = (0..shape.len()).filter(|&ax| shape[ax] == 1).map(Rm).collect::<Vec<_>>();
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
                    AxisOp::arbitrary_with(shape.clone())
                        .prop_flat_map(move |op| {
                            let mut shape = shape.clone();
                            op.change_shape_array(&mut shape, false).unwrap();
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
            let mut wire = model.add_source("source", i64::fact(&self.input))?;
            for (ix, op) in self.ops.iter().enumerate() {
                wire = model.wire_node(format!("op_{ix}"), op.clone(), &[wire])?[0];
            }
            model.set_output_outlets(&[wire])?;
            Ok(model)
        }

        fn input(&self) -> TractResult<Tensor> {
            unsafe {
                let mut t = Tensor::uninitialized::<i64>(&self.input)?;
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
            let raw = model.into_runnable()?.run(tvec!(input.clone().into_tvalue()))?;
            let optimized = self.model()?.into_decluttered()?;
            let opt = optimized.into_runnable()?.run(tvec!(input.into_tvalue()))?;
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

    #[test]
    fn reshape_axes_tracking() {
        let pb = ComposeProblem {
            input: tvec![2, 2, 2],
            ops: tvec![Reshape(0, tvec!(2.to_dim(), 2.to_dim()), tvec!(4.to_dim()))],
        };
        pb.check().unwrap();
    }

    #[test]
    fn simplify_reshape() {
        macro_rules! d {
            ($($dim: expr),*) =>  { tvec!($($dim.to_dim()),*) }
        }
        assert_eq!(Reshape(3, d!(), d!()).simplify(), tvec!());
        assert_eq!(Reshape(3, d!(2, 3), d!(2, 3)).simplify(), tvec!());
        assert_eq!(Reshape(3, d!(1), d!()).simplify(), tvec!(Rm(3)));
        assert_eq!(Reshape(3, d!(), d!(1)).simplify(), tvec!(Add(3)));
        assert_eq!(
            Reshape(3, d!(2, 3, 4), d!(2, 4, 3)).simplify(),
            tvec!(Reshape(4, d!(3, 4), d!(4, 3)))
        );
        assert_eq!(
            Reshape(3, d!(3, 4, 2), d!(4, 3, 2)).simplify(),
            tvec!(Reshape(3, d!(3, 4), d!(4, 3)))
        );
        assert_eq!(
            Reshape(3, d!(1, 2, 3), d!(3, 2)).simplify(),
            tvec!(Rm(3), Reshape(3, d!(2, 3), d!(3, 2)))
        );
        assert_eq!(
            Reshape(3, d!(2, 3), d!(1, 3, 2)).simplify(),
            tvec!(Reshape(3, d!(2, 3), d!(3, 2)), Add(3))
        );
        assert_eq!(
            Reshape(3, d!(2, 3, 1), d!(3, 2)).simplify(),
            tvec!(Rm(5), Reshape(3, d!(2, 3), d!(3, 2)))
        );
        assert_eq!(
            Reshape(3, d!(2, 3), d!(3, 2, 1)).simplify(),
            tvec!(Add(5), Reshape(3, d!(2, 3), d!(3, 2)))
        );
        assert_eq!(
            Reshape(2, d!(2, 2, 1), d!(4)).simplify(),
            tvec!(Rm(4), Reshape(2, d!(2, 2), d!(4)))
        );
        assert_eq!(Reshape(1, d!(1, 2), d!(2)).simplify(), tvec!(Rm(1)));
    }

    macro_rules! s {
        ($($a:expr),*) => {&[ $($a.clone().into()),* ]}
    }

    macro_rules! r {
        ($at: expr ; $($from:expr),* => $($to:expr),*) => {
            AxisOp::Reshape($at, tvec!($($from.into()),*),  tvec!($($to.into()),*))
        }
    }

    #[test]
    fn compute_invalid() {
        assert!(compute_shape_with_tf_rules(s![3, 4, 5], s!(100)).is_err());
    }

    #[test]
    fn compute_with_leading_zero() {
        assert_eq!(&*compute_shape_with_tf_rules(s![3, 4, 5], s!(0, 0, 5)).unwrap(), s![3, 4, 5])
    }

    #[test]
    fn compute_with_leading_zero_with_flatten() {
        assert_eq!(
            &*compute_shape_with_tf_rules(s![2, 3, 5, 7], s!(2, 0, 35)).unwrap(),
            s![2, 3, 35]
        )
    }

    #[test]
    fn compute_with_trailing_zero() {
        assert_eq!(&*compute_shape_with_tf_rules(s![3, 4, 5], s!(3, -1, 0)).unwrap(), s![3, 4, 5])
    }

    #[test]
    fn compute_bug_1() {
        let table = SymbolScope::default();
        let s = table.new_with_prefix("S");
        assert_eq!(
            &*compute_shape_with_tf_rules(s![s, 1, 2, 128], s!(0, 0, -1)).unwrap(),
            s![s, 1, 256]
        )
    }

    #[test]
    fn compute_bug_2() {
        let table = SymbolScope::default();
        let b = table.new_with_prefix("B");
        let s = table.new_with_prefix("S");
        assert_eq!(
            &*compute_shape_with_tf_rules(s![s, b, 2, 128], s!(0, 0, -1)).unwrap(),
            s![s, b, 256]
        )
    }

    #[test]
    fn axis_op_rm_begin() {
        assert_eq!(&*to_axis_ops_with_tf_rules(s![1, 2, 3], s!(2, 3)).unwrap(), &[Rm(0)])
    }

    #[test]
    fn axis_op_rm_end() {
        assert_eq!(&*to_axis_ops_with_tf_rules(s![2, 3, 1], s!(2, 3)).unwrap(), &[Rm(2)])
    }

    #[test]
    fn axis_op_insert_begin() {
        assert_eq!(&*to_axis_ops_with_tf_rules(s![2, 3], s!(1, 2, 3)).unwrap(), &[Add(0)])
    }

    #[test]
    fn axis_op_insert_end() {
        assert_eq!(&*to_axis_ops_with_tf_rules(s![2, 3], s!(2, 3, 1)).unwrap(), &[Add(2)])
    }

    #[test]
    fn axis_op_merge() {
        assert_eq!(
            &*to_axis_ops_with_tf_rules(s![2, 3, 5, 7], s!(2, 0, 35)).unwrap(),
            &[r!(2 ; 5,7 => 35 )]
        )
    }

    #[test]
    fn axis_op_complex() {
        assert_eq!(
            &*to_axis_ops_with_tf_rules(s![1, 2, 3, 5, 7], s!(2, 1, 3, 35, 1)).unwrap(),
            &[Rm(0), Add(1), r!(3 ; 5,7 => 35 ), Add(4)]
        )
    }
}
