use crate::internal::*;
use crate::ops::invariants::*;
use downcast_rs::Downcast;
use std::fmt;

pub trait BinMiniOp:
    fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + DynHash
{
    fn name(&self) -> &'static str;
    fn validation(&self) -> Validation {
        Validation::Accurate
    }
    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType>;
    fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType>;
    fn eval_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()>;
    fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()>;
    fn eval_broadcast_and_typecast(
        &self,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let (a, b) = args_2!(inputs);
        let op_type = self.operating_datum_type(a.datum_type(), b.datum_type())?;
        let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])
            .ok_or("Can not compute resulting shape")?;
        let a = a.cast_to_dt(op_type)?;
        let b = b.cast_to_dt(op_type)?;
        let c_dt = self.result_datum_type(a.datum_type(), b.datum_type())?;
        let mut c = unsafe { Tensor::uninitialized_dt(c_dt, &*c_shape)? };
        self.eval_out_of_place(&mut c, a.as_ref(), b.as_ref())?;
        Ok(tvec!(c.into_arc_tensor()))
    }
    fn eval_broadcast(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (a, b) = args_2!(inputs);
        let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])
            .ok_or("Can not compute resulting shape")?;
        let c_dt = self.result_datum_type(a.datum_type(), b.datum_type())?;
        let mut c = unsafe { Tensor::uninitialized_dt(c_dt, &*c_shape)? };
        self.eval_out_of_place(&mut c, a.as_ref(), b.as_ref())?;
        Ok(tvec!(c.into_arc_tensor()))
    }
    #[allow(unused_variables)]
    fn unary_with_b_const(&self, b: &Arc<Tensor>) -> Option<UnaryOp> {
        None
    }
    #[allow(unused_variables)]
    fn declutter_bin(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
    #[allow(unused_variables)]
    fn declutter_unary(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        a: &Arc<Tensor>,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
    #[allow(unused_variables)]
    fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
        tvec!()
    }
}
dyn_clone::clone_trait_object!(BinMiniOp);
downcast_rs::impl_downcast!(BinMiniOp);

impl Hash for Box<dyn BinMiniOp> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}

#[derive(Debug, Clone, Hash)]
pub struct TypedBinOp(pub Box<dyn BinMiniOp>);
tract_linalg::impl_dyn_hash!(TypedBinOp);

impl Op for TypedBinOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    canonic!();
    op_core_mir!();
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for TypedBinOp {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        debug_assert_eq!(inputs[0].rank(), inputs[1].rank());
        self.0.eval_broadcast(inputs)
    }
}

impl TypedOp for TypedBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!("Typed ops require rank match. Invalid inputs for {}: {:?}", self.name(), inputs);
        }
        Ok(tvec!(TypedFact::dt_shape(
            self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?,
            &*crate::broadcast::multi_broadcast(&[
                &inputs[0].shape.to_tvec(),
                &inputs[1].shape.to_tvec()
            ])
            .ok_or_else(|| format!(
                "Can not broadcast shapes a:{:?} b:{:?}",
                &inputs[0], &inputs[1]
            ))?
        )?))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let a = model.outlet_fact(node.inputs[0])?;
        let b = model.outlet_fact(node.inputs[1])?;
        assert!(a.rank() == b.rank());
        let rank = a.rank();
        Ok((0..rank)
            .into_iter()
            .map(|axis| AxisInfo {
                inputs: tvec!(Some(axis), Some(axis)),
                outputs: tvec!(Some(axis)),
                period: 1,
                disposable: true,
            })
            .collect())
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().maybe_product()?;
        Ok(self
            .0
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let a_input = node.inputs[0];
        let b_input = node.inputs[1];
        let a = model.outlet_fact(a_input)?;
        let b = model.outlet_fact(b_input)?;
        if a.shape == b.shape {
            let a_prec_node = model.node(a_input.node);
            let b_prec_node = model.node(b_input.node);
            let a_sliced = a_prec_node.op.slice_output(
                model,
                a_prec_node,
                patch,
                a_input.slot,
                axis,
                start,
                end,
            )?;
            let b_sliced = b_prec_node.op.slice_output(
                model,
                b_prec_node,
                patch,
                b_input.slot,
                axis,
                start,
                end,
            )?;
            if let (Some(a), Some(b)) = (a_sliced, b_sliced) {
                return Ok(Some(patch.wire_node(&*node.name, self.clone(), &[a, b])?[0]));
            }
        }
        Ok(None)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(patch) = self.0.declutter_bin(model, node)? {
            return Ok(Some(patch));
        }
        for i in 0..2 {
            use super::array::MultiBroadcastTo;
            let prec = model.node(node.inputs[i].node);
            if prec.op_is::<MultiBroadcastTo>() {
                return Ok(Some(TypedModelPatch::shunt_one_op(model, prec)?));
            }
        }
        if let Some(a) = inputs[0].konst.clone() {
            let op = UnaryOp::new(self.0.clone(), a.into_arc_tensor());
            return Ok(Some(TypedModelPatch::replace_single_op(
                &model,
                &node,
                &node.inputs[1..2],
                op,
            )?));
        }
        if let Some(b) = inputs[1].konst.clone() {
            let b = b.into_arc_tensor();
            if let Some(op) = self.0.unary_with_b_const(&b) {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    &model,
                    &node,
                    &node.inputs[0..1],
                    op,
                )?));
            }
        }
        return Ok(None);
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?
            == inputs[0].datum_type
            && inputs[0] == inputs[1]
        {
            Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                MergeOpUnicast(self.0.clone()),
            )?))
        } else {
            Ok(None)
        }
    }

    fn pulsify(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        pulsify_bin(node, self, target, mapping)
    }

    as_op!();
}

impl PulsedOp for TypedBinOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        fact.shape = crate::broadcast::multi_broadcast(&[&inputs[0].shape, &inputs[1].shape])
            .ok_or_else(|| {
                format!("Can not broadcast: {:?} and {:?}", inputs[0].shape, inputs[1].shape)
            })?;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

fn pulsify_bin(
    node: &TypedNode,
    op: &dyn PulsedOp,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<TVec<OutletId>> {
    use crate::pulse::delay::Delay;
    let delay = (0..2)
        .map(|ix| target.outlet_fact(mapping[&node.inputs[ix]]).unwrap().delay)
        .max()
        .unwrap();
    let mut inputs = tvec!();
    for ix in 0..2 {
        let mut input = mapping[&node.inputs[ix]];
        let fact = target.outlet_fact(input)?.clone();
        if fact.delay < delay {
            let add_delay = delay - fact.delay;
            input = target.wire_node(
                format!("{}.Delay", &*node.name),
                Delay::new(&fact, add_delay, 0),
                &[input],
            )?[0];
        }
        inputs.push(input);
    }
    target.wire_node(&*node.name, dyn_clone::clone_box(op), &*inputs)
}

#[derive(Debug, Clone, new, Hash)]
pub struct UnaryOp {
    pub mini_op: Box<dyn BinMiniOp>,
    pub a: Arc<Tensor>,
}
tract_linalg::impl_dyn_hash!(UnaryOp);

impl Op for UnaryOp {
    fn name(&self) -> Cow<str> {
        self.mini_op.name().into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("a: {:?}", self.a)])
    }

    fn validation(&self) -> Validation {
        self.mini_op.validation()
    }

    canonic!();
    op_core_lir_mir!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for UnaryOp {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        debug_assert_eq!(self.a.rank(), inputs[0].rank());
        self.mini_op.eval_broadcast(tvec!(self.a.clone(), inputs[0].clone()))
    }
}

impl TypedOp for UnaryOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if self.a.rank() != inputs[0].rank() {
            bail!("Rank mismatch: constant: {:?}, input: {:?}", self.a, inputs[0]);
        }
        Ok(tvec!(TypedFact::dt_shape(
            self.mini_op.result_datum_type(self.a.datum_type(), inputs[0].datum_type)?,
            &*crate::broadcast::multi_broadcast(&[
                &*self.a.shape().iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
                &*inputs[0].shape.to_tvec()
            ])
            .ok_or_else(|| format!(
                "Failed to broadcast {:?} and {:?}",
                self.a.shape(),
                inputs[0].shape
            ))?
        )?))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let b = model.outlet_fact(node.inputs[0])?;
        debug_assert_eq!(self.a.rank(), b.rank());
        Ok(self
            .a
            .shape()
            .iter()
            .enumerate()
            .map(|(ix, d)| AxisInfo::simple(ix).with_period(*d))
            .collect())
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let b = model.outlet_fact(node.inputs[0])?;
        debug_assert_eq!(self.a.rank(), b.rank());
        let prec = model.node(node.inputs[0].node);
        let wire = prec.op().as_typed().unwrap().slice_output(
            model,
            &prec,
            patch,
            node.inputs[0].slot,
            axis,
            start,
            end,
        )?;
        let wire = if let Some(w) = wire { w } else { return Ok(None) };
        let a = if self.a.shape()[axis] != 1 {
            self.a.slice(axis, start, end)?.into_arc_tensor()
        } else {
            self.a.clone()
        };
        Ok(Some(
            patch.wire_node(
                format!("{}.sliced-{}-{}", node.name, start, end),
                UnaryOp::new(self.mini_op.clone(), a),
                &[wire],
            )?[0],
        ))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().maybe_product()?;
        let mut cost: TVec<_> = self
            .mini_op
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect();
        cost.push((Cost::Params(self.a.datum_type()), self.a.len().into()));
        Ok(cost)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.mini_op.declutter_unary(model, node, &self.a)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let mut a = self.a.clone().into_tensor();
        if change.change_tensor_broadcast_aware(&mut a).is_ok() {
            let op = Some(Box::new(UnaryOp::new(self.mini_op.clone(), a.into_arc_tensor())) as _);
            Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
        } else {
            Ok(None)
        }
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

    as_op!();
}

impl PulsedOp for UnaryOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type =
            self.mini_op.result_datum_type(inputs[0].datum_type, self.a.datum_type())?;
        fact.shape = crate::broadcast::multi_broadcast(&[
            &inputs[0].shape,
            &self.a.shape().iter().map(|d| d.into()).collect(),
        ])
        .unwrap();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

/*
#[derive(Debug, Clone, Hash)]
pub struct MergeOp(pub Box<dyn BinMiniOp>);
tract_linalg::impl_dyn_hash!(MergeOp);

impl Op for MergeOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    canonic!();
    op_core_lir_mir!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for MergeOp {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.0.eval_broadcast(inputs)
    }
}

impl TypedOp for MergeOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!("MergeOp expect inputs of same rank, got {:?}", inputs);
        }
        Ok(tvec!(TypedFact::dt_shape(
            self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?,
            &*crate::broadcast::multi_broadcast(&[
                &inputs[0].shape.to_tvec(),
                &inputs[1].shape.to_tvec()
            ])
            .ok_or_else(|| format!(
                "Could not co-broadcast {:?} and {:?}",
                inputs[0].shape, inputs[1].shape
            ))?
        )?))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let a = model.outlet_fact(node.inputs[0])?;
        let b = model.outlet_fact(node.inputs[1])?;
        if a.shape == b.shape {
            Invariants::new_element_wise(model, node)
        } else {
            Ok(Invariants::none())
        }
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(Some(AxisChangeConsequence {
            substitute_op: None,
            wire_changes: tvec!(
                (InOut::In(0), change.clone()),
                (InOut::In(1), change.clone()),
                (InOut::Out(0), change.clone())
            ),
        }))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().maybe_product()?;
        Ok(self
            .0
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let a_input = node.inputs[0];
        let b_input = node.inputs[1];
        let a = model.outlet_fact(a_input)?;
        let b = model.outlet_fact(b_input)?;
        if a.shape == b.shape {
            let a_prec_node = model.node(a_input.node);
            let b_prec_node = model.node(b_input.node);
            let a_sliced = a_prec_node.op.slice_output(
                model,
                a_prec_node,
                patch,
                a_input.slot,
                axis,
                start,
                end,
            )?;
            let b_sliced = b_prec_node.op.slice_output(
                model,
                b_prec_node,
                patch,
                b_input.slot,
                axis,
                start,
                end,
            )?;
            if let (Some(a), Some(b)) = (a_sliced, b_sliced) {
                return Ok(Some(patch.wire_node(&*node.name, self.clone(), &[a, b])?[0]));
            }
        }
        Ok(None)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?
            == inputs[0].datum_type
            && inputs[0] == inputs[1]
        {
            Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                MergeOpUnicast(self.0.clone()),
            )?))
        } else {
            Ok(None)
        }
    }

    fn pulsify(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        pulsify_bin(node, self, target, mapping)
    }

    as_op!();
}

impl PulsedOp for MergeOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        fact.shape = crate::broadcast::multi_broadcast(&[&inputs[0].shape, &inputs[1].shape])
            .ok_or_else(|| {
                format!("Can not broadcast: {:?} and {:?}", inputs[0].shape, inputs[1].shape)
            })?;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
*/

#[derive(Debug, Clone, Hash)]
pub struct MergeOpUnicast(pub Box<dyn BinMiniOp>);
tract_linalg::impl_dyn_hash!(MergeOpUnicast);

impl Op for MergeOpUnicast {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    op_core_lir_mir!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for MergeOpUnicast {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (a, b) = args_2!(inputs);
        let mut b = b.into_tensor();
        self.0.eval_in_place(a.as_ref(), &mut b)?;
        Ok(tvec!(b.into_arc_tensor()))
    }
}

impl TypedOp for MergeOpUnicast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        debug_assert_eq!(inputs[0].shape, inputs[1].shape);
        Ok(tvec!(inputs[0].clone()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().maybe_product()?;
        Ok(self
            .0
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
    }

    as_op!();
}

impl PulsedOp for MergeOpUnicast {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        fact.shape = inputs[0].shape.clone();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

#[macro_export]
macro_rules! bin_to_super_type {
    ($func:ident, $Op:ident,
     $(cost: $cost:expr,)?
     $(declutter_bin: $declutter_bin:expr,)?
     $(declutter_unary: $declutter_unary:expr,)?
     $(flip: $flip:expr,)?
     $(out_of_place: $out_of_place:expr,)?
     $(validation: $validation:expr,)?
     $( [$($typ:ident),*] => $cab:expr),*) => {
        #[derive(Debug, Clone, Hash)]
        pub struct $Op;
        tract_linalg::impl_dyn_hash!($Op);
        impl $crate::ops::binary::BinMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }

            fn eval_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut $typ, &$typ, &$typ) -> () = $cab;
                        let a = a.as_slice::<$typ>()?;
                        let b = b.as_slice_mut::<$typ>()?;
                        for i in 0..a.len() {
                            let mut c = $typ::default();
                            cab(&mut c, &a[i], &b[i]);
                            b[i] = c;
                        }
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?} (inplace)", self.name(), a.datum_type());
            }

            fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
                $(if $out_of_place(c, a, b)? { return Ok(()) } )?
                $(
                    $(if c.datum_type() == $typ::datum_type() {
                        let a = a.to_array_view::<$typ>()?;
                        let b = b.to_array_view::<$typ>()?;
                        let mut c = c.to_array_view_mut::<$typ>()?;
                        $crate::ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).apply($cab);
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?} (out of place)", self.name(), c.datum_type());
            }

            fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                a.common_super_type(b).ok_or_else(|| format!("No super type for {:?} and {:?}", a, b).into())
            }

            fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                self.operating_datum_type(a, b)
            }

            $(
                fn unary_with_b_const(&self, b: &Arc<Tensor>) -> Option<$crate::ops::binary::UnaryOp> {
                    ($flip)(self, b)
                }
            )?
            $(
                fn declutter_bin(
                    &self,
                    model: &TypedModel,
                    node: &TypedNode,
                ) -> TractResult<Option<TypedModelPatch>> {
                    ($declutter_bin)(self, model, node)
                }
            )?
            $(
                fn declutter_unary(
                    &self,
                    model: &TypedModel,
                    node: &TypedNode,
                    a: &Arc<Tensor>,
                ) -> TractResult<Option<TypedModelPatch>> {
                    ($declutter_unary)(self, model, node, a)
                }
            )?
            $(
                fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
                    ($cost)(dt)
                }
            )?
            $(
            fn validation(&self) -> Validation {
                $validation
            }
            )?
        }

        pub mod $func {
            pub fn bin_typed() -> $crate::ops::binary::TypedBinOp {
                $crate::ops::binary::TypedBinOp(Box::new(super::$Op))
            }
            pub fn unary(t: std::sync::Arc<$crate::prelude::Tensor>) -> $crate::ops::binary::UnaryOp {
                $crate::ops::binary::UnaryOp::new(Box::new(super::$Op), t)
            }
        }
    };
}

macro_rules! bin_to_bool {
    ($func:ident, $Op:ident,
     $( cost: $cost:expr, )?
     $( flip: $flip:expr, )?
     $( [$($typ:ident),*] => $cab:expr),*) => {
        #[derive(Debug, Clone, Hash)]
        pub struct $Op;
        tract_linalg::impl_dyn_hash!($Op);
        impl $crate::ops::binary::BinMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }

            #[allow(unreachable_code)]
            fn eval_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut bool, &bool, &bool) -> () = $cab;
                        let a = a.as_slice::<bool>()?;
                        let b = b.as_slice_mut::<bool>()?;
                        for i in 0..a.len() {
                            let mut c = bool::default();
                            cab(&mut c, &a[i], &b[i]);
                            b[i] = c;
                        }
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?}", self.name(), a.datum_type());
            }

            fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let a = a.to_array_view::<$typ>()?;
                        let b = b.to_array_view::<$typ>()?;
                        let mut c = c.to_array_view_mut::<bool>()?;
                        ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).apply($cab);
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?}", self.name(), a.datum_type());
            }

            fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                a.common_super_type(b).ok_or_else(|| format!("No super type for {:?} and {:?}", a, b).into())
            }

            fn result_datum_type(&self, _a: DatumType, _b: DatumType) -> TractResult<DatumType> {
                Ok(bool::datum_type())
            }

            $(
                fn unary_with_b_const(&self, b: &Arc<Tensor>) -> Option<$crate::ops::binary::UnaryOp> {
                    ($flip)(self, b)
                }
            )?
            $(
                fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
                    ($cost)(dt)
                }
            )?
        }

        pub mod $func {
            /*
            pub fn bin() -> $crate::ops::binary::InferenceBinOp {
                $crate::ops::binary::InferenceBinOp(Box::new(super::$Op))
            }
            */
            pub fn bin_typed() -> $crate::ops::binary::TypedBinOp {
                $crate::ops::binary::TypedBinOp(Box::new(super::$Op))
            }
            pub fn unary(t: std::sync::Arc<$crate::prelude::Tensor>) -> $crate::ops::binary::UnaryOp {
                $crate::ops::binary::UnaryOp::new(Box::new(super::$Op), t)
            }
        }
    };
}

#[inline]
pub fn commute(op: &dyn BinMiniOp, t: &Arc<Tensor>) -> Option<UnaryOp> {
    Some(UnaryOp::new(dyn_clone::clone_box(op), t.clone()))
}
