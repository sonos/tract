use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;

pub trait BinMiniOp: fmt::Debug + objekt::Clone + Send + Sync + 'static + Downcast {
    fn name(&self) -> &'static str;
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
        let a = a.cast_to_dt(op_type)?.into_owned();
        let b = b.cast_to_dt(op_type)?.into_owned();
        self.eval_broadcast(tvec!(a.into_arc_tensor(), b.into_arc_tensor()))
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
    fn unary_with_b_const(&self, b: &Arc<Tensor>) -> Option<UnaryOp>;
}
clone_trait_object!(BinMiniOp);
downcast_rs::impl_downcast!(BinMiniOp);

#[derive(Debug, Clone)]
pub struct InferenceBinOp(pub Box<dyn BinMiniOp>);

impl Op for InferenceBinOp {
    fn name(&self) -> Cow<str> {
        format!("{}Inference", self.0.name()).into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let facts = model.node_input_facts(node.id)?;
        let mut patch = TypedModelPatch::default();
        let operating_datum_type =
            self.0.operating_datum_type(facts[0].datum_type, facts[1].datum_type)?;
        let bin = patch.add_node(
            &*node.name,
            TypedBinOp(self.0.clone()),
            tvec!(node.outputs[0].fact.clone()),
        )?;
        patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(bin, 0))?;
        for i in 0..2 {
            let fact = model.node_input_facts(node.id)?[i];
            let tap = patch.tap_model(model, node.inputs[i])?;
            if fact.datum_type != operating_datum_type {
                let mut fact = fact.clone();
                fact.datum_type = operating_datum_type;
                let cast = patch.chain_after(
                    tap,
                    format!("{}Cast{}", &*node.name, i),
                    super::cast::Cast::new(operating_datum_type),
                    tvec!(fact),
                )?;
                patch.add_edge(OutletId::new(cast, 0), InletId::new(bin, i))?;
            } else {
                patch.add_edge(tap, InletId::new(bin, i))?;
            }
        }
        Ok(Some(patch))
    }

    op_as_typed_op!();
}

impl StatelessOp for InferenceBinOp {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.0.eval_broadcast_and_typecast(inputs)
    }
}

impl InferenceRulesOp for InferenceBinOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;

        s.with(&inputs[0].shape, move |s, a_shape| {
            s.with(&inputs[1].shape, move |s, b_shape| {
                if let Ok(Some(c_shape)) =
                    crate::analyser::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape])
                {
                    s.equals(&outputs[0].shape, c_shape)?;
                }
                Ok(())
            })
        })?;
        s.given_2(&inputs[0].datum_type, &inputs[1].datum_type, move |s, typa, typb| {
            s.equals(&outputs[0].datum_type, self.0.result_datum_type(typa, typb)?)
        })?;
        Ok(())
    }
    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for InferenceBinOp {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(
            self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?,
            &*crate::broadcast::multi_broadcast(&[
                &inputs[0].shape.to_tvec(),
                &inputs[1].shape.to_tvec()
            ])
            .ok_or_else(|| format!(
                "Can not broadcast {:?} and {:?}",
                inputs[0].shape, inputs[1].shape
            ))?
        )?))
    }
}

#[derive(Debug, Clone)]
pub struct Nary(pub Box<dyn BinMiniOp>, pub bool);

impl Nary {
    fn normalize_t<T>(t: &mut Tensor, n: usize) -> TractResult<()>
    where
        T: Datum + std::ops::DivAssign<T> + Copy,
        usize: num_traits::AsPrimitive<T>,
    {
        use num_traits::AsPrimitive;
        let mut t = t.to_array_view_mut::<T>()?;
        let n: T = n.as_();
        t /= &ndarray::arr0(n);
        Ok(())
    }
}

impl Op for Nary {
    fn name(&self) -> Cow<str> {
        format!("{}Nary", self.0.name()).into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Nary {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut t = inputs[0].clone();
        for i in inputs[1..].into_iter() {
            t = self.0.eval_broadcast_and_typecast(tvec!(t.clone(), i.clone()))?.remove(0);
        }
        if self.1 {
            let mut t = t.into_tensor();
            dispatch_numbers!(Self::normalize_t(t.datum_type())(&mut t, inputs.len()))?;
            Ok(tvec!(t.into_arc_tensor()))
        } else {
            Ok(tvec!(t))
        }
    }
}

impl InferenceRulesOp for Nary {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        let n = inputs.len();
        s.equals_all((0..n).map(|i| (&inputs[i].datum_type).bex()).collect())?;
        s.equals_all((0..n).map(|i| inputs[i].rank.bex()).collect())?;
        s.given(&inputs[0].rank, move |s, rank: i32| {
            for dim in 0..(rank as usize) {
                s.equals(&inputs[0].shape[dim], &outputs[0].shape[dim])?;
                s.equals_all((0..n as usize).map(|i| inputs[i].shape[dim].bex()).collect())?;
            }
            Ok(())
        })
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<Vec<_>>();
        let mut wire = inputs[0];
        for (ix, i) in inputs[1..].iter().enumerate() {
            wire = target.wire_node(
                format!("{}-{}", node.name, ix),
                TypedBinOp(self.0.clone()),
                [wire, *i].as_ref(),
            )?[0];
        }
        if self.1 {
            let n = target.add_const(
                format!("{}-n", node.name),
                tensor0(inputs.len() as i32)
                    .cast_to_dt(node.outputs[0].fact.datum_type.concretize().unwrap())?
                    .into_owned(),
            )?;
            wire = target.wire_node(
                format!("{}-norm", node.name),
                crate::ops::math::div::bin(),
                [wire, n.into()].as_ref(),
            )?[0];
        }
        Ok(tvec!(wire))
    }

    inference_op_as_op!();
}

#[derive(Debug, Clone)]
pub struct TypedBinOp(pub Box<dyn BinMiniOp>);

impl Op for TypedBinOp {
    fn name(&self) -> Cow<str> {
        format!("{}Typed", self.0.name()).into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        for i in 0..2 {
            use super::array::TypedMultiBroadcastTo;
            let prec = model.node(node.inputs[i].node);
            if prec.op_is::<TypedMultiBroadcastTo>() {
                return Ok(Some(TypedModelPatch::shunt_one_op(model, prec)?));
            }
        }
        if let Some(a) = inputs[0].konst.clone() {
            let op = UnaryOp::new(self.0.clone(), a.clone());
            return Ok(Some(TypedModelPatch::replace_single_op(
                &model,
                &node,
                &node.inputs[1..2],
                op,
            )?));
        }
        if let Some(b) = inputs[1].konst.clone() {
            if let Some(op) = self.0.unary_with_b_const(&b) {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    &model,
                    &node,
                    &node.inputs[0..1],
                    op,
                )?));
            }
        }
        if inputs[0].shape == inputs[1].shape {
            let op = MergeOp(self.0.clone());
            return Ok(Some(TypedModelPatch::replace_single_op(&model, &node, &node.inputs, op)?));
        }
        Ok(None)
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for TypedBinOp {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.0.eval_broadcast(inputs)
    }
}

impl TypedOp for TypedBinOp {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(
            self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?,
            &*crate::broadcast::multi_broadcast(&[
                &inputs[0].shape.to_tvec(),
                &inputs[1].shape.to_tvec()
            ])
            .unwrap()
        )?))
    }

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        let a = model.outlet_fact(node.inputs[0])?;
        let b = model.outlet_fact(node.inputs[1])?;
        let c = &self.output_facts(&[a, b])?[0];
        let a_pad = c.shape.rank() - a.shape.rank();
        let b_pad = c.shape.rank() - b.shape.rank();
        Ok((0..c.shape.rank())
            .into_iter()
            .map(|axis| {
                let mut info = AxisInfo {
                    inputs: tvec!(None, None),
                    outputs: tvec!(Some(axis)),
                    period: 1,
                    disposable: true,
                };
                if axis >= a_pad && a.shape.dim(axis - a_pad) == 1.to_dim() {
                    info.inputs[0] = Some(axis - a_pad)
                }
                if axis >= b_pad && b.shape.dim(axis - b_pad) == 1.to_dim() {
                    info.inputs[1] = Some(axis - b_pad)
                }
                info
            })
            .collect())
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        use crate::pulse::delay::Delay;
        let delay = (0..2)
            .map(|ix| target.outlet_fact(mapping[&node.inputs[ix]]).unwrap().delay)
            .max()
            .unwrap();
        let mut output_fact = target.outlet_fact(mapping[&node.inputs[0]])?.clone();
        output_fact.shape = crate::broadcast::multi_broadcast(&[
            &target.outlet_fact(mapping[&node.inputs[0]])?.shape,
            &target.outlet_fact(mapping[&node.inputs[1]])?.shape,
        ])
        .unwrap();
        output_fact.delay = delay;

        let id = target.add_node(&*node.name, self.clone(), tvec!(output_fact))?;
        for ix in 0..2 {
            let input = mapping[&node.inputs[ix]];
            let input_delay = target.outlet_fact(input)?.delay;
            let source = if input_delay < delay {
                let add_delay = delay - input_delay;
                let fact = target.outlet_fact(input)?.clone();
                let mut fixed_fact = fact.clone();
                fixed_fact.delay += add_delay;
                let id = target.chain_after(
                    mapping[&node.inputs[ix]],
                    format!("{}/Delay", &*node.name),
                    Delay::new(fact, add_delay, 0),
                    tvec!(fixed_fact),
                )?;
                OutletId::new(id, 0)
            } else {
                input
            };
            target.add_edge(source, InletId::new(id, ix))?;
        }
        Ok(tvec!(OutletId::new(id, 0)))
    }
}

#[derive(Debug, Clone, new)]
pub struct UnaryOp {
    pub mini_op: Box<dyn BinMiniOp>,
    pub a: Arc<Tensor>,
}

impl Op for UnaryOp {
    fn name(&self) -> Cow<str> {
        format!("{}Unary", self.mini_op.name()).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("a: {:?}", self.a)])
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for UnaryOp {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.mini_op.eval_broadcast(tvec!(self.a.clone(), inputs[0].clone()))
    }
}

impl TypedOp for UnaryOp {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(
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

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        let b = model.outlet_fact(node.inputs[0])?;
        if b.shape.rank() < self.a.shape().len() {
            return Ok(AxesInfo::none());
        }
        let mut invs = vec![];
        for i in 0..b.shape.rank() - self.a.shape().len() {
            invs.push(AxisInfo::simple(i))
        }
        for &d in self.a.shape() {
            invs.push(AxisInfo::simple(invs.len()).with_period(d))
        }
        return Ok(invs.into_iter().collect());
    }

    fn dispose_dummy_axis(
        &self,
        _model: &TypedModel,
        node: &TypedNode,
        axis: usize,
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        let a_pad = node.outputs[0].fact.shape.rank() - self.a.shape().len();
        if axis > a_pad {
            let a = self.a.clone().into_tensor();
            let mut shape = a.shape().to_vec();
            assert_eq!(shape[axis - a_pad], 1);
            shape.remove(axis - a_pad);
            let a = unsafe { a.into_shape(&*shape)? };
            Ok(Some(Box::new(UnaryOp::new(self.mini_op.clone(), a.into_arc_tensor()))))
        } else {
            Ok(None)
        }
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
        let mut fact = target.outlet_fact(input)?.clone();
        fact.dt = self.a.datum_type();
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
}

#[derive(Debug, Clone)]
pub struct MergeOp(pub Box<dyn BinMiniOp>);

impl Op for MergeOp {
    fn name(&self) -> Cow<str> {
        format!("{}Merge", self.0.name()).into()
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for MergeOp {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.0.eval_broadcast(inputs)
    }
}

impl TypedOp for MergeOp {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(
            self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?,
            &*crate::broadcast::multi_broadcast(&[
                &inputs[0].shape.to_tvec(),
                &inputs[1].shape.to_tvec()
            ])
            .unwrap()
        )?))
    }

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        let a = model.outlet_fact(node.inputs[0])?;
        Ok((0..a.shape.rank()).into_iter().map(|axis| AxisInfo::simple(axis)).collect())
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
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        use crate::pulse::delay::Delay;
        let delay = (0..2)
            .map(|ix| target.outlet_fact(mapping[&node.inputs[ix]]).unwrap().delay)
            .max()
            .unwrap();
        let mut output_fact = target.outlet_fact(mapping[&node.inputs[0]])?.clone();
        output_fact.delay = delay;
        let id = target.add_node(&*node.name, self.clone(), tvec!(output_fact))?;
        for ix in 0..2 {
            let input = mapping[&node.inputs[ix]];
            let input_delay = target.outlet_fact(input)?.delay;
            let source = if input_delay < delay {
                let add_delay = delay - input_delay;
                let fact = target.outlet_fact(input)?.clone();
                let mut fixed_fact = fact.clone();
                fixed_fact.delay += add_delay;
                let id = target.chain_after(
                    mapping[&node.inputs[ix]],
                    format!("{}/Delay", &*node.name),
                    Delay::new(fact, add_delay, 0),
                    tvec!(fixed_fact),
                )?;
                OutletId::new(id, 0)
            } else {
                input
            };
            target.add_edge(source, InletId::new(id, ix))?;
        }
        Ok(tvec!(OutletId::new(id, 0)))
    }
}

#[derive(Debug, Clone)]
pub struct MergeOpUnicast(pub Box<dyn BinMiniOp>);

impl Op for MergeOpUnicast {
    fn name(&self) -> Cow<str> {
        format!("{}MergeUnicast", self.0.name()).into()
    }

    op_as_typed_op!();
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
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

#[macro_export]
macro_rules! bin_to_super_type {
    ($func:ident, $Op:ident, $( [$($typ:ident),*] => $cab:expr),*) => {
        bin_to_super_type!($func, $Op, flip: |_, _| None, $( [$($typ),*] => $cab),*);
    };
    ($func:ident, $Op:ident, flip: $flip:expr, $( [$($typ:ident),*] => $cab:expr),*) => {
        #[derive(Debug, Clone)]
        pub struct $Op;
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
                bail!("{} does not support {:?}", self.name(), a.datum_type());
            }

            fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
                $(
                    $(if c.datum_type() == $typ::datum_type() {
                        let a = a.to_array_view::<$typ>()?;
                        let b = b.to_array_view::<$typ>()?;
                        let mut c = c.to_array_view_mut::<$typ>()?;
                        ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).apply($cab);
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?}", self.name(), c.datum_type());
            }

            fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                a.common_super_type(b).ok_or_else(|| format!("No super type for {:?} and {:?}", a, b).into())
            }

            fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                self.operating_datum_type(a, b)
            }

            fn unary_with_b_const(&self, b: &Arc<Tensor>) -> Option<$crate::ops::binary::UnaryOp> {
                ($flip)(self, b)
            }
        }

        pub mod $func {
            pub fn bin() -> $crate::ops::binary::InferenceBinOp {
                $crate::ops::binary::InferenceBinOp(Box::new(super::$Op))
            }
            pub fn unary(t: std::sync::Arc<$crate::prelude::Tensor>) -> $crate::ops::binary::UnaryOp {
                $crate::ops::binary::UnaryOp::new(Box::new(super::$Op), t)
            }
        }
    };
}

macro_rules! bin_to_bool {
    ($func:ident, $Op:ident, $( [$($typ:ident),*] => $cab:expr ),*) => {
        bin_to_bool!($func, $Op, flip: |_, _| None, $( [$($typ),*] => $cab),*);
    };
    ($func:ident, $Op:ident, flip: $flip:expr, $( [$($typ:ident),*] => $cab:expr),*) => {
        #[derive(Debug, Clone)]
        pub struct $Op;
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
                bail!("{} does not support {:?}", self.name(), c.datum_type());
            }

            fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                a.common_super_type(b).ok_or_else(|| format!("No super type for {:?} and {:?}", a, b).into())
            }

            fn result_datum_type(&self, _a: DatumType, _b: DatumType) -> TractResult<DatumType> {
                Ok(bool::datum_type())
            }

            fn unary_with_b_const(&self, b: &Arc<Tensor>) -> Option<$crate::ops::binary::UnaryOp> {
                ($flip)(self, b)
            }
        }

        pub mod $func {
            pub fn bin() -> $crate::ops::binary::InferenceBinOp {
                $crate::ops::binary::InferenceBinOp(Box::new(super::$Op))
            }
            pub fn unary(t: std::sync::Arc<$crate::prelude::Tensor>) -> $crate::ops::binary::UnaryOp {
                $crate::ops::binary::UnaryOp::new(Box::new(super::$Op), t)
            }
        }
    };
}

#[inline]
pub fn commute(op: &dyn BinMiniOp, t: &Arc<Tensor>) -> Option<UnaryOp> {
    Some(UnaryOp::new(objekt::clone_box(op), t.clone()))
}
