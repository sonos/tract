use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;
use tract_data::itertools::izip;

pub fn wire_cast(
    prefix: &str,
    target: &mut TypedModel,
    inputs: &[OutletId],
    operating_datum_type: DatumType,
) -> TractResult<TVec<OutletId>> {
    let mut wires = tvec!();
    for (ix, mut wire) in inputs.iter().copied().enumerate() {
        if target.outlet_fact(wire)?.datum_type != operating_datum_type {
            wire = target.wire_node(
                format!("{prefix}.cast-{ix}"),
                crate::ops::cast::cast(operating_datum_type),
                &[wire],
            )?[0];
        }
        wires.push(wire);
    }
    Ok(wires)
}

pub fn wire_rank_broadcast(
    prefix: impl AsRef<str>,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let facts = inputs
        .iter()
        .map(|o| target.outlet_fact(*o).map(|ok| ok.clone()))
        .collect::<TractResult<TVec<_>>>()?;
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
    target.wire_node(prefix, &op.into(), &wires)
}

pub trait BinMiniOp: fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast {
    fn name(&self) -> &'static str;
    fn validation(&self) -> Validation {
        Validation::Accurate
    }
    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        a.common_super_type(b).with_context(|| format_err!("No super type for {:?} and {:?}", a, b))
    }
    fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType>;
    fn eval_unicast_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()>;
    fn eval_uniform_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()>;
    fn eval_in_a(&self, a: &mut Tensor, b: &Tensor) -> TractResult<()>;
    fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()>;
    fn generic_eval(&self, a: TValue, b: TValue) -> TractResult<Tensor> {
        let c_dt = self.result_datum_type(a.datum_type(), b.datum_type())?;
        if c_dt == b.datum_type() && a.len() == 1 {
            let mut b = b.into_tensor();
            self.eval_uniform_in_place(&a, &mut b)?;
            Ok(b)
        } else if a.shape() == b.shape() && c_dt == b.datum_type() {
            let mut b = b.into_tensor();
            self.eval_unicast_in_place(&a, &mut b)?;
            Ok(b)
        } else {
            let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])
                .ok_or_else(|| format_err!("Can not compute resulting shape"))?;
            if &*c_shape == a.shape() && c_dt == a.datum_type() {
                let mut a = a.into_tensor();
                self.eval_in_a(&mut a, &b)?;
                Ok(a)
            } else {
                let mut c = unsafe { Tensor::uninitialized_dt(c_dt, &c_shape)? };
                self.eval_out_of_place(&mut c, &a, &b)?;
                Ok(c)
            }
        }
    }
    fn eval(&self, a: TValue, b: TValue) -> TractResult<Tensor> {
        self.generic_eval(a, b)
    }
    #[allow(unused_variables)]
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
    #[allow(unused_variables)]
    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
    #[allow(unused_variables)]
    fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
        tvec!()
    }
    fn as_linalg_binop(&self) -> Option<tract_linalg::mmm::BinOp> {
        None
    }
}
dyn_clone::clone_trait_object!(BinMiniOp);
downcast_rs::impl_downcast!(BinMiniOp);

#[derive(Debug, Clone)]
pub struct TypedBinOp(pub Box<dyn BinMiniOp>);

impl Op for TypedBinOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    op_as_typed_op!();
}

impl EvalOp for TypedBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        ensure!(a.rank() == b.rank());
        Ok(tvec!(self.0.eval(a, b)?.into_tvalue()))
    }
}

impl TypedOp for TypedBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!("Typed ops require rank match. Invalid inputs for {}: {:?}", self.name(), inputs);
        }
        Ok(tvec!(self.0.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?.fact(
            &*crate::broadcast::multi_broadcast(&[
                &inputs[0].shape.to_tvec(),
                &inputs[1].shape.to_tvec()
            ])
            .ok_or_else(|| format_err!(
                "Can not broadcast shapes a:{:?} b:{:?}",
                &inputs[0],
                &inputs[1]
            ))?
        )))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let AxisOp::Rm(rm) = change {
            let (inputs, outputs) = model.node_facts(node.id)?;
            if !inputs[0].shape[*rm].is_one()
                || !inputs[0].shape[*rm].is_one()
                || !outputs[0].shape[*rm].is_one()
            {
                return Ok(None);
            }
        }
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().product();
        Ok(self
            .0
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: usize,
        _end: usize,
    ) -> TractResult<Option<TVec<OutletId>>> {
        Ok(Some(patch.wire_node(prefix, self.clone(), inputs)?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.0.declutter(model, node)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let facts = model.node_input_facts(node.id)?;
        if self.0.result_datum_type(facts[0].datum_type, facts[1].datum_type)?
            == facts[0].datum_type
            && facts[0].without_value() == facts[1].without_value()
        {
            Ok(Some(
                TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &node.inputs,
                    MergeOpUnicast(self.0.clone()),
                )?
                .with_context("Unicast"),
            ))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

#[derive(Debug, Clone)]
pub struct MergeOpUnicast(pub Box<dyn BinMiniOp>);

impl Op for MergeOpUnicast {
    fn name(&self) -> Cow<str> {
        format!("{}Unicast", self.0.name()).into()
    }

    op_as_typed_op!();
}

impl EvalOp for MergeOpUnicast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        let mut b = b.into_tensor();
        self.0.eval_unicast_in_place(&a, &mut b)?;
        Ok(tvec!(b.into_tvalue()))
    }
}

impl TypedOp for MergeOpUnicast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        debug_assert_eq!(inputs[0].shape, inputs[1].shape);
        Ok(tvec!(inputs[0].without_value()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().product();
        Ok(self
            .0
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.0.declutter(model, node)
    }

    as_op!();
}

#[macro_export]
macro_rules! bin_to_super_type {
    ($func:ident, $Op:ident,
     $(codegen: $codegen:expr,)?
     $(cost: $cost:expr,)?
     $(declutter: $declutter:expr,)?
     $(eval_override: $eval_override: expr,)?
     $(linalg: $linalg:ident,)?
     $(operating_datum_type: $operating_datum_type:expr,)?
     $(out_of_place: $out_of_place:expr,)?
     $(validation: $validation:expr,)?
     $(q: $([$($typ_dt:ident),*] => $cab_dt:expr),* ;)?
     $( [$($typ:ident),*] => $cab:expr),*) => {
        #[derive(Debug, Clone, Hash)]
        pub struct $Op;
        #[allow(clippy::redundant_closure_call)]
        impl $crate::ops::binary::BinMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }

            fn eval_uniform_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut $typ, &$typ, &$typ) -> () = $cab;
                        let a = a.to_scalar::<$typ>()?;
                        let b = b.as_slice_mut::<$typ>()?;
                        unsafe {
                            for i in 0..b.len() {
                                let mut c = $typ::default();
                                cab(&mut c, a, b.get_unchecked_mut(i));
                                b[i] = c;
                            }
                        }
                        return Ok(())
                    }
                    )*
                 )*

                    $(
                        $(
                            $(if a.datum_type().unquantized() == <$typ_dt>::datum_type().unquantized() {
                                let cab: fn(&mut $typ_dt, &$typ_dt, &$typ_dt, i32, f32) -> () = $cab_dt;
                                let (zp, scale) = a.datum_type().qparams().map(|q| q.zp_scale()).unwrap_or((0, 1.));
                                let a = a.to_scalar::<$typ_dt>()?;
                                let b = b.as_slice_mut::<$typ_dt>()?;
                                unsafe {
                                    for i in 0..b.len() {
                                        let mut c = $typ_dt::default();
                                        cab(&mut c, a, b.get_unchecked_mut(i), zp, scale);
                                        b[i] = c;
                                    }
                                }
                                return Ok(())
                            }
                            )*
                         )*
                     )?
                    bail!("{} does not support {:?} (inplace uniform)", self.name(), a.datum_type());
            }

            fn eval_unicast_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut $typ, &$typ, &$typ) -> () = $cab;
                        let a = a.as_slice::<$typ>()?;
                        let b = b.as_slice_mut::<$typ>()?;
                        unsafe {
                            for i in 0..a.len() {
                                let mut c = $typ::default();
                                cab(&mut c, &a[i], b.get_unchecked(i));
                                *b.get_unchecked_mut(i) = c;
                            }
                        }
                        return Ok(())
                    }
                    )*
                 )*
                    $(
                        $(
                            $(if a.datum_type().unquantized() == <$typ_dt>::datum_type().unquantized() {
                                let cab: fn(&mut $typ_dt, &$typ_dt, &$typ_dt, i32, f32) -> () = $cab_dt;
                                let (zp, scale) = a.datum_type().qparams().map(|q| q.zp_scale()).unwrap_or((0, 1.));
                                let a = a.as_slice::<$typ_dt>()?;
                                let b = b.as_slice_mut::<$typ_dt>()?;
                                unsafe {
                                    for i in 0..a.len() {
                                        let mut c = $typ_dt::default();
                                        cab(&mut c, &a[i], b.get_unchecked(i), zp, scale);
                                        *b.get_unchecked_mut(i) = c;
                                    }
                                }
                                return Ok(())
                            }
                            )*
                         )*
                     )?
                    bail!("{} does not support {:?} (inplace)", self.name(), a.datum_type());
            }

            fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
                $(if $out_of_place(c, a, b)? { return Ok(()) } )?
                    $(
                        $(if c.datum_type() == $typ::datum_type() {
                            let a = a.to_array_view::<$typ>()?;
                            let b = b.to_array_view::<$typ>()?;
                            let mut c = c.to_array_view_mut::<$typ>()?;
                            $crate::ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).for_each($cab);
                            return Ok(())
                        })*
                     )*
                    $(
                        $(
                            $(if a.datum_type().unquantized() == <$typ_dt>::datum_type().unquantized() {
                                let cab: fn(&mut $typ_dt, &$typ_dt, &$typ_dt, i32, f32) -> () = $cab_dt;
                                let (zp, scale) = a.datum_type().qparams().map(|q| q.zp_scale()).unwrap_or((0, 1.));
                                let a = a.to_array_view::<$typ_dt>()?;
                                let b = b.to_array_view::<$typ_dt>()?;
                                let mut c = c.to_array_view_mut::<$typ_dt>()?;
                                $crate::ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).for_each(|c, a, b| cab(c, a, b, zp, scale));
                                return Ok(())
                            }
                            )*
                         )*
                     )?
                    bail!("{} does not support {:?} (out of place)", self.name(), c.datum_type());
            }

            fn eval_in_a(&self, a: &mut Tensor, b: &Tensor) -> TractResult<()> {
                // c and a are same type
                $(
                    $(if b.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut $typ, &$typ, &$typ) -> () = $cab;
                        let b = b.to_array_view::<$typ>()?;
                        let mut a = a.to_array_view_mut::<$typ>()?;
                        $crate::ndarray::Zip::from(&mut a).and_broadcast(b).for_each(|a, b| cab(a, &a.clone(), b));
                        return Ok(())
                    })*
                )*
                $(
                    $(
                        $(if a.datum_type().unquantized() == <$typ_dt>::datum_type().unquantized() {
                            let cab: fn(&mut $typ_dt, &$typ_dt, &$typ_dt, i32, f32) -> () = $cab_dt;
                            let (zp, scale) = a.datum_type().qparams().map(|q| q.zp_scale()).unwrap_or((0, 1.));
                            let mut a = a.to_array_view_mut::<$typ_dt>()?;
                            let b = b.to_array_view::<$typ_dt>()?;
                            $crate::ndarray::Zip::from(&mut a).and_broadcast(b).for_each(|a, b| {
                                cab(a, &(a.clone()), b, zp, scale)
                            });
                            return Ok(())
                        })*
                    )*
                )?
                bail!("{} does not support {:?} (eval in a)", self.name(), a.datum_type());
            }

            $(fn eval(&self, a: TValue, b: TValue) -> TractResult<Tensor> {
                $eval_override(a, b)
            })?

            fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                if a.unquantized() == b.unquantized() {
                    if a.is_quantized() || !b.is_quantized() {
                        return Ok(a)
                    }
                    else {
                        return Ok(b)
                    }
                }
                self.operating_datum_type(a, b)
            }

                $(
                    fn declutter(
                        &self,
                        model: &TypedModel,
                        node: &TypedNode,
                        ) -> TractResult<Option<TypedModelPatch>> {
                        ($declutter)(self, model, node)
                    }
                 )?
                $(
                    fn codegen(
                        &self,
                        model: &TypedModel,
                        node: &TypedNode,
                        a: &Arc<Tensor>,
                        ) -> TractResult<Option<TypedModelPatch>> {
                        ($codegen)(self, model, node, a)
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
                $(
                    fn as_linalg_binop(&self) -> Option<tract_linalg::mmm::BinOp> {
                        Some(tract_linalg::mmm::BinOp::$linalg)
                    }
                 )?
                $(
                    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                        ($operating_datum_type)(a, b)
                    })?
        }

        pub fn $func() -> $crate::ops::binary::TypedBinOp {
            $crate::ops::binary::TypedBinOp(Box::new($Op))
        }
    };
}

macro_rules! bin_to_bool {
    ($func:ident, $Op:ident,
     $( codegen: $codegen:expr, )?
     $( cost: $cost:expr, )?
     $( declutter: $declutter:expr, )?
     $( operating_datum_type: $operating_datum_type:expr, )?
     $( [$($typ:ident),*] => $cab:expr),*) => {
        #[derive(Debug, Clone, Hash)]
        pub struct $Op;
        impl $crate::ops::binary::BinMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }

            fn eval_uniform_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut bool, &bool, &bool) -> () = $cab;
                        let a = a.to_scalar::<bool>()?;
                        let b = b.as_slice_mut::<bool>()?;
                        unsafe {
                            for i in 0..b.len() {
                                let mut c = bool::default();
                                cab(&mut c, a, b.get_unchecked(i));
                                *b.get_unchecked_mut(i) = c;
                            }
                        }
                        return Ok(())
                    }
                    )*
                 )*
                    bail!("{} does not support {:?} (inplace uniform)", self.name(), a.datum_type());
            }

            #[allow(unreachable_code)]
            fn eval_unicast_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut bool, &bool, &bool) -> () = $cab;
                        let a = a.as_slice::<bool>()?;
                        let b = b.as_slice_mut::<bool>()?;
                        unsafe {
                            for i in 0..a.len() {
                                let mut c = bool::default();
                                cab(&mut c, a.get_unchecked(i), b.get_unchecked(i));
                                *b.get_unchecked_mut(i) = c;
                            }
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
                        let cab: fn(&mut bool, &$typ, &$typ) -> () = $cab;
                        let a = a.to_array_view::<$typ>()?;
                        let b = b.to_array_view::<$typ>()?;
                        let mut c = c.to_array_view_mut::<bool>()?;
                        ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).for_each(cab);
                        return Ok(())
                    }
                    )*
                 )*
                    bail!("{} does not support {:?}", self.name(), a.datum_type());
            }

            fn eval_in_a(&self, a: &mut Tensor, _b: &Tensor) -> TractResult<()> {
                bail!("{} does not support {:?}", self.name(), a.datum_type());
            }

            fn result_datum_type(&self, _a: DatumType, _b: DatumType) -> TractResult<DatumType> {
                Ok(bool::datum_type())
            }

            $(
                fn codegen(
                    &self,
                    model: &TypedModel,
                    node: &TypedNode,
                    ) -> TractResult<Option<TypedModelPatch>> {
                    ($codegen)(self, model, node)
                }
             )?


                $(
                    fn declutter(
                        &self,
                        model: &TypedModel,
                        node: &TypedNode,
                        ) -> TractResult<Option<TypedModelPatch>> {
                        ($declutter)(self, model, node)
                    }
                 )?

                $(
                    fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
                        ($cost)(dt)
                    }
                 )?

                $(
                    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
                        ($operating_datum_type)(a, b)
                    })?

        }

        pub fn $func() -> $crate::ops::binary::TypedBinOp {
            $crate::ops::binary::TypedBinOp(Box::new($Op))
        }
    };
}

#[derive(Debug)]
pub(crate) struct OneUniformInput {
    pub uni: Arc<Tensor>,
    pub var: OutletId,
    pub left_is_uniform: bool,
}

pub(crate) fn one_input_is_uniform(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<OneUniformInput>> {
    if let &[a, b] = &*model.node_input_facts(node.id)? {
        let uni = if let Some(a) = &a.uniform {
            OneUniformInput { uni: a.clone(), var: node.inputs[1], left_is_uniform: true }
        } else if let Some(b) = &b.uniform {
            OneUniformInput { uni: b.clone(), var: node.inputs[0], left_is_uniform: false }
        } else {
            return Ok(None);
        };
        let var_fact = [a, b][uni.left_is_uniform as usize];
        let uni_fact = [a, b][!uni.left_is_uniform as usize];
        if izip!(var_fact.shape.iter(), uni_fact.shape.iter()).all(|(v, u)| u.is_one() || u == v) {
            return Ok(Some(uni));
        }
    }
    Ok(None)
}
