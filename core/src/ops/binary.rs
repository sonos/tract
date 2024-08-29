use crate::internal::*;
use downcast_rs::Downcast;
use tract_itertools::Itertools;
use std::fmt;
use tract_data::itertools::izip;

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

    #[allow(unused_variables)]
    fn maybe_eval_qbinary_as_float_op(
        &self,
        a: &TValue,
        b: &TValue,
        c_dt: &DatumType,
    ) -> TractResult<Option<Tensor>> {
        Ok(None)
    }

    fn generic_eval(&self, a: TValue, b: TValue, c_dt: DatumType) -> TractResult<Tensor> {
        if let Some(tensor) = self.maybe_eval_qbinary_as_float_op(&a, &b, &c_dt)? {
            Ok(tensor)
        } else if c_dt == b.datum_type() && a.len() == 1 {
            let mut b = b.into_tensor();
            self.eval_uniform_in_place(&a, &mut b)?;
            Ok(b)
        } else if a.shape() == b.shape() && c_dt == b.datum_type() {
            let mut b = b.into_tensor();
            self.eval_unicast_in_place(&a, &mut b)?;
            Ok(b)
        } else {
            let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
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
    fn eval(&self, a: TValue, b: TValue, c_dt: DatumType) -> TractResult<Tensor> {
        self.generic_eval(a, b, c_dt)
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

    #[allow(unused_variables)]
    fn same_as(&self, other: &dyn BinMiniOp) -> bool {
        false
    }
}
dyn_clone::clone_trait_object!(BinMiniOp);
downcast_rs::impl_downcast!(BinMiniOp);

#[derive(Debug, Clone)]
pub struct TypedBinOp(pub Box<dyn BinMiniOp>, pub Option<DatumType>);

impl Op for TypedBinOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<TypedBinOp>() else { return false };
        self.1 == other.1 && self.0.same_as(&*other.0)
    }

    op_as_typed_op!();
}

impl TypedBinOp {
    fn output_datum_type(&self, a_dt: DatumType, b_dt: DatumType) -> TractResult<DatumType> {
        if let Some(dt) = self.1 {
            Ok(dt)
        } else {
            self.0.result_datum_type(a_dt, b_dt)
        }
    }
}

impl EvalOp for TypedBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        ensure!(a.rank() == b.rank());
        let c_dt = self.output_datum_type(a.datum_type(), b.datum_type())?;
        Ok(tvec!(self.0.eval(a, b, c_dt)?.into_tvalue()))
    }
}

impl TypedOp for TypedBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!(
                "Typed ops require rank match. Invalid inputs for {}: {}",
                self.name(),
                inputs.iter().map(|s| format!("{s:?}")).join(" ; ")
            );
        }
        let out_dt = self.output_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        Ok(tvec!(out_dt.fact(&*crate::broadcast::multi_broadcast(&[
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec()
        ])?)))
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
        _model: &TypedModel,
        _node: &TypedNode,
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
        if self.output_datum_type(facts[0].datum_type, facts[1].datum_type)? == facts[0].datum_type
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
     $(eval_in_a: $eval_in_a:expr,)?
     $(eval_override: $eval_override: expr,)?
     $(linalg: $linalg:ident,)?
     $(operating_datum_type: $operating_datum_type:expr,)?
     $(uniform_in_place: $uniform_in_place:expr,)?
     $(unicast_in_place: $unicast_in_place:expr,)?
     $(out_of_place: $out_of_place:expr,)?
     $(validation: $validation:expr,)?
     $(q: $([$($typ_dt:ident),*] => $cab_dt:expr),* ;)?
     $(q_op_on_f32: $q_op_on_f32:expr,)?
     $( [$($typ:ident),*] => $cab:expr),*) => {
        #[derive(Debug, Clone, Hash)]
        pub struct $Op;
        #[allow(clippy::redundant_closure_call)]
        impl $crate::ops::binary::BinMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }

            fn same_as(&self, other: &dyn $crate::ops::binary::BinMiniOp) -> bool {
                other.downcast_ref::<$Op>().is_some()
            }

            fn eval_uniform_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
                $(if $uniform_in_place(a, b)? { return Ok(()) } )?
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut $typ, &$typ, &$typ) -> () = $cab;
                        let a = &a.as_slice::<$typ>()?[0];
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
                                let a = &a.as_slice::<$typ_dt>()?[0];
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
                $(if $unicast_in_place(a, b)? { return Ok(()) } )?
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
                $(if $eval_in_a(a, b)? { return Ok(()) } )?
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

            $(fn eval(&self, a: TValue, b: TValue, c_dt: DatumType) -> TractResult<Tensor> {
                $eval_override(a, b, c_dt)
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


            /// Default simple binary operation for QFormat where
            /// we dequantise & apply requested operation in float & requantize it
            /// several implementation are provided with pro & con
            #[allow(unused_variables)]
            fn maybe_eval_qbinary_as_float_op(
                &self,
                a: &TValue,
                b: &TValue,
                c_dt: &DatumType,
            ) -> TractResult<Option<Tensor>> {
                $(
                    /// Implementation strive to minimise memory allocation and access
                    /// we apply only if type is QU8 zp_scale datum type
                    /// maybe more suited for large models tensors
                    fn memory_optimised_q_binary_as_float_op(
                        a: &TValue,
                        b: &TValue,
                        c_dt: &DatumType,
                    ) -> TractResult<Option<Tensor>> {
                        if let (DatumType::QU8(QParams::ZpScale {zero_point: a_zp, scale: a_scale}),
                                DatumType::QU8(QParams::ZpScale {zero_point: b_zp, scale: b_scale}),
                                DatumType::QU8(QParams::ZpScale {zero_point: c_zp, scale: c_scale})) =
                            (a.datum_type(), b.datum_type(), c_dt)
                        {
                            let c_inv_scale = 1.0 / c_scale;
                            let a = a.to_array_view::<u8>()?;
                            let b = b.to_array_view::<u8>()?;
                            let c_shape = $crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
                            let mut c = Tensor::zero_dt(*c_dt, &c_shape)?;
                            let view = c.to_array_view_mut::<u8>()?;
                            $crate::ndarray::Zip::from(view).and_broadcast(a).and_broadcast(b).for_each(|c, a, b| {
                                *c = (scale_by($q_op_on_f32(
                                            ((*a as i32 - a_zp as i32) as f32 * a_scale),
                                            ((*b as i32 - b_zp as i32) as f32 * b_scale),
                                ), c_inv_scale) as i32
                                    + *c_zp as i32)
                                    .clamp_cast()
                            });
                            return Ok(Some(c));
                        }
                        Ok(None)
                    }

                    /// Apply to all Q types
                    /// Take more memory but hopefully faster than memory_optimised_q_binary_as_float_op
                    /// especially once cast_to_dt will have will have vectorized implementations
                    fn generic_q_binary_as_float_op(
                        a: &TValue,
                        b: &TValue,
                        c_dt: &DatumType,
                        accumulator_dt: DatumType
                    ) -> TractResult<Option<Tensor>> {
                        if a.datum_type().is_quantized() && b.datum_type().is_quantized() && c_dt.is_quantized() {
                            let a = a.cast_to_dt(accumulator_dt)?.into_owned();
                            let b = b.cast_to_dt(accumulator_dt)?.into_owned();
                            let c_shape = $crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
                            let mut c = Tensor::zero_dt(accumulator_dt, &c_shape)?;
                            match accumulator_dt {
                                DatumType::F32 => {
                                    let view = c.to_array_view_mut::<f32>()?;
                                    $crate::ndarray::Zip::from(view).and_broadcast(a.to_array_view()?).and_broadcast(b.to_array_view()?).for_each(|c, a, b| {
                                        *c = $q_op_on_f32(*a,*b);
                                    })
                                },
                                other => bail!("unexpected accumulator data type as {:?}", other)
                            };

                            return Ok(Some(c.cast_to_dt(*c_dt)?.into_owned()));
                        }
                        Ok(None)
                    }

                    if let Some(c) = memory_optimised_q_binary_as_float_op(a, b, c_dt)? {
                        return Ok(Some(c));
                    }
                    if let Some(d) = generic_q_binary_as_float_op(a, b, c_dt, DatumType::F32)? {
                        return Ok(Some(d));
                    }
                )?
                Ok(None)
            }
        }

        pub fn $func() -> $crate::ops::binary::TypedBinOp {
            $crate::ops::binary::TypedBinOp(Box::new($Op), None)
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
