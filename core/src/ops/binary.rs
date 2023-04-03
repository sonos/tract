use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;
use tract_data::itertools::izip;

pub fn wire_rank_broadcast(
    prefix: &str,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let facts = inputs
        .iter()
        .map(|o| target.outlet_fact(*o).map(|ok| ok.clone()))
        .collect::<TractResult<TVec<_>>>()?;
    let max_rank = facts.iter().map(|f| f.rank()).max().unwrap();
    let mut wires = tvec!();
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

#[deprecated]
pub fn wire_with_rank_broadcast(
    prefix: impl ToString,
    target: &mut TypedModel,
    op: impl Into<Box<dyn TypedOp>>,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let prefix = prefix.to_string();
    let wires = wire_rank_broadcast(&prefix, target, inputs)?;
    target.wire_node(prefix, &op.into(), &wires)
}

pub fn wire_bin(
    name: impl ToString,
    target: &mut TypedModel,
    op: impl Into<Box<dyn BinMiniOp>>,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    ensure!(inputs.len() == 2);
    let prefix = name.to_string();
    let rank_a = target.outlet_fact(inputs[0])?.rank();
    let rank_b = target.outlet_fact(inputs[1])?.rank();
    let rank_c = rank_a.max(rank_b);
    let mut axes = AxesMapping::natural_for_rank(2, 1, rank_c)?;
    for _ in rank_a..rank_c {
        axes = axes.remove_input_axis(0, 0)?;
    }
    for _ in rank_b..rank_c {
        axes = axes.remove_input_axis(1, 0)?;
    }
    let op = TypedBinOp { op: op.into(), axes, codegen: None };
    target.wire_node(prefix, &op.into(), &inputs)
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
    fn eval_unicast_in_right(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()>;
    fn eval_uniform_in_right(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()>;
    fn eval_in_a(&self, axes: &AxesMapping, a: &mut Tensor, b: &Tensor) -> TractResult<()>;
    fn eval_out_of_place(
        &self,
        axes: &AxesMapping,
        c: &mut Tensor,
        a: &Tensor,
        b: &Tensor,
    ) -> TractResult<()>;
    fn generic_eval(&self, axes: &AxesMapping, a: TValue, b: TValue) -> TractResult<Tensor> {
        let c_dt = self.result_datum_type(a.datum_type(), b.datum_type())?;
        let c_shape = output_shape(axes, a.shape(), b.shape())?;
        /*
        if c_dt == b.datum_type() && a.len() == 1 && axes.direct(InOut::In(1), InOut::Out(0)) {
        let mut b = b.into_tensor();
        self.eval_uniform_in_place(&a, &mut b)?;
        Ok(b)
        } else if a.shape() == b.shape()
        && c_dt == b.datum_type()
        && axes.direct(InOut::In(0), InOut::In(1))
        && axes.direct(InOut::In(0), InOut::Out(0))
        {
        let mut b = b.into_tensor();
        self.eval_unicast_in_place(&a, &mut b)?;
        Ok(b)
        } else if &*c_shape == a.shape()
        && axes.direct(InOut::In(0), InOut::Out(0))
        && c_dt == a.datum_type()
        {
        let mut a = a.into_tensor();
        self.eval_in_a(axes, &mut a, &b)?;
        Ok(a)
        } else {
        }
        */
        let mut c = unsafe { Tensor::uninitialized_dt(c_dt, &c_shape)? };
        self.eval_out_of_place(axes, &mut c, &a, &b)?;
        Ok(c)
    }
    fn eval(&self, axes: &AxesMapping, a: TValue, b: TValue) -> TractResult<Tensor> {
        self.generic_eval(axes, a, b)
    }
    #[allow(unused_variables)]
    fn declutter(
        &self,
        axes: &AxesMapping,
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

impl<B: BinMiniOp> From<B> for Box<dyn BinMiniOp + 'static> {
    fn from(value: B) -> Self {
        Box::new(value)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum BinOpCodegen {
    #[default]
    Generic,
    UnicastInRight(Vec<AxisOp>),
    UniformInRight(Vec<AxisOp>),
}

#[derive(Debug, Clone)]
pub struct TypedBinOp {
    pub op: Box<dyn BinMiniOp>,
    pub axes: AxesMapping,
    pub codegen: Option<BinOpCodegen>,
}

impl Op for TypedBinOp {
    fn name(&self) -> Cow<str> {
        self.op.name().into()
    }

    fn validation(&self) -> Validation {
        self.op.validation()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let i = if let Some(codegen) = &self.codegen {
            format!("{} ({codegen:?})", self.axes)
        } else {
            self.axes.to_string()
        };
        Ok(vec![i])
    }

    op_as_typed_op!();
}

impl EvalOp for TypedBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        match &self.codegen {
            Some(BinOpCodegen::UnicastInRight(fixes)) => {
                let mut b = b.into_tensor();
                self.op.eval_unicast_in_right(&a, &mut b)?;
                for axis_fix in fixes {
                    axis_fix.change_tensor(&mut b, false)?;
                }
                return Ok(tvec!(b.into_tvalue()));
            }
            Some(BinOpCodegen::UniformInRight(fixes)) => {
                let mut b = b.into_tensor();
                self.op.eval_uniform_in_right(&a, &mut b)?;
                for axis_fix in fixes {
                    axis_fix.change_tensor(&mut b, false)?;
                }
                return Ok(tvec!(b.into_tvalue()));
            }
            _ => (),
        };
        let c = self.op.eval(&self.axes, a, b)?;
        Ok(tvec!(c.into_tvalue()))
    }
}

impl TypedOp for TypedBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != self.axes.input_rank(0)
            || inputs[1].rank() != self.axes.input_rank(1)
        {
            bail!("Invalid binary wiring {} : inputs are {:?}", self.axes, inputs);
        }
        let dt = self.op.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        let shape = output_shape(&self.axes, &inputs[0].shape, &inputs[1].shape)?;
        Ok(tvec!(dt.fact(shape)))
    }

    #[allow(unused_variables)]
    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if self.codegen.is_some() {
            return Ok(None);
        }
        let Some(axes) = self.axes.change_axis_sink(io, change)? else { return Ok(None) };
        Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(Self { axes, ..self.clone() })),
            wire_changes: tvec!((io, change.clone())),
        }))
    }

    fn axes_mapping(
        &self,
        _inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        Ok(self.axes.clone())
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().product();
        Ok(self
            .op
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
        if self.codegen.is_some() {
            return Ok(None);
        }
        Ok(Some(patch.wire_node(prefix, self.clone(), inputs)?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.codegen.is_some() {
            return Ok(None);
        }
        self.op.declutter(&self.axes, model, node).context("binary op declutter")
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.codegen.is_some() {
            return Ok(None);
        }
        let facts = model.node_input_facts(node.id)?;
        let output_shape = output_shape(&self.axes, &facts[0].shape, &facts[1].shape)?;
        let cdt = self.op.result_datum_type(facts[0].datum_type, facts[1].datum_type)?;
        let codegen = if cdt == facts[0].datum_type
            && self.axes.same_layout(InOut::In(0), InOut::In(1), &facts[0].shape, &facts[1].shape)
            && self.axes.same_layout(InOut::In(0), InOut::Out(0), &facts[0].shape, &output_shape)
        {
            BinOpCodegen::UnicastInRight(
                self.axes.extract_sub_mapping(&[1], &[0])?.translate_to_axis_ops()?,
            )
        } else if facts[0].shape.volume().is_one() && cdt == facts[1].datum_type {
            BinOpCodegen::UniformInRight(
                self.axes.extract_sub_mapping(&[1], &[0])?.translate_to_axis_ops()?,
            )
                /*
        } else if facts[1].shape.volume().is_one() && cdt == facts[0].datum_type {
            BinOpCodegen::UniformInPlace {
                mutate: 0,
                fixes: self.axes.extract_sub_mapping(&[0], &[0])?.translate_to_axis_ops()?,
            }
            */
        } else {
            BinOpCodegen::Generic
        };
        let ctx = format!("{codegen:?}");
        Ok(Some(
            TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                Self { codegen: Some(codegen), ..self.clone() },
            )?
            .with_context(ctx),
        ))
    }

    as_op!();
}

pub fn output_shape<D: DimLike>(
    axes: &AxesMapping,
    a: impl AsRef<[D]>,
    b: impl AsRef<[D]>,
) -> TractResult<TVec<D>> {
    let a = a.as_ref();
    let b = b.as_ref();
    let mut shape = tvec!();
    for axis in axes.output_axes(0) {
        if axis.inputs[0].len() > 1 || axis.inputs[1].len() > 1 {
            bail!("Invalid expression for binary mapping {}", axes);
        }
        let one = D::one();
        let dim_a = axis.inputs[0].get(0).map(|ax| &a[*ax]).unwrap_or(&one);
        let dim_b = axis.inputs[1].get(0).map(|ax| &b[*ax]).unwrap_or(&one);
        let dim_c = if dim_a == &one {
            dim_b.clone()
        } else if dim_b == &one {
            dim_a.clone()
        } else if dim_a == dim_b {
            dim_a.clone()
        } else {
            bail!(
                "Can not compute dim for axis {} with inputs {:?} and {:?} and expression {}",
                axis.repr,
                a,
                b,
                axes
            );
        };
        shape.push(dim_c);
    }
    Ok(shape)
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

            fn eval_uniform_in_right(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
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

            fn eval_unicast_in_right(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
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

            fn eval_out_of_place(&self, axes: &AxesMapping, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
                $(if $out_of_place(axes, c, a, b)? { return Ok(()) } )?
                    $(
                        $(if c.datum_type() == $typ::datum_type() {
                            return crate::ops::binary::eval_out_of_place::<$typ, $typ, $typ>(axes, c, a, b, $cab)
                        })*
                     )*
                    $(
                        $(
                            $(if a.datum_type().unquantized() == <$typ_dt>::datum_type().unquantized() {
                                let cab: fn(&mut $typ_dt, &$typ_dt, &$typ_dt, i32, f32) -> () = $cab_dt;
                                let (zp, scale) = a.datum_type().qparams().map(|q| q.zp_scale()).unwrap_or((0, 1.));
                                return crate::ops::binary::eval_out_of_place::<$typ_dt, $typ_dt, $typ_dt>(axes, c, a, b, |c, a, b| cab(c,a,b,zp, scale) )
                            }
                            )*
                         )*
                     )?
                    bail!("{} does not support {:?} (out of place)", self.name(), c.datum_type());
            }

            fn eval_in_a(&self, axes: &AxesMapping, a: &mut Tensor, b: &Tensor) -> TractResult<()> {
                // c and a are same type
                $(
                    $(if b.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut $typ, &$typ, &$typ) -> () = $cab;
                        return crate::ops::binary::eval_in_a::<$typ, $typ>(axes, a, b, cab);
                    })*
                 )*
                    $(
                        $(
                            $(if a.datum_type().unquantized() == <$typ_dt>::datum_type().unquantized() {
                                let cab: fn(&mut $typ_dt, &$typ_dt, &$typ_dt, i32, f32) -> () = $cab_dt;
                                let (zp, scale) = a.datum_type().qparams().map(|q| q.zp_scale()).unwrap_or((0, 1.));
                                return crate::ops::binary::eval_in_a::<$typ_dt, $typ_dt>(axes, a, b,
                                                                                         |c,a,b| cab(c, &(a.clone()), b, zp, scale));
                            })*
                         )*
                     )?
                    bail!("{} does not support {:?} (eval in a)", self.name(), a.datum_type());
            }

            $(fn eval(&self, axes:&AxesMapping, a: TValue, b: TValue) -> TractResult<Tensor> {
                $eval_override(axes, a, b)
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
                    axes: &AxesMapping,
                    model: &TypedModel,
                    node: &TypedNode,
                    ) -> TractResult<Option<TypedModelPatch>> {
                    ($declutter)(self, axes, model, node)
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

            fn eval_uniform_in_right(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
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
            fn eval_unicast_in_right(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
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

            fn eval_out_of_place(&self, axes:&AxesMapping, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
                $(
                    $(if a.datum_type() == $typ::datum_type() {
                        let cab: fn(&mut bool, &$typ, &$typ) -> () = $cab;
                        return crate::ops::binary::eval_out_of_place::<bool, $typ, $typ>(axes, c, a, b, cab)
                    }
                    )*
                 )*
                    bail!("{} does not support {:?}", self.name(), a.datum_type());
            }

            fn eval_in_a(&self, _axes: &AxesMapping, a: &mut Tensor, _b: &Tensor) -> TractResult<()> {
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
                        axes: &AxesMapping,
                        model: &TypedModel,
                        node: &TypedNode,
                        ) -> TractResult<Option<TypedModelPatch>> {
                        ($declutter)(self, axes, model, node)
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

pub fn eval_out_of_place<C: Datum, A: Datum, B: Datum>(
    axes: &AxesMapping,
    c: &mut Tensor,
    a: &Tensor,
    b: &Tensor,
    cab: impl FnMut(&mut C, &A, &B),
) -> TractResult<()> {
    let mut a = a.to_array_view::<A>()?;
    let mut b = b.to_array_view::<B>()?;
    let mut c = c.to_array_view_mut::<C>()?;
    axes.view_to_canonical(InOut::In(0), &mut a)?;
    axes.view_to_canonical(InOut::In(1), &mut b)?;
    axes.view_to_canonical_mut(InOut::Out(0), &mut c)?;
    tract_ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).for_each(cab);
    Ok(())
}

pub fn eval_in_a<A: Datum, B: Datum>(
    axes: &AxesMapping,
    a: &mut Tensor,
    b: &Tensor,
    mut cab: impl FnMut(&mut A, &A, &B),
) -> TractResult<()> {
    let mut a = a.to_array_view_mut::<A>()?;
    let mut b = b.to_array_view::<B>()?;
    axes.view_to_canonical_mut(InOut::In(0), &mut a)?;
    axes.view_to_canonical(InOut::In(1), &mut b)?;
    tract_ndarray::Zip::from(&mut a).and_broadcast(b).for_each(|a, b| cab(a, &a.clone(), b));
    Ok(())
}
