use crate::internal::*;
use crate::ndarray::Dimension;
use downcast_rs::Downcast;
use std::fmt::{self, Debug};
use tract_data::itertools::izip;
use tract_itertools::Itertools;
use tract_linalg::{BinOp, LinalgFn};

use super::math::{Add, Max, Min, Mul, Sub};
use super::{cast::cast, math::SubF};

pub trait BinMiniOp: fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast {
    fn name(&self) -> &'static str;
    fn validation(&self) -> Validation {
        Validation::Accurate
    }
    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        a.common_super_type(b).with_context(|| format_err!("No super type for {:?} and {:?}", a, b))
    }
    fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType>;
    fn eval_in_a(&self, a: &mut Tensor, b: &Tensor) -> TractResult<()>;
    fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()>;

    fn is_commutative(&self) -> bool {
        true
    }
    fn neutral_element(&self) -> Option<i64> {
        None
    }

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
    fn as_linalg_binop(&self) -> Option<tract_linalg::BinOp> {
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
                || !inputs[1].shape[*rm].is_one()
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
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        Ok(Some(patch.wire_node(prefix, self.clone(), inputs)?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let (a_dt, b_dt) = if let &[a, b] = &*model.node_input_facts(node.id)? {
            (a.datum_type().unwrap(), b.datum_type().unwrap())
        } else {
            unreachable!("TypedBinOp has two inputs.")
        };
        if let Some(neutral_patch) =
            declutter_neutral(model, node, self.0.as_ref(), self.output_datum_type(a_dt, b_dt)?)?
        {
            return Ok(Some(neutral_patch));
        }
        if let Some(broadcast_patch) =
            declutter_broadcasting_operand_1(model, node, self.0.clone())?
        {
            return Ok(Some(broadcast_patch));
        }
        self.0.declutter(model, node)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(linalg_bin_op) = self.0.as_linalg_binop() {
            let input_facts = model.node_input_facts(node.id)?;
            let must_swap_inputs =
                input_facts.iter().collect_tuple().is_some_and(|(a_fact, b_fact)| {
                    (a_fact.shape.volume() - b_fact.shape.volume()).prove_strict_negative()
                });
            let (operand_1, operand_2) = if must_swap_inputs {
                (input_facts[1], input_facts[0])
            } else {
                (input_facts[0], input_facts[1])
            };

            let (by_scalar_should_be_efficient, unicast_should_be_efficient) =
                find_most_efficient_config(model, node, must_swap_inputs)?;

            // Check if op is quantized
            let c_dt = self.output_datum_type(operand_1.datum_type, operand_2.datum_type)?;
            let op_is_quant = c_dt.is_quantized()
                || operand_1.datum_type.is_quantized()
                || operand_2.datum_type.is_quantized();

            // Check if it can be evaluated in a
            let c_dt = self.output_datum_type(operand_1.datum_type, operand_2.datum_type)?;
            let c_shape = crate::broadcast::multi_broadcast(&[
                operand_1.shape.clone(),
                operand_2.shape.clone(),
            ])?;
            let can_eval_in_a =
                (c_shape.to_vec() == operand_1.shape.to_vec()) && (c_dt == operand_1.datum_type);

            // Swap input if required
            let inputs = if must_swap_inputs {
                let mut swap_input = node.inputs.clone();
                swap_input.swap(0, 1);
                swap_input
            } else {
                node.inputs.clone()
            };
            let actual_linalg_op =
                if must_swap_inputs { linalg_bin_op.flip() } else { linalg_bin_op };
            let actual_core_op = core_op_for_linalg_op(&actual_linalg_op);

            let dt = model.node_input_facts(node.id)?[0].datum_type;
            if by_scalar_should_be_efficient & can_eval_in_a & !op_is_quant {
                let Some(func) = tract_linalg::bin_by_scalar(dt, actual_linalg_op) else {
                    return Ok(None);
                };
                let eval_fn = Arc::from(func);
                return Ok(Some(
                    TypedModelPatch::replace_single_op(
                        model,
                        node,
                        &inputs,
                        OptBinByScalar { binop: actual_core_op, eval_fn },
                    )?
                    .with_context("ByScalar"),
                ));
            }

            if unicast_should_be_efficient & can_eval_in_a & !op_is_quant {
                let Some(func) = tract_linalg::bin_unicast(dt, actual_linalg_op) else {
                    return Ok(None);
                };
                let eval_fn = Arc::from(func);
                return Ok(Some(
                    TypedModelPatch::replace_single_op(
                        model,
                        node,
                        &inputs,
                        OptBinUnicast { binop: actual_core_op, eval_fn },
                    )?
                    .with_context("Unicast"),
                ));
            }
        }

        Ok(None)
    }
    as_op!();
}

fn core_op_for_linalg_op(linalg: &BinOp) -> Box<dyn BinMiniOp> {
    match linalg {
        BinOp::Min => Box::new(Min),
        BinOp::Max => Box::new(Max),
        BinOp::Add => Box::new(Add),
        BinOp::Mul => Box::new(Mul),
        BinOp::Sub => Box::new(Sub),
        BinOp::SubF => Box::new(SubF),
    }
}
fn declutter_broadcasting_operand_1(
    model: &TypedModel,
    node: &TypedNode,
    mini_op: Box<dyn BinMiniOp>,
) -> TractResult<Option<TypedModelPatch>> {
    let (a_shape, b_shape) = if let &[a, b] = &*model.node_input_facts(node.id)? {
        (a.shape.clone(), b.shape.clone())
    } else {
        unreachable!("TypedBinOp has two inputs.")
    };

    let a_num_elements = a_shape.iter().product::<TDim>();
    let b_num_elements = b_shape.iter().product::<TDim>();
    let a_should_be_broadcast = (a_num_elements - b_num_elements).prove_strict_negative();
    if a_should_be_broadcast & mini_op.is_commutative() {
        let mut swap_input = node.inputs.clone();
        swap_input.swap(0, 1);
        return Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &swap_input,
            TypedBinOp(mini_op, None),
        )?));
    }

    Ok(None)
}

fn declutter_neutral(
    model: &TypedModel,
    node: &TypedNode,
    mini_op: &dyn BinMiniOp,
    out_dt: DatumType,
) -> TractResult<Option<TypedModelPatch>> {
    if let Some(uniform) = crate::ops::binary::one_input_is_uniform(model, node)? {
        let is_neutral = mini_op
            .neutral_element()
            .map(|neutral| tensor0(neutral).close_enough(&uniform.uni, false).is_ok())
            .unwrap_or(false);

        // For some operand neural element can be the left one while for other
        // it is not the case (neutral - 1 -> not ok, 1 - neutal -> ok)
        let pos_checked = mini_op.is_commutative() || !uniform.left_is_uniform;

        if is_neutral && pos_checked {
            // Neutral decluttering for quant values is special.
            // - if (fa) (a-az)*as + (fb = 0) (b-bz)*bs = (fc) (c-cz)*cs
            // - then even if fa = fc, quant params needs to be updated (a != c).
            // So it's not a no_op.
            if uniform.uni.datum_type().is_quantized() {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &[node.inputs[0]],
                    cast(out_dt),
                )?));
            // In the non quantized case, it's a no_op.
            } else {
                return Ok(Some(TypedModelPatch::rewire(
                    model,
                    &[uniform.var],
                    &[node.id.into()],
                    &|_, inputs| Ok(inputs.into()),
                )?));
            }
        }
    }
    Ok(None)
}

fn find_most_efficient_config(
    model: &TypedModel,
    node: &TypedNode,
    swap_input: bool,
) -> TractResult<(bool, bool)> {
    if let &[a, b] = &*model.node_input_facts(node.id)? {
        let a_shape = if swap_input { b.shape.clone() } else { a.shape.clone() };
        let b_shape = if swap_input { a.shape.clone() } else { b.shape.clone() };

        let by_scalar_is_possible = OptBinByScalar::check_input_shapes(&a_shape, &b_shape);
        let num_by_scalar_elements = if by_scalar_is_possible {
            a_shape
                .iter()
                .zip(b_shape.iter())
                .rev()
                .take_while(|(_, rev_b_dim)| **rev_b_dim == TDim::Val(1))
                .map(|(rev_a_dim, _)| rev_a_dim)
                .product::<TDim>()
        } else {
            TDim::Val(0)
        };

        let unicast_is_possible = OptBinUnicast::check_input_shapes(&a_shape, &b_shape);
        let num_unicast_elements = if unicast_is_possible {
            a_shape
                .iter()
                .zip(b_shape.iter())
                .rev()
                .take_while(|(a_dim, b_dim)| a_dim == b_dim)
                .map(|(a_dim, _)| a_dim)
                .product::<TDim>()
        } else {
            TDim::Val(0)
        };

        let min_num_elements = 32;
        let by_scalar_should_be_efficient = gt_tdim(num_by_scalar_elements, min_num_elements);
        let unicast_should_be_efficient = gt_tdim(num_unicast_elements, min_num_elements);
        return Ok((by_scalar_should_be_efficient, unicast_should_be_efficient));
    }
    Ok((false, false))
}

pub fn gt_tdim(x: TDim, min_val: i64) -> bool {
    TDim::Val(min_val).mini(x).to_i64().is_ok_and(|v| v == min_val)
}

#[derive(Clone)]
pub struct OptBinByScalar {
    pub binop: Box<dyn BinMiniOp>,
    eval_fn: Arc<LinalgFn>,
}

impl Debug for OptBinByScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("OptBinByScalar").field("binop", &self.binop).finish()
    }
}

impl OptBinByScalar {
    fn check_input_shapes(a_shape: &[TDim], b_shape: &[TDim]) -> bool {
        if a_shape.len() != b_shape.len() {
            return false;
        };

        a_shape
            .iter()
            .zip(b_shape.iter())
            .skip_while(|(a_dim, b_dim)| a_dim == b_dim)
            .all(|(_, b_dim)| *b_dim == 1.to_dim())
    }
}

impl Op for OptBinByScalar {
    fn name(&self) -> Cow<str> {
        format!("Opt{}ByScalar", self.binop.name()).into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<OptBinByScalar>() else { return false };
        self.binop.same_as(&*other.binop)
    }

    op_as_typed_op!();
}

impl EvalOp for OptBinByScalar {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        // Not a requirement as TensorView doesn't require a owned tensor but in reality
        // "a "should be mutable (it's omitted here as Rust compiler advise to remove it)
        let a = a.into_tensor();
        let b_shape = b.shape();

        let first_unary_axis = b_shape
            .iter()
            .enumerate()
            .rev()
            .take_while(|&(_, &dim)| dim == 1)
            .map(|(i, _)| i)
            .last()
            .context("Cannot use by_scalar when no trailing dimensions are unary")?;

        let iterating_shape = &a.shape()[..first_unary_axis];
        if !iterating_shape.is_empty() {
            for it_coords in tract_ndarray::indices(iterating_shape) {
                let mut view = TensorView::at_prefix(&a, it_coords.slice())?;
                let b_view = TensorView::at_prefix(&b, it_coords.slice())?;
                debug_assert_eq!(b_view.shape().iter().product::<usize>(), 1);
                (self.eval_fn)(&mut view, &b_view)?;
            }
        } else {
            let mut view = a.view();
            let b_view = b.view();
            debug_assert_eq!(b_view.shape().iter().product::<usize>(), 1);
            (self.eval_fn)(&mut view, &b_view)?;
        }
        Ok(tvec!(a.into_tvalue()))
    }
}

impl TypedOp for OptBinByScalar {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(Self::check_input_shapes(&inputs[0].shape, &inputs[1].shape));
        let out_dt = self.binop.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        let out_shape = inputs[0].shape.clone();
        Ok(tvec!(out_dt.fact(out_shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().product();
        Ok(self
            .binop
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
    }

    as_op!();
}

#[derive(Clone)]
pub struct OptBinUnicast {
    pub binop: Box<dyn BinMiniOp>,
    eval_fn: Arc<LinalgFn>,
}

impl Debug for OptBinUnicast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("OptBinUnicast").field("binop", &self.binop).finish()
    }
}

impl OptBinUnicast {
    fn check_b_alignement(a_shape: &[TDim], b_shape: &[TDim]) -> bool {
        let num_iterations: TDim = a_shape
            .iter()
            .zip(b_shape.iter())
            .take_while(|(_, b_dim)| **b_dim == 1.to_dim())
            .map(|(a_dim, _)| a_dim)
            .product();

        if num_iterations.is_one() {
            return true;
        }

        let elements_per_iteration: TDim = a_shape
            .iter()
            .zip(b_shape.iter())
            .skip_while(|(_, b_dim)| **b_dim == 1.to_dim())
            .map(|(_, b_dim)| b_dim)
            .product();

        if let Ok(num_element) = elements_per_iteration.to_i64() {
            let required_alignment = vector_size();
            (num_element as usize % required_alignment) == 0
        } else {
            false
        }
    }
    fn check_input_shapes(a_shape: &[TDim], b_shape: &[TDim]) -> bool {
        if a_shape.len() != b_shape.len() {
            return false;
        };

        let unicast_possible = a_shape
            .iter()
            .zip(b_shape.iter())
            .skip_while(|(_, b_dim)| **b_dim == 1.to_dim())
            .all(|(a_dim, b_dim)| a_dim == b_dim);
        let unicast_is_aligned = Self::check_b_alignement(a_shape, b_shape);

        unicast_possible && unicast_is_aligned
    }
}

impl Op for OptBinUnicast {
    fn name(&self) -> Cow<str> {
        format!("Opt{}Unicast", self.binop.name()).into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<OptBinUnicast>() else { return false };
        self.binop.same_as(&*other.binop)
    }
    op_as_typed_op!();
}

impl EvalOp for OptBinUnicast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        // Not a requirement as TensorView doesn't require a owned tensor but in reality
        // "a "should be mutable (it's omitted here as Rust compiler advise to remove it)
        let a = a.into_tensor();
        let b_shape = b.shape();
        let b_view = b.view();
        let first_non_unary_axis =
            b_shape.iter().enumerate().take_while(|&(_, &dim)| dim == 1).map(|(i, _)| i + 1).last();

        if let Some(first_non_unary_axis) = first_non_unary_axis {
            // Iterate on outter dimensions and evaluate with unicast subviews
            let iterating_shape = a.shape()[..first_non_unary_axis].to_vec();
            for it_coords in tract_ndarray::indices(iterating_shape) {
                let mut view = TensorView::at_prefix(&a, it_coords.slice())?;
                debug_assert_eq!(view.shape(), &b_view.shape()[it_coords.slice().len()..]);
                (self.eval_fn)(&mut view, &b_view)?;
            }
        } else {
            let mut view = a.view();
            debug_assert_eq!(view.shape(), b_view.shape());
            (self.eval_fn)(&mut view, &b_view)?;
        }

        Ok(tvec!(a.into_tvalue()))
    }
}

impl TypedOp for OptBinUnicast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(Self::check_input_shapes(&inputs[0].shape, &inputs[1].shape));
        let out_dt = self.binop.result_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        let out_shape = inputs[0].shape.clone();
        Ok(tvec!(out_dt.fact(out_shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = self.output_facts(inputs)?[0].shape.iter().product();
        Ok(self
            .binop
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
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
     $(is_commutative: $is_commutative:expr,)?
     $(neutral_element: $neutral_element:expr,)?
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

            $(fn is_commutative(&self) -> bool {
                $is_commutative
            })?
            $(fn neutral_element(&self) -> Option<i64> {
                Some($neutral_element)
            })?
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
                    fn as_linalg_binop(&self) -> Option<tract_linalg::BinOp> {
                        Some(tract_linalg::BinOp::$linalg)
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
