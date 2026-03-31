use crate::broadcast::multi_broadcast;
use crate::internal::*;
use crate::ndarray::Zip;
use crate::ops::binary::BinMiniOp;

use tract_data::TooEarly;

/// Extract a `TDim::Val(n)` from a scalar constant fact (integer or integer-valued float).
fn scalar_konst_to_tdim(fact: &TypedFact) -> Option<TDim> {
    let konst = fact.konst.as_ref()?;
    if konst.len() != 1 {
        return None;
    }
    if konst.datum_type().is_integer() || konst.datum_type().is::<bool>() {
        konst.cast_to_scalar::<i64>().ok().map(TDim::Val)
    } else if konst.datum_type().is_float() {
        konst.cast_to_scalar::<f64>().ok().and_then(|f| {
            if (f - f.round()).abs() < 1e-6 { Some(TDim::Val(f.round() as i64)) } else { None }
        })
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Comp {
    Eq,
    NE,
    LT,
    GT,
    GTE,
    LTE,
}

impl Comp {
    pub fn to_bin_mini_op(&self) -> Box<dyn BinMiniOp> {
        match self {
            Comp::Eq => Box::new(CompEq),
            Comp::NE => Box::new(CompNE),
            Comp::LT => Box::new(CompLT),
            Comp::GT => Box::new(CompGT),
            Comp::GTE => Box::new(CompGTE),
            Comp::LTE => Box::new(CompLTE),
        }
    }
}

// Helper for eval_out_of_place dispatch
fn eval_comp_oop<T: Datum + PartialOrd>(
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(&T, &T) -> bool,
) -> TractResult<Tensor> {
    let a = a.to_plain_array_view::<T>()?;
    let b = b.to_plain_array_view::<T>()?;
    let shape = multi_broadcast(&[a.shape(), b.shape()])?;
    let mut c = unsafe { Tensor::uninitialized::<bool>(&shape)? };
    let mut c_plain = c.try_as_plain_mut()?;
    let mut view = c_plain.to_array_view_mut::<bool>()?;
    Zip::from(&mut view).and_broadcast(&a).and_broadcast(&b).for_each(|c, a, b| *c = f(a, b));
    Ok(c)
}

// Helper for TDim symbolic eval
fn eval_tdim_symbolic(
    session: &TurnState,
    inputs: &TVec<TValue>,
    prove: impl Fn(&TDim, &TDim) -> TractResult<bool>,
) -> TractResult<Option<TVec<TValue>>> {
    if inputs[0].datum_type() != TDim::datum_type() {
        return Ok(None);
    }
    let mut a = inputs[0].clone().into_tensor();
    let mut b = inputs[1].clone().into_tensor();
    for a in a.try_as_plain_mut()?.as_slice_mut::<TDim>()? {
        *a = a.eval(&session.resolved_symbols);
    }
    for b in b.try_as_plain_mut()?.as_slice_mut::<TDim>()? {
        *b = b.eval(&session.resolved_symbols);
    }
    if let (Ok(a_i64), Ok(b_i64)) = (a.cast_to::<i64>(), b.cast_to::<i64>()) {
        let result = eval_comp_oop::<i64>(&a_i64, &b_i64, |a, b| {
            prove(&(*a).into(), &(*b).into()).unwrap_or(false)
        })?;
        return Ok(Some(tvec!(result.into_tvalue())));
    }
    let a_view = inputs[0].to_plain_array_view::<TDim>()?;
    let b_view = inputs[1].to_plain_array_view::<TDim>()?;
    let shape = multi_broadcast(&[a_view.shape(), b_view.shape()])?;
    let mut c = unsafe { Tensor::uninitialized::<bool>(&shape)? };
    let mut c_plain = c.try_as_plain_mut()?;
    let mut view = c_plain.to_array_view_mut::<bool>()?;
    let a_bc = a_view.broadcast(&*shape).unwrap();
    let b_bc = b_view.broadcast(&*shape).unwrap();
    for ixs in tract_ndarray::indices(&*shape) {
        view[&ixs] = prove(&a_bc[&ixs], &b_bc[&ixs])?;
    }
    Ok(Some(tvec!(c.into_tvalue())))
}

macro_rules! comp_bin_mini_op {
    ($Op:ident, $name:literal, $cmp:tt, $prove_tdim:expr, $uniform_tdim:expr) => {
        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $Op;

        impl BinMiniOp for $Op {
            fn name(&self) -> &'static str {
                $name
            }

            fn result_datum_type(&self, _a: DatumType, _b: DatumType) -> TractResult<DatumType> {
                Ok(bool::datum_type())
            }

            fn eval_in_a(&self, _a: &mut Tensor, _b: &Tensor) -> TractResult<()> {
                bail!("Comparison changes datum type, eval_in_a not supported")
            }

            fn eval_out_of_place(
                &self,
                c: &mut Tensor,
                a: &Tensor,
                b: &Tensor,
            ) -> TractResult<()> {
                let dt = a.datum_type();
                if dt == String::datum_type() {
                    let a = a.to_plain_array_view::<String>()?;
                    let b = b.to_plain_array_view::<String>()?;
                    let mut c_plain = c.try_as_plain_mut()?;
                    let mut view = c_plain.to_array_view_mut::<bool>()?;
                    Zip::from(&mut view).and_broadcast(&a).and_broadcast(&b)
                        .for_each(|c, a, b| *c = a $cmp b);
                    return Ok(());
                }
                fn inner<T: Datum + PartialOrd>(c: &mut Tensor, a: &Tensor, b: &Tensor, f: impl Fn(&T, &T) -> bool) -> TractResult<()> {
                    let a = a.to_plain_array_view::<T>()?;
                    let b = b.to_plain_array_view::<T>()?;
                    let mut c_plain = c.try_as_plain_mut()?;
                    let mut view = c_plain.to_array_view_mut::<bool>()?;
                    Zip::from(&mut view).and_broadcast(&a).and_broadcast(&b)
                        .for_each(|c, a, b| *c = f(a, b));
                    Ok(())
                }
                dispatch_numbers!(inner(dt)(c, a, b, |a: &_, b: &_| a $cmp b))
            }

            fn eval(&self, a: TValue, b: TValue, c_dt: DatumType) -> TractResult<Tensor> {
                let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
                let mut c = unsafe { Tensor::uninitialized_dt(c_dt, &c_shape)? };
                self.eval_out_of_place(&mut c, &a, &b)?;
                Ok(c)
            }

            fn eval_symbolic(
                &self,
                session: &TurnState,
                inputs: TVec<TValue>,
            ) -> TractResult<Option<TVec<TValue>>> {
                eval_tdim_symbolic(session, &inputs, $prove_tdim)
            }

            fn uniform_tdim_comparison(
                &self,
                a: &TDim,
                b: &TDim,
            ) -> Option<TDim> {
                Some(($uniform_tdim)(a, b))
            }
        }
    };
}

fn prove_eq(a: &TDim, b: &TDim) -> TractResult<bool> {
    Ok(a == b)
}

fn prove_ne(a: &TDim, b: &TDim) -> TractResult<bool> {
    Ok(a != b)
}

fn prove_gte(a: &TDim, b: &TDim) -> TractResult<bool> {
    let diff = a.clone() - b;
    if diff.prove_positive_or_zero() {
        Ok(true)
    } else if diff.prove_strict_negative() {
        Ok(false)
    } else {
        bail!(TooEarly::UndeterminedSymbol(diff.to_string()))
    }
}

fn prove_gt(a: &TDim, b: &TDim) -> TractResult<bool> {
    let diff = a.clone() - b;
    if diff.prove_strict_positive() {
        Ok(true)
    } else if diff.prove_negative_or_zero() {
        Ok(false)
    } else {
        bail!(TooEarly::UndeterminedSymbol(diff.to_string()))
    }
}

fn prove_lte(a: &TDim, b: &TDim) -> TractResult<bool> {
    prove_gte(b, a)
}

fn prove_lt(a: &TDim, b: &TDim) -> TractResult<bool> {
    prove_gt(b, a)
}

comp_bin_mini_op!(CompEq, "Eq", ==, prove_eq, |a: &TDim, b: &TDim|
    TDim::Eq(Box::new(a.clone()), Box::new(b.clone())).reduce()
);

comp_bin_mini_op!(CompNE, "NE", !=, prove_ne, |a: &TDim, b: &TDim|
    (TDim::Val(1) - TDim::Eq(Box::new(a.clone()), Box::new(b.clone()))).reduce()
);

comp_bin_mini_op!(CompLT, "LT", <, prove_lt, |a: &TDim, b: &TDim|
    TDim::Ge(Box::new(b.clone()), Box::new((a.clone() + TDim::Val(1)).reduce())).reduce()
);

comp_bin_mini_op!(CompGT, "GT", >, prove_gt, |a: &TDim, b: &TDim|
    TDim::Ge(Box::new((a.clone() + TDim::Val(1)).reduce()), Box::new(b.clone())).reduce()
);

comp_bin_mini_op!(CompLTE, "LTE", <=, prove_lte, |a: &TDim, b: &TDim|
    TDim::Ge(Box::new(b.clone()), Box::new(a.clone())).reduce()
);

comp_bin_mini_op!(CompGTE, "GTE", >=, prove_gte, |a: &TDim, b: &TDim|
    TDim::Ge(Box::new(a.clone()), Box::new(b.clone())).reduce()
);

// Factory functions
pub fn comp_eq() -> Box<dyn BinMiniOp> {
    Box::new(CompEq)
}
pub fn comp_ne() -> Box<dyn BinMiniOp> {
    Box::new(CompNE)
}
pub fn comp_lt() -> Box<dyn BinMiniOp> {
    Box::new(CompLT)
}
pub fn comp_gt() -> Box<dyn BinMiniOp> {
    Box::new(CompGT)
}
pub fn comp_lte() -> Box<dyn BinMiniOp> {
    Box::new(CompLTE)
}
pub fn comp_gte() -> Box<dyn BinMiniOp> {
    Box::new(CompGTE)
}

// Keep old Comp as Op for backward compat during migration
use Comp::*;

impl Op for Comp {
    fn name(&self) -> StaticName {
        match *self {
            Eq => "==",
            NE => "!=",
            LT => "<",
            GT => ">",
            LTE => "<=",
            GTE => ">=",
        }
        .into()
    }
    op_as_typed_op!();
}

impl EvalOp for Comp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let mini = self.to_bin_mini_op();
        if let Some(result) = mini.eval_symbolic(session, inputs.clone())? {
            return Ok(result);
        }
        let c_dt = mini.result_datum_type(inputs[0].datum_type(), inputs[1].datum_type())?;
        Ok(tvec!(mini.eval(inputs[0].clone(), inputs[1].clone(), c_dt)?.into_tvalue()))
    }
}

impl TypedOp for Comp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mini = self.to_bin_mini_op();
        let shape = multi_broadcast(&[&inputs[0].shape, &inputs[1].shape])?;
        let mut fact = bool::datum_type().fact(shape);
<<<<<<< HEAD
        if let (Some(a), Some(b)) = (&inputs[0].uniform_tdim, &inputs[1].uniform_tdim) {
            fact.uniform_tdim = mini.uniform_tdim_comparison(a, b);
        }
        Ok(tvec!(fact))
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
            rule_if!(inputs[0].shape[*rm].is_one());
            rule_if!(inputs[1].shape[*rm].is_one());
            rule_if!(outputs[0].shape[*rm].is_one());
        }
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
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
        Ok(Some(patch.wire_node(prefix, *self, inputs)?))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    as_op!();
}
