use crate::broadcast::multi_broadcast;
use crate::internal::*;
use crate::ndarray::Zip;
use crate::ops::binary::BinMiniOp;

use tract_data::TooEarly;

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
    let mut c_plain = c.try_as_plain_ram_mut()?;
    let mut view = c_plain.to_array_view_mut::<bool>()?;
    Zip::from(&mut view).and_broadcast(&a).and_broadcast(&b).for_each(|c, a, b| *c = f(a, b));
    Ok(c)
}

// Helper for TDim symbolic eval
fn eval_tdim_symbolic(
    ctx: &EvalContext,
    inputs: &TVec<TValue>,
    prove: impl Fn(&TDim, &TDim) -> TractResult<bool>,
) -> TractResult<Option<TVec<TValue>>> {
    rule_if!(
        inputs[0].datum_type() == TDim::datum_type()
            || inputs[1].datum_type() == TDim::datum_type()
    );
    // a dynamic Range yields i64 while a TDim konst stays TDim: cast the
    // integer side up so the pair goes through the symbolic path
    let to_tdim = |t: &TValue| -> TractResult<Tensor> {
        let t = t.clone().into_tensor();
        if t.datum_type() == TDim::datum_type() {
            Ok(t)
        } else {
            Ok(t.cast_to_dt(TDim::datum_type())?.into_owned())
        }
    };
    let mut a = to_tdim(&inputs[0])?;
    let mut b = to_tdim(&inputs[1])?;
    for a in a.try_as_plain_ram_mut()?.as_slice_mut::<TDim>()? {
        *a = a.eval(ctx.symbols);
    }
    for b in b.try_as_plain_ram_mut()?.as_slice_mut::<TDim>()? {
        *b = b.eval(ctx.symbols);
    }
    if let (Ok(a_i64), Ok(b_i64)) = (a.cast_to::<i64>(), b.cast_to::<i64>()) {
        let result = eval_comp_oop::<i64>(&a_i64, &b_i64, |a, b| {
            prove(&(*a).into(), &(*b).into()).unwrap_or(false)
        })?;
        return Ok(Some(tvec!(result.into_tvalue())));
    }
    let a_view = a.to_plain_array_view::<TDim>()?;
    let b_view = b.to_plain_array_view::<TDim>()?;
    let shape = multi_broadcast(&[a_view.shape(), b_view.shape()])?;
    let mut c = unsafe { Tensor::uninitialized::<bool>(&shape)? };
    let mut c_plain = c.try_as_plain_ram_mut()?;
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

            fn is_commutative(&self) -> bool {
                false
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
                    let mut c_plain = c.try_as_plain_ram_mut()?;
                    let mut view = c_plain.to_array_view_mut::<bool>()?;
                    Zip::from(&mut view).and_broadcast(&a).and_broadcast(&b)
                        .for_each(|c, a, b| *c = a $cmp b);
                    return Ok(());
                }
                fn inner<T: Datum + PartialOrd>(c: &mut Tensor, a: &Tensor, b: &Tensor, f: impl Fn(&T, &T) -> bool) -> TractResult<()> {
                    let a = a.to_plain_array_view::<T>()?;
                    let b = b.to_plain_array_view::<T>()?;
                    let mut c_plain = c.try_as_plain_ram_mut()?;
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
                ctx: &EvalContext,
                inputs: TVec<TValue>,
            ) -> TractResult<Option<TVec<TValue>>> {
                eval_tdim_symbolic(ctx, &inputs, $prove_tdim)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::*;
    #[allow(unused_imports)]
    // A dynamic Range output (i64) compared against a TDim konst must evaluate
    // through the symbolic path (the i64 side is cast up), not error on the
    // mixed datum types.
    #[test]
    fn mixed_i64_tdim_comparison() -> TractResult<()> {
        use crate::ops::array::Range;
        use crate::ops::binary::TypedBinOp;

        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let definer = model.add_source("definer", f32::datum_type().fact([s.to_dim()]))?;
        let start = model.add_const("start", tensor0(TDim::Val(0)))?;
        let step = model.add_const("step", tensor0(TDim::Val(1)))?;
        let end = model.wire_node(
            "end",
            crate::ops::konst::Const::new(tensor0(TDim::Sym(s.clone())).into_arc_tensor())?,
            &[],
        )?;
        let range = model.wire_node("range", Range::new(s.to_dim()), &[start, end[0], step])?;
        let t_const = model.wire_node(
            "t_const",
            crate::ops::konst::Const::new(tensor0(TDim::Sym(s.clone())).into_arc_tensor())?,
            &[],
        )?;
        let t_unsq =
            model.wire_node("t_unsq", crate::ops::change_axes::AxisOp::Add(0), &[t_const[0]])?;
        let lt = model.wire_node("lt", TypedBinOp(comp_lt(), None), &[range[0], t_unsq[0]])?;
        model.select_output_outlets(&[lt[0], definer])?;
        let found = crate::internal::TypedSimplePlan::new(model)?
            .run(tvec!(tensor1(&[0f32; 3]).into_tvalue()))?;
        // all of range(0..S) is strictly below S: true, true, true
        assert_eq!(*found[0], tensor1(&[true, true, true]));
        Ok(())
    }
}
