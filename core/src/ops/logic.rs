#![allow(clippy::bool_comparison)]
#![allow(clippy::unnecessary_cast)]

mod comparison;
mod ite;
pub use comparison::Comp;
pub use ite::IfThenElse;

use ndarray::*;

use crate::broadcast::multi_broadcast;
use crate::internal::*;

bin_to_super_type!(and, And,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 && b as i64 != 0) as _);
bin_to_super_type!(or, Or,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 || b as i64 != 0) as _);
bin_to_super_type!(xor, Xor, /*flip: commute, */ [bool] => |c, &a, &b| *c = a ^ b);

element_wise!(not, Not, [bool] => |_, vs| {
    vs.iter_mut().for_each(|a| *a = !*a);
    Ok(())
});

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct Iff;

impl Iff {
    pub unsafe fn eval_t<T: Datum>(
        cond: &ArrayViewD<bool>,
        out: &mut Tensor,
        t: &Tensor,
        f: &Tensor,
    ) {
        unsafe {
            Zip::from(out.to_array_view_mut_unchecked::<T>())
                .and_broadcast(cond)
                .and_broadcast(t.to_array_view_unchecked::<T>())
                .and_broadcast(f.to_array_view_unchecked::<T>())
                .for_each(|r, c, t, f| *r = if *c { t.clone() } else { f.clone() })
        }
    }
}

impl Op for Iff {
    fn name(&self) -> StaticName {
        "Iff".into()
    }
    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for Iff {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (cond, t, f) = args_3!(inputs);
        anyhow::ensure!(t.datum_type() == f.datum_type());
        let shape: TVec<usize> = multi_broadcast(&[cond.shape(), t.shape(), f.shape()])?;
        unsafe {
            let mut result = Tensor::uninitialized_dt(t.datum_type(), &shape)?;
            let cond = cond.to_dense_array_view::<bool>()?;
            dispatch_datum_by_size!(Self::eval_t(t.datum_type())(&cond, &mut result, &t, &f));
            Ok(tvec!(result.into_tvalue()))
        }
    }
}

fn coord_bound_assertions(expr: &TDim, shape: &ShapeFact) -> Vec<Assertion> {
    expr.symbols()
        .into_iter()
        .filter_map(|s| {
            let name = format!("{s}");
            name.strip_prefix('x')
                .and_then(|rest| rest.parse::<usize>().ok())
                .filter(|k| *k < shape.rank())
                .map(|k| (k, s))
        })
        .flat_map(|(k, sym)| {
            [
                Assertion::GTE(TDim::Sym(sym.clone()), TDim::Val(0)),
                Assertion::LTE(TDim::Sym(sym), shape[k].clone() - TDim::Val(1)),
            ]
        })
        .collect()
}

fn is_provably_all_false(expr: &TDim, shape: &ShapeFact) -> bool {
    let extra = coord_bound_assertions(expr, shape);
    expr.clone().simplify_with_extra_assertions(&extra) == TDim::Val(0)
}

fn is_provably_all_true(expr: &TDim, shape: &ShapeFact) -> bool {
    let extra = coord_bound_assertions(expr, shape);
    expr.clone().simplify_with_extra_assertions(&extra) == TDim::Val(1)
}

impl TypedOp for Iff {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "Iff expects 3 intputs.");
        ensure!(inputs[1].datum_type == inputs[2].datum_type);
        ensure!(inputs[0].datum_type.is::<bool>());
        ensure!(inputs[0].rank() == inputs[1].rank());
        ensure!(inputs[0].rank() == inputs[2].rank());
        let shape = multi_broadcast(&[
            inputs[0].shape.to_tvec(),
            inputs[1].shape.to_tvec(),
            inputs[2].shape.to_tvec(),
        ])
        .unwrap();
        let mut fact = inputs[1].datum_type.fact(shape);
        // Propagate uniform_tdim when condition is provably constant
        fact.uniform_tdim = match inputs[0].uniform_tdim.as_ref().map(|d| d.clone().simplify()) {
            Some(TDim::Val(0)) => inputs[2].uniform_tdim.clone(), // always false → false branch
            Some(TDim::Val(_)) => inputs[1].uniform_tdim.clone(), // always true → true branch
            _ => None,
        };
        Ok(tvec!(fact))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let cond_outlet = node.inputs[0];
        let cond_fact = model.outlet_fact(cond_outlet)?;

        // Get uniform_tdim from condition fact, or trace through BitNot one level.
        let mut cond_tdim = cond_fact.uniform_tdim.clone();
        if cond_tdim.is_none() {
            let cond_node = model.node(cond_outlet.node);
            if let Some(ew) = cond_node.op_as::<crate::ops::element_wise::ElementWiseOp>() {
                if ew.0.downcast_ref::<BitNot>().is_some() && cond_node.inputs.len() == 1 {
                    let inner_fact = model.outlet_fact(cond_node.inputs[0])?;
                    if let Some(inner_tdim) = &inner_fact.uniform_tdim {
                        cond_tdim = Some((TDim::Val(1) - inner_tdim.clone()).reduce());
                    }
                }
            }
        }

        let Some(expr) = cond_tdim else { return Ok(None) };

        let simplified = expr.clone().simplify();

        // Helper: replace Iff output with a specific input branch
        let shunt_to = |branch: OutletId| -> TractResult<Option<TypedModelPatch>> {
            let mut patch = TypedModelPatch::default();
            let wire = patch.tap_model(model, branch)?;
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            Ok(Some(patch))
        };

        // Rule 1a: condition is provably always false → use false branch (inputs[2])
        if simplified == TDim::Val(0) {
            return shunt_to(node.inputs[2]);
        }

        // Rule 1b: condition is provably always true → use true branch (inputs[1])
        if simplified == TDim::Val(1) {
            return shunt_to(node.inputs[1]);
        }

        let cond_shape = &cond_fact.shape;

        // Rule 1c: prove all-false using coordinate extremes
        if is_provably_all_false(&simplified, cond_shape) {
            return shunt_to(node.inputs[2]);
        }

        // Rule 1d: prove all-true using coordinate extremes
        if is_provably_all_true(&simplified, cond_shape) {
            return shunt_to(node.inputs[1]);
        }

        Ok(None)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }
}

bin_to_super_type!(bitand, BitAnd,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a & b);
bin_to_super_type!(bitor, BitOr,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a | b);
bin_to_super_type!(bitxor, BitXor,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a ^ b);

element_wise!(bitnot, BitNot, [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = !*x);
    Ok(())
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::change_axes::AxisOp;
    use crate::ops::logic::Comp;

    /// Test Case 1: Iff where condition is Eq(T, 0) with T >= 1 assertion.
    /// After declutter, the Iff should fold to the false branch (inputs[2]).
    #[test]
    fn iff_fold_case1_eq_t_zero() -> TractResult<()> {
        let mut model = TypedModel::default();
        model.symbols.add_assertion("T >= 1")?;
        let t_sym = model.symbols.sym("T");
        let t_dim = TDim::Sym(t_sym.clone());

        // Const T (scalar TDim)
        let t_wire = model.wire_node(
            "T",
            crate::ops::konst::Const::new(tensor0(t_dim.clone()).into_arc_tensor())?,
            &[],
        )?[0];

        // Const 0 (scalar TDim)
        let zero_wire = model.wire_node(
            "zero",
            crate::ops::konst::Const::new(tensor0(TDim::Val(0)).into_arc_tensor())?,
            &[],
        )?[0];

        // Eq(T, 0) → bool scalar
        let eq_wire = model.wire_node("eq", Comp::Eq, &[t_wire, zero_wire])?[0];

        // Some data wire for the false branch
        let data_wire = model.add_source("data", TDim::datum_type().scalar_fact())?;

        // Iff(eq, zero, data) — zero is "true" branch, data is "false" branch
        let iff_wire = model.wire_node("iff", Iff, &[eq_wire, zero_wire, data_wire])?[0];
        model.set_output_outlets(&[iff_wire])?;

        let model = model.into_decluttered()?;

        // The Iff should have been folded away (condition is always false given T >= 1)
        let iff_count = model.nodes().iter().filter(|n| n.op_as::<Iff>().is_some()).count();
        assert_eq!(iff_count, 0, "Expected Iff to be folded, but found {iff_count} Iff nodes");
        Ok(())
    }

    /// Test Case 2: range(0,T,1) → unsqueeze(0) → lt(_, T_unsqueezed) → bitnot → Iff
    /// The bitnot produces Ge(x1, T), all-false for x1 in [0, T-1].
    /// After declutter, the Iff should fold to the false branch (data input).
    #[test]
    fn iff_fold_case2_not_lt_x1_t() -> TractResult<()> {
        use crate::ops::array::Range;

        let mut model = TypedModel::default();
        model.symbols.add_assertion("T >= 1")?;
        let t_sym = model.symbols.sym("T");
        let t_dim = TDim::Sym(t_sym.clone());

        // Const start=0 (TDim) and step=1 (TDim) — these get uniform_tdim set in output_facts
        let start = model.wire_node(
            "start",
            crate::ops::konst::Const::new(tensor0(TDim::Val(0)).into_arc_tensor())?,
            &[],
        )?[0];
        let step = model.wire_node(
            "step",
            crate::ops::konst::Const::new(tensor0(TDim::Val(1)).into_arc_tensor())?,
            &[],
        )?[0];
        // T is a dynamic TDim input (not a Const) so Range takes the else branch and
        // sets uniform_tdim = start + step * x0 = x0
        let end = model.add_source("T_dyn", TDim::datum_type().scalar_fact())?;

        // Range(start=0, end=T, step=1) → [T] TDim with uniform_tdim = x0
        let range = model.wire_node("range", Range::new(t_dim.clone()), &[start, end, step])?[0];

        // unsqueeze(0) → [1, T] TDim, remap x0→x1 → uniform_tdim = x1
        let range_unsq = model.wire_node("range_unsq", AxisOp::Add(0), &[range])?[0];

        // T const for comparison, scalar TDim with uniform_tdim = Sym(T)
        let t_const = model.wire_node(
            "T_const",
            crate::ops::konst::Const::new(tensor0(t_dim.clone()).into_arc_tensor())?,
            &[],
        )?[0];
        // unsqueeze T_const → [1] TDim (scalar value, no coord remapping needed)
        let t_unsq = model.wire_node("T_unsq", AxisOp::Add(0), &[t_const])?[0];

        // lt(range_unsq=[1,T], T_unsq=[1]) → bool [1,T], uniform_tdim = Lt(x1,T)
        let lt = model.wire_node("lt", Comp::LT, &[range_unsq, t_unsq])?[0];

        // bitnot(lt): BitNot doesn't propagate uniform_tdim in output_facts,
        // but Iff::declutter traces through it to get Not(Lt(x1,T))=Ge(x1,T)
        let bn = model.wire_node("bitnot", bitnot(), &[lt])?[0];

        // Data source [1, T]
        let data_shape = tvec![TDim::Val(1), t_dim.clone()];
        let data = model.add_source("data", TDim::datum_type().fact(data_shape.clone()))?;

        // zeros broadcast to [1, T], uniform_tdim = Val(0)
        let zero_scalar = model.wire_node(
            "zero_s",
            crate::ops::konst::Const::new(tensor0(TDim::Val(0)).into_arc_tensor())?,
            &[],
        )?[0];
        let zeros = model.wire_node(
            "zeros",
            crate::ops::array::MultiBroadcastTo {
                shape: ShapeFact::from_dims(data_shape.iter().cloned()),
            },
            &[zero_scalar],
        )?[0];

        // Iff(bn, zeros, data): condition Ge(x1,T) is all-false → fold to data
        let iff = model.wire_node("iff", Iff, &[bn, zeros, data])?[0];
        model.set_output_outlets(&[iff])?;

        let model = model.into_decluttered()?;

        let iff_count = model.nodes().iter().filter(|n| n.op_as::<Iff>().is_some()).count();
        assert_eq!(iff_count, 0, "Expected Iff to be folded, but found {iff_count} Iff nodes");
        Ok(())
    }

    /// Verify that uniform_tdim propagation produces the expected values at each stage.
    #[test]
    fn verify_uniform_tdim_propagation() -> TractResult<()> {
        use crate::ops::array::Range;

        let mut model = TypedModel::default();
        model.symbols.add_assertion("T >= 1")?;
        let t_sym = model.symbols.sym("T");
        let t_dim = TDim::Sym(t_sym.clone());

        let start = model.wire_node(
            "start",
            crate::ops::konst::Const::new(tensor0(TDim::Val(0)).into_arc_tensor())?,
            &[],
        )?[0];
        let step = model.wire_node(
            "step",
            crate::ops::konst::Const::new(tensor0(TDim::Val(1)).into_arc_tensor())?,
            &[],
        )?[0];
        let end = model.add_source("T_dyn", TDim::datum_type().scalar_fact())?;
        let range = model.wire_node("range", Range::new(t_dim.clone()), &[start, end, step])?[0];
        let range_unsq = model.wire_node("range_unsq", AxisOp::Add(0), &[range])?[0];
        let t_const = model.wire_node(
            "T_const",
            crate::ops::konst::Const::new(tensor0(t_dim.clone()).into_arc_tensor())?,
            &[],
        )?[0];
        let t_unsq = model.wire_node("T_unsq", AxisOp::Add(0), &[t_const])?[0];
        let lt = model.wire_node("lt", Comp::LT, &[range_unsq, t_unsq])?[0];

        let range_fact = model.outlet_fact(range)?;
        let range_unsq_fact = model.outlet_fact(range_unsq)?;
        let t_unsq_fact = model.outlet_fact(t_unsq)?;
        let lt_fact = model.outlet_fact(lt)?;

        assert!(range_fact.uniform_tdim.is_some(), "range should have uniform_tdim");
        assert!(range_unsq_fact.uniform_tdim.is_some(), "range_unsq should have uniform_tdim");
        assert!(t_unsq_fact.uniform_tdim.is_some(), "t_unsq should have uniform_tdim");
        assert!(lt_fact.uniform_tdim.is_some(), "lt should have uniform_tdim");

        Ok(())
    }
}
