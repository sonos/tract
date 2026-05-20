use crate::internal::*;
use crate::ops::logic::sym_to_coord_axis;
use crate::optim::OptimizerSession;

/// Backward pass that propagates `region_of_interest` annotations by
/// calling `TypedOp::input_roi` on each node.
///
/// Ops can **introduce** ROIs (e.g. Iff reads its mask's uniform_tdim and
/// creates a ROI on the scores input) or **bubble** them (e.g. element-wise
/// ops pass an output ROI through to their inputs).
///
/// When multiple consumers of a wire produce different ROIs, they are merged
/// via boolean OR using De Morgan: `a ∨ b = a + b - a * b`.
/// If any consumer returns `None` for a wire (needs all positions), that wire
/// gets no ROI.
///
/// The pass iterates to fixpoint: introductions may enable further bubbling.
#[derive(Clone, Debug, Default)]
pub struct PropagateRoi;

/// Merge two ROI expressions via boolean OR: `a ∨ b = a + b - a * b`.
fn roi_union(a: &TDim, b: &TDim) -> TDim {
    if a == b {
        return a.clone();
    }
    a.clone() + b.clone() - a.clone() * b.clone()
}

/// Bubble output ROI to inputs using the op's axes_mapping.
///
/// For each input, builds a coordinate substitution map from the axes mapping:
/// each output axis that appears in this input gets 🎯{out_pos} → 🎯{in_pos}.
/// If any ROI coordinate symbol has no corresponding input axis (contracted,
/// broadcast from dim=1, or absent), returns None for that input.
pub fn bubble_roi(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TVec<Option<TDim>>>> {
    let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
    rule_if_some!(roi = &output_fact.region_of_interest);

    let input_facts: TVec<&TypedFact> =
        node.inputs.iter().map(|i| model.outlet_fact(*i)).collect::<TractResult<_>>()?;
    let output_facts = tvec![output_fact];
    let inputs_ref: Vec<&TypedFact> = input_facts.iter().copied().collect();
    let outputs_ref: Vec<&TypedFact> = output_facts.iter().copied().collect();
    let mapping = node.op.as_typed().unwrap().axes_mapping(&inputs_ref, &outputs_ref)?;

    // Collect ROI coordinate symbols and their output axis positions.
    let roi_coord_syms: Vec<(usize, Symbol)> =
        roi.symbols().into_iter().filter_map(|s| sym_to_coord_axis(&s).map(|k| (k, s))).collect();

    let remap_for_input = |input_ix: usize| -> Option<TDim> {
        let mut sub_map: HashMap<Symbol, TDim> = HashMap::new();
        for (out_pos, sym) in &roi_coord_syms {
            let logical = mapping
                .iter_all_axes()
                .find(|a| a.outputs.first().is_some_and(|o| o.contains(out_pos)))?;
            if logical.inputs[input_ix].is_empty() {
                return None;
            }
            let in_pos = logical.inputs[input_ix][0];
            if input_facts[input_ix].shape[in_pos] != output_fact.shape[*out_pos] {
                return None;
            }
            if in_pos != *out_pos {
                let scope = sym.scope()?;
                sub_map.insert(sym.clone(), TDim::Sym(scope.coord_sym(in_pos)));
            }
        }
        if sub_map.is_empty() { Some(roi.clone()) } else { roi.substitute_all(&sub_map).ok() }
    };
    let result: TVec<Option<TDim>> = (0..node.inputs.len()).map(|ix| remap_for_input(ix)).collect();
    Ok(Some(result))
}

/// Recognise a chunked-band predicate on output coords `(p, k_axis)` of the
/// shape produced by `DiagGather::input_roi`'s `c → r + q − offset`
/// substitution applied to a `Mul(Ge(L, q/k − c/k), Ge(q/k − c/k, 0))` band,
/// and return the projected band on `k_axis` after existentially
/// quantifying `p` over its dim bound.
///
/// Specifically, recognises:
///
///   `Mul(Ge(L_val, A), Ge(A, 0))`
///
/// where `A = ⌊p/k⌋ − ⌊(p + k_axis − offset)/k⌋` with `p` the projected
/// coord symbol (e.g. query) and `k_axis` the kept coord symbol (e.g.
/// rel-pos index).  Closed-form projection: as `p` varies, `A` takes
/// values in `{−⌈(k_axis − offset)/k⌉, −⌊(k_axis − offset)/k⌋}`, so the
/// existential `0 ≤ A ≤ L_val` is satisfiable iff
///
///   `k_axis ∈ [offset − (L_val + 1)·k + 1, offset + (k − 1)]`
///
/// — a constant band of width `(L_val + 2)·k − 1`.
///
/// Returns `None` if the pattern doesn't match.
pub fn recognise_chunked_band_project(roi: &TDim, p_sym: &Symbol, k_sym: &Symbol) -> Option<TDim> {
    // Match Mul(Ge(L, A), Ge(A, R)).
    let TDim::Mul(terms) = roi else { return None };
    if terms.len() != 2 {
        return None;
    }
    let TDim::Ge(top_l, top_r) = &terms[0] else { return None };
    let TDim::Ge(bot_l, bot_r) = &terms[1] else { return None };

    // Identify which orientation: top = Ge(L, A) and bot = Ge(A, R)?
    // We need the same `A` to appear as second arg of first and first arg
    // of second.
    let (l_val, a, r_val) = if top_r.as_ref() == bot_l.as_ref() {
        (top_l.as_ref(), top_r.as_ref(), bot_r.as_ref())
    } else if top_l.as_ref() == bot_r.as_ref() {
        // Reverse: top is Ge(A, L'), bot is Ge(R', A) — swap roles.
        (bot_l.as_ref(), top_l.as_ref(), top_r.as_ref())
    } else {
        return None;
    };

    // R side must be 0 (the band is 0 ≤ X ≤ L).
    if r_val != &TDim::Val(0) {
        return None;
    }
    let big_l = l_val.to_i64().ok()?;
    if big_l < 0 {
        return None;
    }

    // `A` may have a constant offset c factored out by the simplifier (e.g.
    // when the original offset isn't a multiple of k, the simplifier
    // rewrites `(p+r-9)/k` as `(p+r+5)/k - 1` for k=14).  Peel c off so
    // we can match the inner diff-of-divs, then re-fold c·k into the
    // recovered offset.
    let (a_no_const, c) = split_const(a);
    let (k, p_num, q_num) = match_diff_of_divs(&a_no_const)?;
    let derived_inner_offset = (p_num + TDim::Sym(k_sym.clone()) - q_num).reduce();
    if derived_inner_offset.symbols().contains(p_sym)
        || derived_inner_offset.symbols().contains(k_sym)
    {
        return None;
    }
    let actual_offset = (derived_inner_offset + TDim::Val(c * k as i64)).reduce();

    // The projected band on k_sym: [offset − (L+1)·k + 1, offset + (k − 1)].
    let high = (actual_offset.clone() + TDim::Val(k as i64 - 1)).reduce();
    let low = (actual_offset - TDim::Val((big_l + 1) * k as i64 - 1)).reduce();
    Some(
        TDim::Mul(vec![
            TDim::Ge(Box::new(high), Box::new(TDim::Sym(k_sym.clone()))),
            TDim::Ge(Box::new(TDim::Sym(k_sym.clone())), Box::new(low)),
        ])
        .reduce(),
    )
}

/// Split `expr` into `(expr_without_constant, constant_part)`.  If `expr`
/// is `Add([...constants..., ...non-constants...])`, sum up the constant
/// terms and return the non-constant remainder.  Otherwise returns
/// `(expr, 0)`.
fn split_const(expr: &TDim) -> (TDim, i64) {
    if let TDim::Add(terms) = expr {
        let mut c = 0i64;
        let mut rest: Vec<TDim> = vec![];
        for t in terms {
            match t {
                TDim::Val(v) => c += *v,
                _ => rest.push(t.clone()),
            }
        }
        let new_expr = if rest.is_empty() {
            TDim::Val(0)
        } else if rest.len() == 1 {
            rest.into_iter().next().unwrap()
        } else {
            TDim::Add(rest)
        };
        return (new_expr, c);
    }
    (expr.clone(), 0)
}

/// If `expr` matches `Div(p_expr, k) − Div(q_expr, k)` (in either order),
/// returns `(k, p_expr, q_expr)` where `p_expr` is the numerator with the
/// positive coefficient.
fn match_diff_of_divs(expr: &TDim) -> Option<(u64, TDim, TDim)> {
    let TDim::Add(terms) = expr else { return None };
    if terms.len() != 2 {
        return None;
    }
    let mut pos_div: Option<(TDim, u64)> = None;
    let mut neg_div: Option<(TDim, u64)> = None;
    for t in terms {
        match t {
            TDim::Div(inner, k) => {
                pos_div = Some(((**inner).clone(), *k));
            }
            TDim::MulInt(-1, inner) => {
                if let TDim::Div(num, k) = inner.as_ref() {
                    neg_div = Some(((**num).clone(), *k));
                }
            }
            _ => {}
        }
    }
    let (p_expr, k1) = pos_div?;
    let (q_expr, k2) = neg_div?;
    if k1 != k2 {
        return None;
    }
    Some((k1, p_expr, q_expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed-form recognition: chunked-band predicate after DG substitution
    /// `c → r + q − offset` should project `q` out and yield a constant band
    /// on `r` of width `(L+2)·k − 1`, centred around `offset`.
    #[test]
    fn recognise_chunked_band_yields_constant_band() {
        let scope = SymbolScope::default();
        let p = scope.coord_sym(0); // q (projected)
        let k_ax = scope.coord_sym(1); // r (kept)
        let offset = 9i64;
        let k: u64 = 14;
        let big_l = 5i64;

        // A = p/k − (p + k_ax − offset)/k
        let num1 = TDim::Sym(p.clone());
        let num2 = TDim::Sym(p.clone()) + TDim::Sym(k_ax.clone()) - TDim::Val(offset);
        let a = (TDim::Div(Box::new(num1), k) - TDim::Div(Box::new(num2), k)).reduce();
        let band = TDim::Mul(vec![
            TDim::Ge(Box::new(TDim::Val(big_l)), Box::new(a.clone())),
            TDim::Ge(Box::new(a), Box::new(TDim::Val(0))),
        ])
        .reduce();
        eprintln!("input band: {band}");

        let projected =
            recognise_chunked_band_project(&band, &p, &k_ax).expect("recogniser should match");
        eprintln!("projected: {projected}");

        // Expected: r ∈ [offset − (L+1)·k + 1, offset + (k − 1)]
        //         = [9 − 84 + 1, 9 + 13] = [-74, 22] (width 97)
        let high_expected = offset + k as i64 - 1; // 22
        let low_expected = offset - (big_l + 1) * k as i64 + 1; // -74
        let TDim::Mul(terms) = &projected else { panic!("expected Mul") };
        assert_eq!(terms.len(), 2);
        // Position-independent: one Ge term is `Ge(high, r)` (= r ≤ high),
        // the other is `Ge(r, low)` (= r ≥ low).
        let mut saw_high = false;
        let mut saw_low = false;
        for t in terms {
            let TDim::Ge(l, r) = t else { panic!("expected Ge inside Mul") };
            if **l == TDim::Val(high_expected) && **r == TDim::Sym(k_ax.clone()) {
                saw_high = true;
            } else if **l == TDim::Sym(k_ax.clone()) && **r == TDim::Val(low_expected) {
                saw_low = true;
            }
        }
        assert!(saw_high, "missing Ge(high={high_expected}, r); got: {projected}");
        assert!(saw_low, "missing Ge(r, low={low_expected}); got: {projected}");
    }
}

impl super::TypedPass for PropagateRoi {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        _model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    fn run_direct(&mut self, model: &mut TypedModel) -> TractResult<bool> {
        let order = model.eval_order()?;
        let mut any_changed = false;

        loop {
            let mut changed = false;
            let mut demands: HashMap<OutletId, Option<TDim>> = HashMap::new();

            for &node_id in &order {
                let node = &model.nodes()[node_id];
                let Some(input_rois) = node.op.as_typed().unwrap().input_roi(model, node)? else {
                    continue;
                };
                for (ix, roi) in input_rois.into_iter().enumerate() {
                    let outlet = node.inputs[ix];
                    match (demands.get(&outlet), &roi) {
                        (_, None) => {
                            demands.insert(outlet, None);
                        }
                        (Option::None, Some(roi)) => {
                            demands.insert(outlet, Some(roi.clone()));
                        }
                        (Some(None), Some(_)) => {}
                        (Some(Some(existing)), Some(new)) => {
                            demands.insert(outlet, Some(roi_union(existing, new)));
                        }
                    }
                }
            }

            // Apply demands to model facts.
            for (outlet, demand) in demands {
                if let Some(roi) = demand {
                    let roi = roi.simplify();
                    // ROI of 1 means "all positions matter" — equivalent to None.
                    if roi == TDim::Val(1) {
                        continue;
                    }
                    let fact = &mut model.nodes_mut()[outlet.node].outputs[outlet.slot].fact;
                    if fact.region_of_interest.as_ref() != Some(&roi) {
                        fact.region_of_interest = Some(roi);
                        changed = true;
                    }
                }
            }

            any_changed |= changed;
            if !changed {
                break;
            }
        }

        Ok(any_changed)
    }
}
