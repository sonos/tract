/// `UniformTDim` operator.
///
/// Materialises a bool tensor whose element at index `[i0, i1, ...]` is
/// determined by evaluating a `TDim` boolean expression with the coordinate
/// symbols `🎯0=i0, 🎯1=i1, …` substituted by concrete index values.
///
/// This is the analogue of `Const` for `uniform_tdim`: `FoldUniformTDim`
/// replaces an entire mask computation subgraph with a single `UniformTDim`
/// node whenever the wire's `uniform_tdim` is known.
///
/// # Inputs
/// Zero or one.  When the shape contains model symbols (e.g. S), `FoldUniformTDim`
/// wires a model input as a dummy dependency (input 0) to force topological
/// ordering after the Source node so that the symbol is resolved in
/// `session.resolved_symbols` before eval.  The value of input 0 is unused.
///
/// # Output
/// A bool tensor, shape = `self.shape` evaluated to concrete dims,
/// with `uniform_tdim = self.expr` on the output fact.
use crate::internal::*;
use crate::ops::logic::sym_to_coord_axis;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct UniformTDim {
    /// Boolean TDim expression in coordinate symbols 🎯0, 🎯1, ...
    pub expr: TDim,
    /// Symbolic output shape (may contain model symbols such as S).
    pub shape: ShapeFact,
    /// Output datum type (typically bool).
    pub dt: DatumType,
}

impl UniformTDim {
    pub fn new(expr: TDim, shape: ShapeFact, dt: DatumType) -> Self {
        UniformTDim { expr, shape, dt }
    }
}

impl Op for UniformTDim {
    fn name(&self) -> StaticName {
        "UniformTDim".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("expr: {}", self.expr), format!("shape: {:?}", self.shape)])
    }

    op_as_typed_op!();
}

impl EvalOp for UniformTDim {
    fn is_stateless(&self) -> bool {
        false // needs resolved_symbols from the session
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        _inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let resolved = &session.resolved_symbols;

        // Evaluate each symbolic dimension to a concrete size.
        let shape: Vec<usize> =
            self.shape.iter().map(|d| d.eval(resolved).to_usize()).collect::<TractResult<_>>()?;

        let rank = shape.len();
        let total: usize = shape.iter().product();

        // Extract coordinate symbols referenced in the expression.
        let coord_syms: Vec<(usize, Symbol)> = self
            .expr
            .symbols()
            .into_iter()
            .filter_map(|s| sym_to_coord_axis(&s).filter(|&k| k < rank).map(|k| (k, s)))
            .collect();

        // Compute per-axis strides (row-major).
        let strides: Vec<usize> = {
            let mut s = vec![1usize; rank];
            for ax in (0..rank.saturating_sub(1)).rev() {
                s[ax] = s[ax + 1] * shape[ax + 1];
            }
            s
        };

        let mut values = vec![false; total];
        for flat in 0..total {
            let mut remaining = flat;
            let mut idx = vec![0usize; rank];
            for ax in 0..rank {
                idx[ax] = remaining / strides[ax];
                remaining %= strides[ax];
            }

            let mut sv = SymbolValues::default();
            for &(k, ref sym) in &coord_syms {
                sv.set(sym, idx[k] as i64);
            }

            let val = self.expr.eval(&sv).to_i64().unwrap_or(0);
            values[flat] = val != 0;
        }

        let mut output = Tensor::zero_dt(self.dt, &shape)?;
        output.try_as_plain_mut()?.as_slice_mut::<bool>()?.copy_from_slice(&values);
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for UniformTDim {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = self.dt.fact(self.shape.clone());
        fact.uniform_tdim = Some(self.expr.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build chunk-window expr: 0 ≤ floor(🎯0/P) - floor(🎯1/P) ≤ L
    fn chunk_window_expr(scope: &SymbolScope, p: u64, l: i64) -> TDim {
        let row = TDim::Sym(scope.coord_sym(0));
        let col = TDim::Sym(scope.coord_sym(1));
        let diff = (TDim::Div(Box::new(row), p) - TDim::Div(Box::new(col), p)).simplify();
        let ge_upper = TDim::Ge(Box::new(TDim::Val(l)), Box::new(diff.clone()));
        let ge_lower = TDim::Ge(Box::new(diff), Box::new(TDim::Val(0)));
        TDim::Mul(vec![ge_upper, ge_lower])
    }

    #[test]
    fn uniform_tdim_chunk_window_eval() -> TractResult<()> {
        // P=2, L=1, S=4: produces a 4x4 bool mask
        // mask[i,j] = true iff 0 <= floor(i/2) - floor(j/2) <= 1
        let scope = SymbolScope::default();
        let expr = chunk_window_expr(&scope, 2, 1);
        let op = UniformTDim::new(
            expr.clone(),
            ShapeFact::from_dims([4.to_dim(), 4.to_dim()]),
            bool::datum_type(),
        );

        // Evaluate via a minimal model so we get a TurnState.
        let mut model = TypedModel::default();
        let out = model.wire_node("uniform", op, &[])?[0];
        model.select_output_outlets(&[out])?;
        let model = model.into_runnable()?;
        let result = model.run(tvec!())?;

        let mask = result[0].try_as_plain()?.as_slice::<bool>()?;

        // chunk[0]=chunk[1]=0, chunk[2]=chunk[3]=1
        // i=0,1: see chunks 0..0 → T T F F
        // i=2,3: see chunks 0..1 → T T T T
        let expected = [
            true, true, false, false, true, true, false, false, true, true, true, true, true, true,
            true, true,
        ];
        assert_eq!(mask, &expected);
        Ok(())
    }
}
