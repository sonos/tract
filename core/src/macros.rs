#[macro_export]
macro_rules! dims {
    ($($dim:expr),*) => {
        ShapeFact::from(&[$(TDim::from($dim.clone())),*])
    }
}

/// Boilerplate body for `TypedOp::substitute_symbols` when the op
/// derives `SubstituteSymbols`. Expands to:
///
/// ```ignore
/// let new_op = self.auto_subst_symbols(subs)?;
/// let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
/// target.wire_node(&node.name, new_op, &inputs)
/// ```
///
/// Usage inside `impl TypedOp for MyOp`:
///
/// ```ignore
/// fn substitute_symbols(
///     &self,
///     _source: &TypedModel,
///     node: &TypedNode,
///     target: &mut TypedModel,
///     mapping: &HashMap<OutletId, OutletId>,
///     subs: &HashMap<Symbol, TDim>,
/// ) -> TractResult<TVec<OutletId>> {
///     $crate::substitute_symbols_default!(self, node, target, mapping, subs)
/// }
/// ```
#[macro_export]
macro_rules! substitute_symbols_default {
    ($self:ident, $node:ident, $target:ident, $mapping:ident, $subs:ident) => {{
        let new_op = $self.auto_subst_symbols($subs)?;
        let inputs = $node.inputs.iter().map(|i| $mapping[i]).collect::<TVec<_>>();
        $target.wire_node(&$node.name, new_op, &inputs)
    }};
}
