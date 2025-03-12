use std::collections::HashSet;

use tract_core::internal::*;

pub fn figure_out_b_s_p(model: &TypedModel) -> TractResult<(Option<Symbol>, Symbol, Symbol)> {
    // expectations:
    // - one input is for tokens, so integer dt (i64 ?) and typically of shape S or 1,S, or B,S
    // - other inputs are kv cache, some kind of float. shape features both S and P, and B if B is present in tokens
    let token_input = model
        .inputs
        .iter()
        .position(|i| model.outlet_fact(*i).unwrap().datum_type.is_integer())
        .context("No token input found")?;
    let tokens_symbols = model.input_fact(token_input)?.shape.volume().symbols();
    let kv_input = model
        .inputs
        .iter()
        .position(|i| model.outlet_fact(*i).unwrap().datum_type.is_float())
        .context("No kv input found")?;
    let kv_symbols = model.input_fact(kv_input)?.shape.volume().symbols();
    let b = tokens_symbols.intersection(&kv_symbols).cloned().collect::<HashSet<_>>();
    let s = tokens_symbols.difference(&b).cloned().collect::<HashSet<_>>();
    let p = kv_symbols.difference(&b).cloned().collect::<HashSet<_>>();
    Ok((b.into_iter().next(), s.into_iter().next().unwrap(), p.into_iter().next().unwrap()))
}
