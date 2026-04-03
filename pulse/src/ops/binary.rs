/// Pulsifier for `TypedBinOp` when the output wire carries a ROI annotation
/// **and** a `uniform_tdim` expression (integer coordinate function).
///
/// This fires for position-derived wires like `rel_pos = i - j` (Sub of two
/// position arrays) that participate in an attention position bias.  At pulse
/// time such a wire is window-constant: for chunk `c`, position bias
/// `[p, l] = f(left_chunks·P + p, l)` (using chunk c=left_chunks as reference),
/// which is independent of `c`.  We materialise it as a `Const` once.
///
/// The pulsifier is intentionally generic: it only requires ROI + uniform_tdim
/// on the output; it does not care which specific binary op produces the wire.
use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::{classify_chunk_window, sym_to_coord_axis};

register_all!(TypedBinOp: pulsify);

fn pulsify(
    _op: &TypedBinOp,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    _mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let outlet = OutletId::new(node.id, 0);
    let fact = source.outlet_fact(outlet)?;

    // Special case: Bool dtype with chunk-window uniform_tdim.
    // Fires for nodes like attMask = And(padMask, UniformTDim) where
    // uniform_tdim propagates a chunk-window expression.  The Iff consumer
    // will be replaced by ChunkWindowMask and ignores the actual bool values,
    // so we produce an all-true stub of the right shape [1,…,P,…,(L+1)P,…].
    if fact.datum_type == DatumType::Bool {
        // The stored output fact may not have uniform_tdim even when an input
        // does (FoldUniformTDim creates UniformTDim nodes but does not
        // re-propagate uniform_tdim to successor nodes).  Fall back to
        // scanning the node's input outlets.
        let uniform_expr = if let Some(e) = &fact.uniform_tdim {
            e.clone()
        } else {
            // Check if any input carries a chunk-window uniform_tdim.
            let found = node.inputs.iter().find_map(|inp| {
                let f = source.outlet_fact(*inp).ok()?;
                let e = f.uniform_tdim.as_ref()?;
                classify_chunk_window(&e.clone().simplify())?;
                Some(e.clone())
            });
            match found {
                Some(e) => e,
                None => return Ok(None),
            }
        };
        let cw = match classify_chunk_window(&uniform_expr.clone().simplify()) {
            Some(cw) => cw,
            None => return Ok(None),
        };
        let pulse_i64 = pulse.to_i64()?;
        // Compute effective token-axis pulse using shape delta (handles downsampling).
        let pulse_size =
            if let Some(dim) = fact.shape.iter().find(|d| d.symbols().contains(_symbol)) {
                let mut sv_at = SymbolValues::default();
                sv_at.set(_symbol, pulse_i64);
                let mut sv_zero = SymbolValues::default();
                sv_zero.set(_symbol, 0);
                let at_pulse = dim.eval(&sv_at).to_i64()?;
                let at_zero = dim.eval(&sv_zero).to_i64()?;
                (at_pulse - at_zero) as usize
            } else {
                cw.p as usize
            };
        ensure!(
            pulse_size == cw.p as usize,
            "Bool chunk-window pulsifier: pulse size {pulse_size} != chunk size {}",
            cw.p
        );
        let key_window = (cw.left_chunks as usize + 1) * pulse_size;
        let rank = fact.rank();
        let mut sv_zero = SymbolValues::default();
        sv_zero.set(_symbol, 0);
        let mut shape: Vec<usize> = Vec::with_capacity(rank);
        for (ax, dim) in fact.shape.iter().enumerate() {
            if ax == cw.row_axis {
                shape.push(pulse_size);
            } else if ax == cw.col_axis {
                shape.push(key_window);
            } else {
                // Non-window axis: evaluate at symbol=0, fall back to 1 for
                // batch or other undetermined symbols (the stub broadcasts).
                shape.push(dim.eval(&sv_zero).to_usize().unwrap_or(1));
            }
        }
        let total: usize = shape.iter().product();
        let tensor = Tensor::from_shape(&shape, &vec![true; total])?;
        return Ok(Some(target.wire_node(
            &node.name,
            NonPulsingWrappingOp(Box::new(Const::new(tensor.into_arc_tensor())?)),
            &[],
        )?));
    }

    // Need ROI + uniform_tdim on this wire, and a non-bool numeric dtype.
    let roi_expr = match &fact.region_of_interest {
        Some(e) => e.clone(),
        None => return Ok(None),
    };
    let uniform_expr = match &fact.uniform_tdim {
        Some(e) => e.clone(),
        None => return Ok(None),
    };

    let cw = match classify_chunk_window(&roi_expr.clone().simplify()) {
        Some(cw) => cw,
        None => return Ok(None),
    };

    let pulse_size = pulse.to_i64()? as usize;
    ensure!(
        pulse_size == cw.p as usize,
        "TypedBinOp ROI pulsifier: pulse size {pulse_size} != chunk size {}",
        cw.p
    );
    let key_window = (cw.left_chunks as usize + 1) * pulse_size;

    // Collect coordinate symbols from the expression and map each to its axis.
    let coord_syms: Vec<(usize, Symbol)> = uniform_expr
        .symbols()
        .into_iter()
        .filter_map(|s| sym_to_coord_axis(&s).map(|k| (k, s)))
        .collect();

    // Build the output shape: [1, …, P, …, (L+1)P, …] with row/col axes filled.
    let rank = fact.rank();
    let mut shape = vec![1usize; rank];
    shape[cw.row_axis] = pulse_size;
    shape[cw.col_axis] = key_window;

    // Compute per-axis strides (row-major).
    let strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for ax in (0..rank.saturating_sub(1)).rev() {
            s[ax] = s[ax + 1] * shape[ax + 1];
        }
        s
    };

    let total: usize = shape.iter().product();

    // Evaluate uniform_tdim for each position in the window.
    // Coordinate mapping (using reference chunk c = left_chunks):
    //   row_axis dim: absolute coord = left_chunks * P + p_local
    //   col_axis dim: absolute coord = l_local  (starts at 0)
    //   other dims:   absolute coord = local index (usually 0 for size-1 dims)
    let mut int_values = vec![0i64; total];
    for flat in 0..total {
        let mut remaining = flat;
        let mut idx = vec![0usize; rank];
        for ax in 0..rank {
            idx[ax] = remaining / strides[ax];
            remaining %= strides[ax];
        }

        let mut sv = SymbolValues::default();
        for &(k, ref sym) in &coord_syms {
            let coord = if k == cw.row_axis {
                cw.left_chunks as i64 * cw.p as i64 + idx[k] as i64
            } else if k == cw.col_axis {
                idx[k] as i64
            } else {
                idx[k] as i64
            };
            sv.set(sym, coord);
        }
        int_values[flat] = uniform_expr.eval(&sv).to_i64()?;
    }

    // Cast to the wire's datum type.
    let tensor = match fact.datum_type {
        DatumType::F32 => {
            let vals: Vec<f32> = int_values.iter().map(|&v| v as f32).collect();
            Tensor::from_shape(&shape, &vals)?
        }
        DatumType::I64 => Tensor::from_shape(&shape, &int_values)?,
        DatumType::I32 => {
            let vals: Vec<i32> = int_values.iter().map(|&v| v as i32).collect();
            Tensor::from_shape(&shape, &vals)?
        }
        _ => return Ok(None),
    };

    Ok(Some(target.wire_node(
        &node.name,
        NonPulsingWrappingOp(Box::new(Const::new(tensor.into_arc_tensor())?)),
        &[],
    )?))
}
