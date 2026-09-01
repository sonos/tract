#![allow(clippy::type_complexity)]

use dyn_clone::clone_box;
use tract_itertools::Itertools;
use tract_linalg::WeightType;
use tract_linalg::block_quant::BlockQuantFact;
use tract_linalg::mmm::{MMMInputFormat, Query, Suitable, pick_by_shape, retain_best};

use crate::internal::*;
use crate::ops::matmul::ModePicker;

use super::einsum_matmul::EinSumMatMul;

pub type Impl = Suitable;
pub type Strat = (ModePicker, Box<dyn MMMInputFormat>, Vec<Impl>);

fn single_strat(it: Impl) -> Strat {
    (ModePicker::Single, it.0.packings()[it.1].0.clone(), vec![it])
}

/// True when C's n axis is the innermost non-unit axis, so a row-major AMX/SME
/// store hits its aligned bulk path (`col_byte_stride == item_size`).
fn n_axis_contiguous(model: &TypedModel, node: &TypedNode, op: &EinSumMatMul) -> bool {
    let Some(cn) = op.c_n() else {
        return true;
    };
    let Ok(fact) = model.outlet_fact(node.id.into()) else {
        return true;
    };
    fact.shape.iter().skip(cn + 1).all(|d| d.is_one())
}

/// AMX and SME 32x32 `stz`/`st1w` write a contiguous 128-byte row. A strided n
/// axis makes that kernel 20× slower in-graph than isolated (50 µs vs 2.4 µs
/// on DPDFNet 48 kHz 160×64×64). NEON 8x8 is the right pick there. GEMV
/// (nr==1) is unaffected.
fn tile_store_needs_n_contiguous(name: &str, nr: usize) -> bool {
    nr > 1
        && (name.starts_with("apple_amx")
            || name.starts_with("sme_mmm")
            || name.starts_with("sme_qmmm"))
}

pub fn strategize(model: &TypedModel, node: &TypedNode, op: &EinSumMatMul) -> TractResult<Strat> {
    let query = query(model, node, op)?;
    let mut suitable = tract_linalg::MmmDispatch::native().suitable(&query);
    ensure!(suitable.len() > 0);
    // Only with `n` in hand: a symbolic `n` is what the packing-group reasoning below is for,
    // and it serves both roles at once, which a single pick cannot.
    if query.n.is_some()
        && let Some(chosen) = tract_linalg::MmmDispatch::native().preferred(&query, &suitable)
    {
        if tile_store_needs_n_contiguous(chosen.0.name(), chosen.0.nr())
            && !n_axis_contiguous(model, node, op)
        {
            suitable.retain(|(k, _, _)| !tile_store_needs_n_contiguous(k.name(), k.nr()));
            if let Some(fallback) = tract_linalg::MmmDispatch::native().preferred(&query, &suitable)
            {
                return Ok(single_strat(fallback));
            }
        } else {
            return Ok(single_strat(chosen));
        }
    }
    retain_best(&mut suitable);
    if suitable.len() == 1 {
        return Ok(single_strat(suitable.remove(0)));
    }
    if let Some(ix) = pick_by_shape(&query, &suitable) {
        return Ok(single_strat(suitable.swap_remove(ix)));
    }
    let mut grouped_by_left_packing = Vec::<(&dyn MMMInputFormat, Vec<_>)>::new();
    'mmm: for (m, p, pe) in &suitable {
        let left_packing: &dyn MMMInputFormat =
            pe.as_ref().map(|pe| &*pe.from).unwrap_or(&*m.packings()[*p].0);
        for group in &mut grouped_by_left_packing {
            if let Some(merged) = group.0.merge_with(left_packing) {
                group.0 = merged;
                group.1.push((m, p, pe));
                continue 'mmm;
            }
        }
        grouped_by_left_packing.push((left_packing, vec![(m, p, pe)]));
    }
    let (p, mmv, mmm) = grouped_by_left_packing
        .iter()
        .map(|(p, group)| {
            let best_for_mmv =
                group.iter().max_by_key(|(m, _, pe)| (m.nr() == 1, pe.is_none())).unwrap();
            let best_for_mmm = group.iter().max_by_key(|(m, _, _)| m.nr()).unwrap();
            (p, best_for_mmv, best_for_mmm)
        })
        .max_by_key(|(_, mmv, mmm)| {
            // When no group offers the ideal (true GEMV nr==1 + true matrix nr>1)
            // pair, still prefer a group whose matrix-role kernel is a real matrix
            // (nr > 1) over a GEMV-only group. Without this, int8 — whose GEMV
            // (64x1), SMLAL (8x8) and SDOT (8x8_dot) kernels each use a different
            // packing, so no single group is ideal — falls through to `mmm.mr` and
            // picks the 64x1 GEMV even for symbolic (dynamic) n. f32/f16/block-quant
            // are unaffected: they have a packing group that IS ideal (e.g. f32
            // 32x1/32x3, q40 32x1/32x3), so the first key already decides.
            (
                mmv.0.nr() == 1 && mmm.0.nr() > 1,
                mmv.2.is_none(),
                mmm.0.nr() > 1,
                mmm.0.mr(),
                mmm.0.nr(),
            )
        })
        .unwrap();

    if mmm == mmv {
        Ok((ModePicker::Single, clone_box(*p), vec![(mmv.0.clone(), *mmv.1, mmv.2.clone())]))
    } else {
        Ok((
            ModePicker::VecVsMat,
            clone_box(*p),
            vec![(mmv.0.clone(), *mmv.1, mmv.2.clone()), (mmm.0.clone(), *mmm.1, mmm.2.clone())],
        ))
    }
}

/// The node's matmul as kernel selection sees it: operand types from the input facts, dims
/// wherever they are already concrete.
pub fn query(model: &TypedModel, node: &TypedNode, op: &EinSumMatMul) -> TractResult<Query> {
    let (a_fact, b_fact) = model.node_input_facts(node.id)?.into_iter().collect_tuple().unwrap();
    let a_dt = a_fact.datum_type;
    let b_dt = b_fact.datum_type;

    let a_weight: WeightType = if let Some(of) = a_fact.exotic_fact() {
        if let Some(bqf) = of.downcast_ref::<BlockQuantFact>() {
            WeightType::BlockQuant(bqf.format.clone())
        } else {
            bail!("Can not translate to matmul operand {a_fact:?}");
        }
    } else {
        a_dt.into()
    };

    Ok(Query {
        weight: a_weight,
        activation: b_dt,
        accumulators: op.acceptable_accumulators(),
        store: Some(op.operating_dt.unquantized()),
        allow_extractor: true,
        m: op.m.as_i64().map(|d| d as usize),
        k: op.k.as_i64().map(|d| d as usize),
        n: op.n.as_i64().map(|d| d as usize),
    })
}

#[cfg(test)]
mod amx_strided_store {
    use super::*;
    use crate::ops::einsum::EinSum;
    use crate::ops::matmul::optimized::OptMatMul;

    fn optimize_mn(m: usize, k: usize, n: usize) -> TypedModel {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([m, k])).unwrap();
        let b = model.add_source("b", f32::fact([k, n])).unwrap();
        let out = model
            .wire_node(
                "gemm",
                EinSum::new("mk,kn->mn".parse().unwrap(), f32::datum_type()),
                &[a, b],
            )
            .unwrap();
        model.select_output_outlets(&out).unwrap();
        model.optimize().unwrap();
        model
    }

    fn has_tile_kernel() -> bool {
        tract_linalg::MmmDispatch::native()
            .runnable()
            .iter()
            .any(|k| tile_store_needs_n_contiguous(k.name(), k.nr()))
    }

    /// m<n transposes so n is no longer innermost. AMX/SME 32x32 must not be the kernel.
    #[test]
    fn strided_n_does_not_select_amx_32x32() {
        if !has_tile_kernel() {
            return;
        }
        let model = optimize_mn(64, 64, 160);
        let mm = model.nodes.iter().find_map(|n| n.op_as::<OptMatMul>()).expect("OptMatMul");
        assert!(
            !tile_store_needs_n_contiguous(mm.mmm[0].name(), mm.mmm[0].nr()),
            "strided-n GEMM selected {}",
            mm.mmm[0].name()
        );
    }

    /// m>n keeps n innermost. AMX 32x32 (M1–M3) or SME 32x32 (M4+) is correct.
    #[test]
    fn contiguous_n_selects_wide_tile() {
        if !has_tile_kernel() {
            return;
        }
        let model = optimize_mn(160, 64, 64);
        let mm = model.nodes.iter().find_map(|n| n.op_as::<OptMatMul>()).expect("OptMatMul");
        assert!(
            tile_store_needs_n_contiguous(mm.mmm[0].name(), mm.mmm[0].nr()),
            "contiguous-n GEMM selected {}",
            mm.mmm[0].name()
        );
    }
}
