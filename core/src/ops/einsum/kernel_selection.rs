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

/// True when C's n axis is the innermost non-unit axis, so a row-major SME
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

/// SME 32×32 `st1w` writes a contiguous 128-byte row; a strided n axis would
/// need a scatter, which is illegal in streaming mode without
/// FEAT_SME_FA64, which Apple does not enable. `sme_mmm_f32_32x32` is the
/// exception once M is the contiguous axis: it stores a vertical ZA slice with
/// an ordinary `st1w`. GEMV (nr==1) is unaffected.
fn tile_store_needs_n_contiguous(name: &str, nr: usize) -> bool {
    nr > 1 && (name.starts_with("sme_mmm") || name.starts_with("sme_qmmm"))
}

/// True when C's m axis is the innermost non-unit axis, so a vertical SME
/// slice is a contiguous vector.
fn m_axis_contiguous(model: &TypedModel, node: &TypedNode, op: &EinSumMatMul) -> bool {
    let Some(cm) = op.c_m() else {
        return true;
    };
    let Ok(fact) = model.outlet_fact(node.id.into()) else {
        return true;
    };
    fact.shape.iter().skip(cm + 1).all(|d| d.is_one())
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
        let n_ok = n_axis_contiguous(model, node, op);
        let m_ok = m_axis_contiguous(model, node, op);
        // A strided N is fine for the f32 SME tile when M is the contiguous
        // axis: the kernel stores that direction as a vertical slice. Every
        // other wide tile still scatters and stays out.
        let vertical_sme = |name: &str| m_ok && name == "sme_mmm_f32_32x32";
        if tile_store_needs_n_contiguous(chosen.0.name(), chosen.0.nr())
            && !n_ok
            && !vertical_sme(chosen.0.name())
        {
            suitable.retain(|(k, _, _)| {
                !tile_store_needs_n_contiguous(k.name(), k.nr()) || vertical_sme(k.name())
            });
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

/// The strategy codegen uses for this node: the tuning table's choice for its
/// [`tuning_key`] when the table names one of the node's [`candidates`], the
/// [`strategize`] heuristic otherwise. With no tuning table in effect this is
/// exactly [`strategize`].
pub fn select(model: &TypedModel, node: &TypedNode, op: &EinSumMatMul) -> TractResult<Strat> {
    if crate::tuning::recording() {
        crate::tuning::record(crate::tuning::CandidateReport {
            node: node.name.clone(),
            key: tuning_key(model, node, op)?,
            candidates: candidates(model, node, op)?
                .iter()
                .map(|s| (s.2[0].0.name().to_string(), s.2[0].1))
                .collect(),
        });
    }
    if crate::tuning::active()
        && let Some(key) = tuning_key(model, node, op)?
        && let Some(choice) = crate::tuning::lookup(&key)
    {
        let tuned = candidates(model, node, op)?.into_iter().find(|s| {
            s.2.len() == 1 && s.2[0].0.name() == choice.kernel && s.2[0].1 == choice.packing
        });
        if let Some(tuned) = tuned {
            return Ok(tuned);
        }
    }
    strategize(model, node, op)
}

/// Every legal strategy for this node, the [`strategize`] pick first, then each
/// other suitable kernel and packing on its own. Legality is the store rule
/// `strategize` applies: an SME tile that scatters on a strided `n` is left out.
/// A node with a symbolic `n` has no alternatives: it keeps the heuristic's
/// vector/matrix pair.
pub fn candidates(
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSumMatMul,
) -> TractResult<Vec<Strat>> {
    let first = strategize(model, node, op)?;
    let query = query(model, node, op)?;
    if query.n.is_none() {
        return Ok(vec![first]);
    }
    let n_ok = n_axis_contiguous(model, node, op);
    let m_ok = m_axis_contiguous(model, node, op);
    let mut all = vec![first];
    for it in tract_linalg::MmmDispatch::native().suitable(&query) {
        let name = it.0.name();
        let vertical_sme = m_ok && name == "sme_mmm_f32_32x32";
        if tile_store_needs_n_contiguous(name, it.0.nr()) && !n_ok && !vertical_sme {
            continue;
        }
        let known = all.iter().any(|s| {
            s.2.len() == 1 && s.2[0].0.name() == name && s.2[0].1 == it.1 && s.2[0].2.is_none()
        });
        if !known {
            all.push(single_strat(it));
        }
    }
    Ok(all)
}

/// Identity of this node's matmul for the tuning table: the einsum expression,
/// operand and accumulator types, batch, m, k and n, whether A is a constant
/// (which decides how it is packed) and which of m and n is strided in C. Two
/// nodes with the same key take the same choice. `None` while any of those dims
/// is symbolic.
pub fn tuning_key(
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSumMatMul,
) -> TractResult<Option<String>> {
    let query = query(model, node, op)?;
    let (Some(m), Some(k), Some(n)) = (query.m, query.k, query.n) else {
        return Ok(None);
    };
    let c_fact = model.outlet_fact(node.id.into())?;
    let mut batch = 1i64;
    for (ix, d) in c_fact.shape.iter().enumerate() {
        if Some(ix) != op.c_m() && Some(ix) != op.c_n() {
            let Some(d) = d.as_i64() else { return Ok(None) };
            batch *= d;
        }
    }
    let a_const = model.node_input_facts(node.id)?[0].konst.is_some();
    let raw = format!(
        "einsum_{}_{:?}_{:?}_{:?}_b{batch}_m{m}_k{k}_n{n}{}{}{}",
        op.op.axes,
        query.weight,
        query.activation,
        op.operating_dt.unquantized(),
        if a_const { "_aconst" } else { "" },
        if n_axis_contiguous(model, node, op) { "" } else { "_nstrided" },
        if m_axis_contiguous(model, node, op) { "" } else { "_mstrided" },
    );
    Ok(Some(raw.chars().filter(|c| c.is_ascii_alphanumeric() || "_,->".contains(*c)).collect()))
}

/// The tuning key and candidates of every matmul codegen meets while
/// optimising `model`, in the order it meets them. The keys are the ones codegen
/// looks up, since they are recorded from the same call.
pub fn report(model: &TypedModel) -> TractResult<Vec<crate::tuning::CandidateReport>> {
    let (optimized, reports) = crate::tuning::with_recorder(|| model.clone().into_optimized());
    optimized?;
    Ok(reports)
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
mod sme_strided_store {
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

    /// m<n transposes so n is no longer innermost and m is. The f32 SME tile
    /// stores that direction; the i32 SME tile still must not.
    #[test]
    fn strided_n_does_not_select_scattering_tile() {
        if !has_tile_kernel() {
            return;
        }
        let model = optimize_mn(64, 64, 160);
        let mm = model.nodes.iter().find_map(|n| n.op_as::<OptMatMul>()).expect("OptMatMul");
        let name = mm.mmm[0].name();
        let sme_f32 = tract_linalg::MmmDispatch::native()
            .runnable()
            .iter()
            .any(|k| k.name() == "sme_mmm_f32_32x32");
        if sme_f32 {
            assert_eq!(name, "sme_mmm_f32_32x32", "m-contiguous GEMM selected {name}");
        } else {
            assert!(
                !tile_store_needs_n_contiguous(name, mm.mmm[0].nr()),
                "strided-n GEMM selected {name}"
            );
        }
    }

    /// m>n keeps n innermost, so the SME 32x32 tile is correct.
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

#[cfg(test)]
mod tuning {
    use super::*;
    use crate::ops::einsum::EinSum;
    use crate::ops::matmul::optimized::OptMatMul;
    use crate::tuning::Tuning;
    use crate::tuning::with_tuning;

    fn gemm(m: impl Into<TDim>, k: usize, n: usize) -> TypedModel {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact(&[m.into(), k.into()])).unwrap();
        let b = model.add_source("b", f32::fact([k, n])).unwrap();
        let out = model
            .wire_node(
                "gemm",
                EinSum::new("mk,kn->mn".parse().unwrap(), f32::datum_type()),
                &[a, b],
            )
            .unwrap();
        model.select_output_outlets(&out).unwrap();
        let mut model = model.into_decluttered().unwrap();
        let (node, op) =
            model.nodes.iter().find_map(|n| n.op_as::<EinSum>().map(|op| (n, op))).unwrap();
        let patch =
            crate::ops::einsum::einsum_matmul::detect_rule(&(), &model, node, &node.name, op)
                .unwrap()
                .unwrap();
        patch.apply(&mut model).unwrap();
        model
    }

    fn matmul_node(model: &TypedModel) -> (&TypedNode, &EinSumMatMul) {
        model.nodes.iter().find_map(|n| n.op_as::<EinSumMatMul>().map(|op| (n, op))).unwrap()
    }

    fn kernel_after_codegen(model: TypedModel) -> String {
        let model = model.into_optimized().unwrap();
        let mm = model.nodes.iter().find_map(|n| n.op_as::<OptMatMul>()).unwrap();
        mm.mmm[0].name().to_string()
    }

    #[test]
    fn candidates_start_with_the_heuristic_pick() {
        let model = gemm(64, 64, 64);
        let (node, op) = matmul_node(&model);
        let heuristic = strategize(&model, node, op).unwrap();
        let all = candidates(&model, node, op).unwrap();
        assert_eq!(all[0].2[0].0.name(), heuristic.2[0].0.name());
        assert!(all.iter().skip(1).all(|s| s.2.len() == 1));
    }

    #[test]
    fn key_needs_concrete_dims() {
        let model = gemm(64, 32, 16);
        let (node, op) = matmul_node(&model);
        let key = tuning_key(&model, node, op).unwrap().unwrap();
        assert!(key.starts_with("einsum_mk,kn->mn_"), "{key}");
        assert!(key.contains("_m64_k32_n16"), "{key}");
        let symbolic = gemm(model.symbols.sym("M"), 32, 16);
        let (node, op) = matmul_node(&symbolic);
        assert!(tuning_key(&symbolic, node, op).unwrap().is_none());
    }

    #[test]
    fn tuning_table_overrides_the_heuristic() {
        let model = gemm(64, 64, 64);
        let (node, op) = matmul_node(&model);
        let all = candidates(&model, node, op).unwrap();
        let Some(other) = all.iter().skip(1).find(|s| s.2[0].0.name() != all[0].2[0].0.name())
        else {
            return;
        };
        let (kernel, packing) = (other.2[0].0.name().to_string(), other.2[0].1);
        let key = tuning_key(&model, node, op).unwrap().unwrap();
        let table = Tuning::parse(&format!("{key} {kernel} {packing}")).unwrap();
        assert_eq!(with_tuning(table, || kernel_after_codegen(model.clone())), kernel);
        assert_eq!(kernel_after_codegen(model), all[0].2[0].0.name());
    }
}

#[cfg(test)]
mod report_keys {
    use super::*;
    use crate::ops::einsum::EinSum;
    use crate::ops::matmul::optimized::OptMatMul;
    use crate::tuning::Tuning;
    use crate::tuning::with_tuning;

    fn const_weight_gemm() -> TypedModel {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([64, 96])).unwrap();
        let w = model.add_const("w", Tensor::zero::<f32>(&[288, 96]).unwrap()).unwrap();
        let out = model
            .wire_node(
                "proj",
                EinSum::new("mk,nk->mn".parse().unwrap(), f32::datum_type()),
                &[x, w],
            )
            .unwrap();
        model.select_output_outlets(&out).unwrap();
        model.into_decluttered().unwrap()
    }

    fn kernels(model: TypedModel) -> Vec<String> {
        let model = model.into_optimized().unwrap();
        model
            .nodes
            .iter()
            .filter_map(|n| n.op_as::<OptMatMul>())
            .map(|mm| mm.mmm[0].name().to_string())
            .collect()
    }

    #[test]
    fn report_keys_are_the_ones_codegen_looks_up() {
        let model = const_weight_gemm();
        let reports = report(&model).unwrap();
        let r = &reports[0];
        let key = r.key.clone().unwrap();
        let Some((kernel, packing)) = r.candidates.iter().find(|c| c.0 != r.candidates[0].0) else {
            return;
        };
        let table = Tuning::parse(&format!("{key} {kernel} {packing}")).unwrap();
        assert_eq!(with_tuning(table, || kernels(model.clone())), vec![kernel.clone()]);
        assert_eq!(kernels(model), vec![r.candidates[0].0.clone()]);
    }
}
