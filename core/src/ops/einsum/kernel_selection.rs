#![allow(clippy::type_complexity)]

use dyn_clone::clone_box;
use tract_itertools::Itertools;
use tract_linalg::WeightType;
use tract_linalg::block_quant::BlockQuantFact;
use tract_linalg::mmm::{ImplementationQuality, MMMInputFormat, MatMatMul, PanelExtractor};

use crate::internal::*;
use crate::ops::matmul::ModePicker;

use super::einsum_matmul::EinSumMatMul;

pub type Impl = (Box<dyn MatMatMul>, usize, Option<PanelExtractor>);
pub type Strat = (ModePicker, Box<dyn MMMInputFormat>, Vec<Impl>);

fn single_strat(it: Impl) -> Strat {
    (ModePicker::Single, it.0.packings()[it.1].0.clone(), vec![it])
}

pub fn strategize(model: &TypedModel, node: &TypedNode, op: &EinSumMatMul) -> TractResult<Strat> {
    let input_facts = model.node_input_facts(node.id)?;
    if let (Some(m), Some(k), Some(n)) = (op.m.as_i64(), op.k.as_i64(), op.n.as_i64()) {
        if op.op.operating_dt == input_facts[0].datum_type
            && op.op.operating_dt == input_facts[1].datum_type
        {
            if let Some(mmm) = tract_linalg::ops().mmm(
                op.operating_dt,
                Some(m as usize),
                Some(k as usize),
                Some(n as usize),
            ) {
                if mmm.quality() == ImplementationQuality::ManuallyOptimized {
                    return Ok((
                        ModePicker::Single,
                        mmm.packings()[0].0.clone(),
                        vec![(mmm, 0, None)],
                    ));
                }
            }
        };
    }

    let mut impls = list_impls(model, node, op)?;
    ensure!(impls.len() > 0);
    fn score(mmm: &dyn MatMatMul) -> isize {
        -(mmm.quality().cost() as isize * 1000) + mmm.dynamic_boost()
    }
    let wanted_quality = impls.iter().map(|(mmm, _, _)| score(&**mmm)).max().unwrap();
    impls.retain(|(mmm, _, _)| score(&**mmm) == wanted_quality);
    if impls.len() == 1 {
        return Ok(single_strat(impls.remove(0)));
    }
    if op.n.is_one() {
        let it =
            impls.into_iter().max_by_key(|(m, _, pe)| (m.nr() == 1, pe.is_none(), m.mr())).unwrap();
        return Ok(single_strat(it));
    }
    if op.n.as_i64().is_some_and(|n| n > 1) {
        let it =
            impls.into_iter().max_by_key(|(m, _, pe)| (pe.is_none(), m.nr() * m.mr())).unwrap();
        return Ok(single_strat(it));
    }
    let mut grouped_by_left_packing = Vec::<(&dyn MMMInputFormat, Vec<_>)>::new();
    'mmm: for (m, p, pe) in &impls {
        let left_packing: &dyn MMMInputFormat =
            pe.as_ref().map(|pe| &*pe.from).unwrap_or(&*m.packings()[*p].0);
        for kit in &mut grouped_by_left_packing {
            if let Some(merged) = kit.0.merge_with(left_packing) {
                kit.0 = merged;
                kit.1.push((m, p, pe));
                continue 'mmm;
            }
        }
        grouped_by_left_packing.push((left_packing, vec![(m, p, pe)]));
    }
    let (p, mmv, mmm) = grouped_by_left_packing
        .iter()
        .map(|(p, kit)| {
            let best_for_mmv =
                kit.iter().max_by_key(|(m, _, pe)| (m.nr() == 1, pe.is_none())).unwrap();
            let best_for_mmm = kit.iter().max_by_key(|(m, _, _)| m.nr()).unwrap();
            (p, best_for_mmv, best_for_mmm)
        })
        .max_by_key(|(_, mmv, mmm)| {
            (mmv.0.nr() == 1 && mmm.0.nr() > 1, mmv.2.is_none(), mmm.0.mr(), mmm.0.nr())
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

pub fn list_impls(
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSumMatMul,
) -> TractResult<Vec<Impl>> {
    let (a_fact, b_fact) = model.node_input_facts(node.id)?.into_iter().collect_tuple().unwrap();
    let a_dt = a_fact.datum_type;
    let b_dt = b_fact.datum_type;

    let a_weight: WeightType = if let Some(of) = a_fact.opaque_fact() {
        if let Some(bqf) = of.downcast_ref::<BlockQuantFact>() {
            WeightType::BlockQuant(bqf.format.clone())
        } else {
            bail!("Can not translate to matmul operand {a_fact:?}");
        }
    } else {
        a_dt.into()
    };

    let impls = tract_linalg::ops()
        .mmm_impls()
        .iter()
        .filter(|mmm| {
            op.acceptable_accumulators().contains(&mmm.internal_type())
                && mmm.stores().contains(&op.operating_dt.unquantized())
        })
        .flat_map(move |mmm| {
            mmm.packings().iter().enumerate().map(|(ix, p)| (mmm.clone(), ix, &p.0, &p.1))
        })
        .filter_map(|(m, p, pa, pb)| {
            if pb.precursor().as_dt().is_none_or(|dt| dt != b_dt.unquantized()) {
                return None;
            }
            if pa.precursor() == a_weight {
                Some((m, p, None))
            } else {
                tract_linalg::ops()
                    .panel_extractors()
                    .iter()
                    .find(|pe| pe.from.precursor() == a_weight && pe.to.same_as(&**pa))
                    .map(|pe| (m, p, Some(pe.clone())))
            }
        })
        .collect_vec();
    Ok(impls)
}
