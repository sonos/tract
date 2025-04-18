#![allow(clippy::type_complexity)]
use tract_itertools::Itertools;
use tract_linalg::block_quant::BlockQuantFact;
use tract_linalg::mmm::{ImplementationQuality, MMMInputFormat, MatMatMul, PanelExtractor};
use tract_linalg::WeightType;

use crate::internal::*;
use crate::ops::matmul::ModePicker;

use super::einsum_matmul::EinSumMatMul;

pub type Impl = (Box<dyn MatMatMul>, usize, Option<PanelExtractor>);
pub type Strat = (ModePicker, Vec<Impl>);

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
                    return Ok((ModePicker::Single, vec![(mmm, 0, None)]));
                }
            }
        };
    }

    let mut impls = list_impls(model, node, op)?;
    ensure!(impls.len() > 0);
    let wanted_quality = impls.iter().map(|(mmm, _, _)| mmm.quality().cost()).min().unwrap();
    impls.retain(|(mmm, _, _)| mmm.quality().cost() == wanted_quality);
    if impls.len() == 1 {
        return Ok((ModePicker::Single, impls));
    }
    if op.n.is_one() {
        let it =
            impls.into_iter().max_by_key(|(m, _, pe)| (m.nr() == 1, pe.is_none(), m.mr())).unwrap();
        return Ok((ModePicker::Single, vec![it]));
    }
    if op.n.as_i64().is_some_and(|n| n > 1) {
        let it =
            impls.into_iter().max_by_key(|(m, _, pe)| (pe.is_none(), m.nr() * m.mr())).unwrap();
        return Ok((ModePicker::Single, vec![it]));
    }
    let mut index = HashMap::<String, Vec<_>>::new();
    for (m, p, pe) in &impls {
        let key = pe.as_ref().map(|pe| &pe.from).unwrap_or(&m.packings()[*p].0).to_string();
        index.entry(key).or_default().push((m, p, pe));
    }
    let (mmv, mmm) = index
        .values()
        .map(|kit| {
            let best_for_mmv =
                kit.iter().max_by_key(|(m, _, pe)| (m.nr() == 1, pe.is_none())).unwrap();
            let best_for_mmm = kit.iter().max_by_key(|(m, _, _)| m.nr()).unwrap();
            (best_for_mmv, best_for_mmm)
        })
        .max_by_key(|(mmv, mmm)| (mmv.2.is_none(), mmm.0.mr()))
        .unwrap();
    Ok((
        ModePicker::VecVsMat,
        vec![(mmv.0.clone(), *mmv.1, mmv.2.clone()), (mmm.0.clone(), *mmm.1, mmm.2.clone())],
    ))
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
            if !pb.precursor().as_dt().is_some_and(|dt| dt == b_dt.unquantized()) {
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
