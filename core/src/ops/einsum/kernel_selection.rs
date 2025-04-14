#![allow(clippy::type_complexity)]
use tract_itertools::Itertools;
use tract_linalg::block_quant::BlockQuantFact;
use tract_linalg::mmm::{MMMInputFormat, MMMInputValue, MatMatMul, PanelExtractor};
use tract_linalg::pack::PackedFormat;
use tract_linalg::WeightType;

use crate::internal::*;
use crate::ops::matmul::pack::OptMatMulPack;
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
                return Ok((ModePicker::Single, vec![(mmm, 0, None)]));
            }
        };
    }

    let mut impls = list_impls(model, node, op)?;
    let wanted_quality = impls.iter().map(|(mmm, _, _)| mmm.quality().cost()).min().unwrap();
    impls.retain(|(mmm, _, _)| mmm.quality().cost() == wanted_quality);
    if impls.len() == 1 {
        return Ok((ModePicker::Single, impls));
    }
    dbg!(impls);
    todo!();
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

pub fn wire_packing(
    patch: &mut TypedModelPatch,
    prefix: &str,
    operands: &[OutletId],
    op: &EinSumMatMul,
) -> TractResult<(
    OutletId,
    OutletId,
    Vec<(Box<dyn MatMatMul>, usize, Option<PanelExtractor>)>,
    ModePicker,
)> {
    panic!();
    let a_fact = patch.outlet_fact(operands[0])?.clone();
    let b_fact = patch.outlet_fact(operands[1])?.clone();
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

    let mut able = tract_linalg::ops()
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
                Some((m, p, pa, pb, None))
            } else {
                tract_linalg::ops()
                    .panel_extractors()
                    .iter()
                    .find(|pe| pe.from.precursor() == a_weight && pe.to.same_as(&**pa))
                    .map(|pe| (m, p, pa, pb, Some(pe)))
            }
        })
        .collect_vec();

    if able.is_empty() {
        bail!("Not matmul implementation found");
    }
    let wanted_quality = able.iter().map(|(mmm, _, _, _, _)| mmm.quality().cost()).min().unwrap();
    able.retain(|(mmm, _, _, _, _)| mmm.quality().cost() == wanted_quality);

    // "simple" kernel selection
    let (mmm, p, pa, pb, pe) = able
        .into_iter()
        .min_by_key(|(mmm, _, _, _, _pe)| {
            1_000_000_000 + mmm.quality().cost() * 10_000 - mmm.mr() * mmm.nr()
        })
        .unwrap();

    dbg!(&mmm, p, pa, pb, pe);

    let pa = patch.wire_node(
        format!("{prefix}.pack_a"),
        OptMatMulPack {
            k_axis: op.a_k(),
            mn_axis: op.a_m(),
            packers: vec![pa.downcast_ref::<PackedFormat>().unwrap().clone()],
            mode_picker: ModePicker::Single,
        },
        &[operands[0]],
    )?[0];

    let pb = patch.wire_node(
        format!("{prefix}.pack_b"),
        OptMatMulPack {
            k_axis: op.b_k(),
            mn_axis: op.b_n(),
            packers: vec![pb.downcast_ref::<PackedFormat>().unwrap().clone()],
            mode_picker: ModePicker::Single,
        },
        &[operands[1]],
    )?[0];

    Ok((pa, pb, vec![(mmm, p, None)], ModePicker::Single))
}

pub fn wire_prepacked(
    patch: &mut TypedModelPatch,
    prefix: &str,
    op: &EinSumMatMul,
    a: OutletId,
    b: OutletId,
) -> TractResult<(
    OutletId,
    OutletId,
    Vec<(Box<dyn MatMatMul>, usize, Option<PanelExtractor>)>,
    ModePicker,
)> {
    ensure!(patch.outlet_fact(a)?.konst.is_some());
    let b_dt = patch.outlet_fact(b)?.datum_type.unquantized();

    let a_konst = patch.outlet_fact(a)?.konst.as_ref().unwrap();
    // preemptive packing ?
    let prepack = a_konst
        .to_scalar::<Opaque>()
        .ok()
        .and_then(|opaq| opaq.0.downcast_ref::<Box<dyn MMMInputValue>>());
    ensure!(prepack.is_some());
    let prepack = prepack.unwrap();
    let (mode_picker, configs) = if let Some((mmm, pack)) = op
        .n
        .as_i64()
        .and_then(|n| {
            tract_linalg::ops().mmm(
                op.operating_dt,
                Some(prepack.mn()),
                Some(prepack.k()),
                Some(n as usize),
            )
        })
        .and_then(|mmm| {
            mmm.packings()
                .iter()
                .enumerate()
                .find(|(_, (a, pb))| a.same_as(prepack.format()) && pb.precursor() == b_dt.into())
                .map(|(ix, _)| (mmm.clone(), ix))
        }) {
        (ModePicker::Single, vec![(mmm, pack, None)])
    } else {
        let mmms = tract_linalg::ops()
            .mmm_impls()
            .iter()
            .filter(|mmm| {
                op.acceptable_accumulators().contains(&mmm.internal_type())
                    && mmm.stores().contains(&op.operating_dt.unquantized())
            })
            .flat_map(move |mmm| {
                mmm.packings().iter().enumerate().map(|(ix, p)| (mmm.clone(), ix, &p.0, &p.1))
            })
            .filter_map(|(mmm, packing, pa, pb)| {
                if pb.precursor().as_dt().is_some_and(|dt| dt != b_dt) {
                    None
                } else if prepack.format().same_as(&**pa) {
                    Some((mmm, packing, None))
                } else {
                    tract_linalg::ops()
                        .panel_extractors()
                        .iter()
                        .find(|pe| prepack.format().same_as(&*pe.from) && pe.to.same_as(&**pa))
                        .map(|pe| (mmm, packing, Some(pe)))
                }
            })
            .collect_vec();
        ensure!(
            !mmms.is_empty(),
            "No kernel found for {op:?} on {:?} {:?}",
            patch.outlet_fact(a)?,
            patch.outlet_fact(b)?
        );
        let mmv = mmms
            .iter()
            .min_by_key(|(mmm, _packing, pe)| {
                mmm.quality().cost() * 10000 + 100 * mmm.nr() + pe.is_some() as usize
            })
            .unwrap();
        let mmm = mmms
            .iter()
            .min_by_key(|(mmm, _packing, pe)| {
                1_000_000 + mmm.quality().cost() * 10000 + pe.is_some() as usize - mmm.nr() * 100
            })
            .unwrap();

        (
            ModePicker::VecVsMat,
            [mmv, mmm]
                .iter()
                .map(|(mmm, packing, pe)| (mmm.clone(), *packing, pe.cloned()))
                .collect_vec(),
        )
    };

    let b_packers = configs
        .iter()
        .map(|(mmm, packing, _pe)| {
            mmm.packings()[*packing].1.downcast_ref::<PackedFormat>().unwrap().clone()
        })
        .collect_vec();
    let pb = patch.wire_node(
        format!("{prefix}.pack_b"),
        OptMatMulPack {
            k_axis: op.b_k(),
            mn_axis: op.b_n(),
            packers: b_packers,
            mode_picker: mode_picker.clone(),
        },
        &[b],
    )?[0];

    Ok((a, pb, configs, mode_picker))
}
