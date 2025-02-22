#![allow(clippy::type_complexity)]
use dyn_clone::clone_box;
use tract_itertools::Itertools;
use tract_linalg::mmm::{MMMInputFormat, MMMInputValue, MatMatMul, PanelExtractor};
use tract_linalg::pack::PackedFormat;

use crate::internal::*;
use crate::ops::matmul::pack::OptMatMulPack;
use crate::ops::matmul::ModePicker;

use super::optimize::EinSumAnnotatedAsMatMul;

pub fn wire_packing(
    patch: &mut TypedModelPatch,
    prefix: &str,
    operands: &[OutletId],
    op: &EinSumAnnotatedAsMatMul,
) -> TractResult<(
    OutletId,
    OutletId,
    Vec<(Box<dyn MatMatMul>, usize, Option<PanelExtractor>)>,
    ModePicker,
)> {
    let a_fact = patch.outlet_fact(operands[0])?.clone();
    let b_fact = patch.outlet_fact(operands[1])?.clone();
    let a_dt = a_fact.datum_type;
    let b_dt = b_fact.datum_type;

    if a_fact.konst.is_some() && a_fact.datum_type.is_opaque() {
        return wire_prepacked(patch, prefix, op, operands[0], operands[1])
            .context("wire_prepacked");
    }

    let acc = if op.operating_dt.is::<f16>() && !tract_linalg::has_fp16() {
        f32::datum_type()
    } else {
        op.operating_dt
    };

    let (mmm, packing, pa, pb) = tract_linalg::ops()
        .all_mmm_and_packings()
        .filter(|(m, _, a, b)| {
            m.internal_type() == acc
                && a.precursor() == a_dt.into()
                && b.precursor() == b_dt.into()
                && a.is::<PackedFormat>()
                && b.is::<PackedFormat>()
        })
        .min_by_key(|(mmm, _ix, _a, _b)| 10_000_000 + mmm.quality().cost() - mmm.mr() * mmm.nr())
        .unwrap();

    let pa = pa.downcast_ref::<PackedFormat>().unwrap();
    let pb = pb.downcast_ref::<PackedFormat>().unwrap();

    let mode_picker = ModePicker::Single;
    let pa = patch.wire_node(
        format!("{prefix}.pack_a"),
        OptMatMulPack {
            k_axis: op.a_k(),
            mn_axis: op.a_m(),
            packers: vec![pa.clone()],
            mode_picker: ModePicker::Single,
        },
        &[operands[0]],
    )?[0];

    let pb = patch.wire_node(
        format!("{prefix}.pack_b"),
        OptMatMulPack {
            k_axis: op.b_k(),
            mn_axis: op.b_n(),
            packers: vec![pb.clone()],
            mode_picker: ModePicker::Single,
        },
        &[operands[1]],
    )?[0];

    Ok((pa, pb, vec![(clone_box(mmm), packing, None)], mode_picker))
}

pub fn wire_prepacked(
    patch: &mut TypedModelPatch,
    prefix: &str,
    op: &EinSumAnnotatedAsMatMul,
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

    let mmms = tract_linalg::ops()
        .mmm_impls()
        .iter()
        .filter(|mmm| op.acceptable_accumulators().contains(&mmm.internal_type()))
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

    let configs = [mmv, mmm]
        .iter()
        .map(|(mmm, packing, pe)| (mmm.clone(), *packing, pe.cloned()))
        .collect_vec();

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
            mode_picker: ModePicker::VecVsMat,
        },
        &[b],
    )?[0];

    Ok((a, pb, configs, ModePicker::VecVsMat))
}
