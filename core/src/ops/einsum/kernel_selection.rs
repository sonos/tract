#![allow(clippy::type_complexity)]
use tract_itertools::Itertools;
use tract_linalg::frame::block_quant::PackedBlockQuantFormat;
use tract_linalg::frame::PackedFormat;
use tract_linalg::mmm::panel_extract::PanelExtractor;
use tract_linalg::mmm::{KitDatumType, MMMInputValue, MatMatMul, WeightType};

use crate::internal::*;
use crate::ops::matmul::de_block_quant::BlockQuantValue;
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

    if a_fact.konst.is_some() && op.n.as_i64().is_none() {
        let a = a_fact.konst.unwrap();
        return wire_linear(patch, prefix, op, &a, operands[1]);
    }

    // "simple" kernel selection
    let mmm = tract_linalg::ops()
        .mmm(op.operating_dt, op.m.to_usize().ok(), op.k.to_usize().ok(), op.n.to_usize().ok())
        .unwrap();
    let mode_picker = ModePicker::Single;
    let (packing, pa, pb) = mmm
        .packings()
        .iter()
        .enumerate()
        .filter_map(|(ix, p)| {
            Some((ix, p.0.downcast_ref::<PackedFormat>()?, p.1.downcast_ref::<PackedFormat>()?))
        })
        .find(|(_ix, pa, pb)| pa.dt == a_dt.unquantized() && pb.dt == b_dt.unquantized())
        .with_context(|| format!("No packing for {mmm:?} with inputs {a_dt:?} and {b_dt:?}"))?;
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

    Ok((pa, pb, vec![(mmm, packing, None)], mode_picker))
}

pub fn wire_linear(
    patch: &mut TypedModelPatch,
    prefix: &str,
    op: &EinSumAnnotatedAsMatMul,
    a: &Arc<Tensor>,
    b: OutletId,
) -> TractResult<(
    OutletId,
    OutletId,
    Vec<(Box<dyn MatMatMul>, usize, Option<PanelExtractor>)>,
    ModePicker,
)> {
    let a_as_bqv = a.to_scalar::<Opaque>().ok().and_then(|a| a.downcast_ref::<BlockQuantValue>());
    let weight = if let Some(a_payload) = a_as_bqv {
        WeightType::BlockQuant(a_payload.fact.format.clone())
    } else {
        WeightType::Plain(a.datum_type())
    };
    let b_fact = patch.outlet_fact(b)?;
    let accumulator = if op.operating_dt.is_integer() {
        KitDatumType::I32
    } else if op.operating_dt == f16::datum_type() && tract_linalg::has_fp16() {
        KitDatumType::F16
    } else {
        KitDatumType::F32
    };
    let activation = match b_fact.datum_type {
        DatumType::F16 => KitDatumType::F16,
        DatumType::F32 => KitDatumType::F32,
        _ => todo!(),
    };
    let kit = tract_linalg::ops()
        .mmm_kits()
        .iter()
        .filter(|kit| {
            kit.weight == weight && kit.accumulator == accumulator && kit.activation == activation
        })
        .min_by_key(|kit| kit.generic_fallback as usize)
        .with_context(|| format!("No kit found for matmul {weight:?} {accumulator:?} {activation:?}"))?;
    let configs = [kit.item_for_mv(), kit.item_for_squarish()];
    let packed: Box<dyn MMMInputValue> = if let Some(a_payload) = a_as_bqv {
        let packed = kit
            .static_packer
            .downcast_ref::<PackedBlockQuantFormat>()
            .unwrap()
            .pack(&a_payload.value, op.k.to_usize().unwrap())?;
        Box::new(packed) as _
    } else {
        kit.static_packer.prepare_tensor(a, op.a_k(), op.a_m())?
    };
    let konst = tensor0(Opaque::from(packed));
    let pa = patch.add_const(format!("{prefix}.pack_a"), konst)?;

    let packers = configs
        .iter()
        .map(|conf| {
            conf.mmm.packings()[conf.packing].1.downcast_ref::<PackedFormat>().unwrap().clone()
        })
        .collect_vec();
    let pb = patch.wire_node(
        format!("{prefix}.pack_b"),
        OptMatMulPack {
            k_axis: op.b_k(),
            mn_axis: op.b_n(),
            packers,
            mode_picker: ModePicker::VecVsMat,
        },
        &[b],
    )?[0];

    Ok((
        pa,
        pb,
        configs
            .iter()
            .map(|cf| (cf.mmm.clone(), cf.packing, cf.weight_panel_extractor.clone()))
            .collect_vec(),
        ModePicker::VecVsMat,
    ))
}
