use tract_linalg::frame::block_quant::{BlockQuant, PackedBlockQuantFormat};
use tract_linalg::frame::PackedFormat;
use tract_linalg::mmm::{MMMInputValue, MatMatMul};

use crate::internal::*;
use crate::ops::matmul::de_block_quant::{BlockQuantFact, BlockQuantValue};
use crate::ops::matmul::pack::MatMatMulPack;

use super::optimize::EinSumAnnotatedAsMatMul;

pub fn wire_packing(
    model: &TypedModel,
    patch: &mut TypedModelPatch,
    prefix: &str,
    operands: &[OutletId],
    op: &EinSumAnnotatedAsMatMul,
) -> TractResult<(OutletId, OutletId, Vec<(Box<dyn MatMatMul>, usize)>)> {
    let a_fact = patch.outlet_fact(operands[0])?.clone();
    let b_fact = patch.outlet_fact(operands[1])?.clone();
    let a_dt = a_fact.datum_type;
    let b_dt = b_fact.datum_type;

    if a_fact.konst.is_some()
        && a_fact.datum_type.is_opaque()
        && a_fact.opaque_fact.as_ref().is_some_and(|of| of.is::<BlockQuantFact>())
        && op.op.operating_dt.is_float()
    {
        dbg!("block quant");
        let weights =
            a_fact.opaque_fact.as_ref().unwrap().downcast_ref::<BlockQuantFact>().unwrap();
        return wire_block_quant_matmul(model, patch, prefix, op, operands, &*weights.format, b_dt);
    }

    dbg!("simple one");
    // "simple" kernel selection
    let mmm = tract_linalg::ops()
        .mmm(op.op.operating_dt, op.m.to_usize().ok(), op.k.to_usize().ok(), op.n.to_usize().ok())
        .unwrap();
    let (packing, pa, pb) = mmm
        .packings()
        .into_iter()
        .enumerate()
        .filter_map(|(ix, p)| {
            Some((ix, p.0.downcast_ref::<PackedFormat>()?, p.1.downcast_ref::<PackedFormat>()?))
        })
        .find(|(_ix, pa, pb)| pa.dt == a_dt.unquantized() && pb.dt == b_dt.unquantized())
        .with_context(|| format!("No packing for {mmm:?} with inputs {a_dt:?} and {b_dt:?}"))?;
    let pa = patch.wire_node(
        format!("{prefix}.pack_a"),
        MatMatMulPack { k_axis: op.a_k(), mn_axis: op.a_m(), packers: vec![pa.clone()] },
        &[operands[0]],
    )?[0];

    let pb = patch.wire_node(
        format!("{prefix}.pack_b"),
        MatMatMulPack { k_axis: op.b_k(), mn_axis: op.b_n(), packers: vec![pb.clone()] },
        &[operands[1]],
    )?[0];

    Ok((pa, pb, vec![(mmm, packing)]))
}

fn wire_block_quant_matmul(
    model: &TypedModel,
    patch: &mut TypedModelPatch,
    prefix: &str,
    op: &EinSumAnnotatedAsMatMul,
    operands: &[OutletId],
    a_bq: &dyn BlockQuant,
    b_dt: DatumType,
) -> TractResult<(OutletId, OutletId, Vec<(Box<dyn MatMatMul>, usize)>)> {
    let m = op.m.to_usize().expect("m is expected to be an integer");
    if model.symbols.all_scenarios().into_iter().count() < 2 {
        if op.n.is_one() {
            return wire_block_quant_matvec(patch, prefix, op, operands, m, a_bq, b_dt);
        } else {
            todo!("Missing strategy for squarish matmat with blockquant input");
        }
    } else {
        let (pa, pb, mut mmv) =
            wire_block_quant_matvec(patch, prefix, op, operands, m, a_bq, b_dt)?;
        let mr = mmv[0].0.mr();
        let matching_matmat = tract_linalg::ops()
            .mmm_impls()
            .into_iter()
            .filter(|mm| mm.mr() == mr && mm.internal_type().is_float())
            .max_by_key(|mm| mm.nr())
            .unwrap();

        let alternative_b_packing = matching_matmat.packings()[0]
            .1
            .downcast_ref::<PackedFormat>()
            .context("First B packing should be trivial")?
            .clone();
        patch
            .node_mut(pb.node)
            .op_as_mut::<MatMatMulPack>()
            .context("Expected MatMatMulPack on B")?
            .packers
            .push(alternative_b_packing);

        mmv.push((matching_matmat.clone(), 0));
        return Ok((pa, pb, mmv));
    }
}

fn wire_block_quant_matvec(
    patch: &mut TypedModelPatch,
    prefix: &str,
    op: &EinSumAnnotatedAsMatMul,
    operands: &[OutletId],
    m: usize,
    a_bq: &dyn BlockQuant,
    b_dt: DatumType,
) -> TractResult<(OutletId, OutletId, Vec<(Box<dyn MatMatMul>, usize)>)> {
    let mut options: Vec<(&Box<dyn MatMatMul>, usize, &PackedBlockQuantFormat, &PackedFormat)> =
        vec![];
    for imp in tract_linalg::ops().mmm_impls() {
        if imp.nr() > 1 {
            continue;
        }
        for (packing, (pack_a, pack_b)) in imp.packings().iter().enumerate() {
            if let (Some(pa), Some(pb)) = (
                pack_a.downcast_ref::<PackedBlockQuantFormat>(),
                pack_b.downcast_ref::<PackedFormat>(),
            ) {
                if pa.bq.same_as(&*a_bq) && pb.dt == b_dt {
                    options.push((imp, packing, pa, pb));
                }
            }
        }
    }
    ensure!(options.len() > 0, "should always have at least a generic impl");
    let (mmm, packing, pa, pb) = options
        .into_iter()
        .min_by_key(|a| ((m as usize).divceil(a.0.mr())) * (a.0.mr() + 100))
        .unwrap();
    let value = patch.outlet_fact(operands[0])?.konst.as_ref().context("A should be a const")?;
    let value = value
        .to_scalar::<Opaque>()?
        .downcast_ref::<BlockQuantValue>()
        .context("A should be a BlockQuantValue")?;
    let packed = pa.pack(&value.value, op.k.to_usize()?)?;
    let konst = tensor0(Opaque::from(Box::new(packed) as Box<dyn MMMInputValue>));

    let pa = patch.add_const(format!("{prefix}.pack_a"), konst)?;
    let pb = patch.wire_node(
        format!("{prefix}.pack_b"),
        MatMatMulPack { k_axis: op.b_k(), mn_axis: op.b_n(), packers: vec![pb.clone()] },
        &[operands[1]],
    )?[0];

    Ok((pa, pb, vec![(mmm.clone(), packing)]))
}
