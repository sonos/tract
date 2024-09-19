use tract_itertools::Itertools;
use tract_linalg::frame::block_quant::PackedBlockQuantFormat;
use tract_linalg::frame::PackedFormat;
use tract_linalg::mmm::MatMatMul;

use crate::internal::*;
use crate::ops::matmul::de_block_quant::BlockQuantValue;

pub fn select_kernel_and_packing(
    model: &TypedModel,
    node: &TypedNode,
    m: &TDim,
    k: &TDim,
    n: &TDim,
    operating_dt: DatumType,
) -> TractResult<Vec<(Box<dyn MatMatMul>, usize)>> {
    let input_facts = model.node_input_facts(node.id)?;
    let a_dt = input_facts[0].datum_type;
    let b_dt = input_facts[1].datum_type;

    if let Some(bqv) = input_facts[0]
        .konst
        .as_ref()
        .filter(|t| t.volume() == 1 && t.datum_type().is_opaque())
        .and_then(|t| t.as_slice::<Opaque>().unwrap()[0].downcast_ref::<BlockQuantValue>())
    {
        let all_n_values = model
            .symbols
            .all_scenarios()
            .into_iter()
            .map(|(s, _)| n.eval_with_scenario(&s))
            .collect_vec();
        let _need_matvec = all_n_values.iter().any(|n| n.is_one());
        //        println!("{m} {n} {all_n_values:?} {need_matvec:?}");
        let mut options: Vec<(&Box<dyn MatMatMul>, usize)> = vec![];
        for imp in tract_linalg::ops().mmm_impls() {
            for (packing, (pack_a, pack_b)) in imp.packings().iter().enumerate() {
                //                println!("{imp:?} {packing} {pack_a} {pack_b}");
                if let (Some(input), Some(b)) = (
                    pack_a.downcast_ref::<PackedBlockQuantFormat>(),
                    pack_b.downcast_ref::<PackedFormat>(),
                ) {
                    if input.bq.same_as(&*bqv.fact.format) && b.dt == b_dt {
                        options.push((imp, packing));
                    }
                }
            }
        }
        if options.len() > 0 {
            let pair = if let (Some(m), Some(n)) = (m.as_i64(), n.as_i64()) {
                options
                    .iter()
                    .min_by_key(|a| {
                        ((m as usize).divceil(a.0.mr()) * (n as usize).divceil(a.0.nr()))
                            * (a.0.mr() * a.0.nr() + 100)
                    })
                    .unwrap()
            } else {
                options.iter().max_by_key(|a| a.0.mr() * a.0.nr()).unwrap()
            };
            return Ok(vec!((pair.0.clone(), pair.1)));
        }
    }

    // "simple" kernel selection
    let mmm = tract_linalg::ops()
        .mmm(operating_dt, m.to_usize().ok(), k.to_usize().ok(), n.to_usize().ok())
        .unwrap();
    let packing = mmm
        .packings()
        .iter()
        .position(|p| {
            p.0.downcast_ref::<PackedFormat>()
                .is_some_and(|pf| pf.dt.unquantized() == a_dt.unquantized())
                && p.1
                    .downcast_ref::<PackedFormat>()
                    .is_some_and(|pf| pf.dt.unquantized() == b_dt.unquantized())
        })
        .with_context(|| format!("No packing for {mmm:?} with inputs {a_dt:?} and {b_dt:?}"))?;
    Ok(vec!((mmm, packing)))
}
