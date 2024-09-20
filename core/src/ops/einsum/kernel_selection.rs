use tract_itertools::Itertools;
use tract_linalg::frame::block_quant::PackedBlockQuantFormat;
use tract_linalg::frame::PackedFormat;
use tract_linalg::mmm::MatMatMul;

use crate::internal::*;
use crate::ops::matmul::de_block_quant::BlockQuantValue;
use crate::ops::matmul::pack::Packer;

#[derive(Debug)]
pub struct Strategy {
    pub static_packing_for_a: Packer,
    pub static_packing_for_b: Packer,
    pub scenarios: Vec<(Box<dyn MatMatMul>, usize, Packer, Packer)>,
}

pub fn select_kernel_and_packing(
    model: &TypedModel,
    node: &TypedNode,
    m: &TDim,
    k: &TDim,
    n: &TDim,
    operating_dt: DatumType,
) -> TractResult<Strategy> {
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
        println!("{m} {n} {all_n_values:?} {_need_matvec:?}");
        let mut options: Vec<(&Box<dyn MatMatMul>, usize, &PackedBlockQuantFormat, &PackedFormat)> =
            vec![];
        for imp in tract_linalg::ops().mmm_impls() {
            for (packing, (pack_a, pack_b)) in imp.packings().iter().enumerate() {
                println!("{imp:?} {packing} {pack_a} {pack_b}");
                if let (Some(pa), Some(pb)) = (
                    pack_a.downcast_ref::<PackedBlockQuantFormat>(),
                    pack_b.downcast_ref::<PackedFormat>(),
                ) {
                    if pa.bq.same_as(&*bqv.fact.format) && pb.dt == b_dt {
                        options.push((imp, packing, pa, pb));
                    }
                }
            }
        }
        if options.len() > 0 {
            let (mmm, packing, pa, pb) = if let (Some(m), Some(n)) = (m.as_i64(), n.as_i64()) {
                options
                    .into_iter()
                    .min_by_key(|a| {
                        ((m as usize).divceil(a.0.mr()) * (n as usize).divceil(a.0.nr()))
                            * (a.0.mr() * a.0.nr() + 100)
                    })
                    .unwrap()
            } else {
                options.into_iter().max_by_key(|a| a.0.mr() * a.0.nr()).unwrap()
            };
            return Ok(Strategy {
                static_packing_for_a: Packer::PackBlockQuant(pa.clone()),
                static_packing_for_b: Packer::Regular(pb.clone()),
                scenarios: vec![(mmm.clone(), packing, Packer::Identity, Packer::Identity)],
            });
        }
    }

    // "simple" kernel selection
    let mmm = tract_linalg::ops()
        .mmm(operating_dt, m.to_usize().ok(), k.to_usize().ok(), n.to_usize().ok())
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
    Ok(Strategy {
        static_packing_for_a: Packer::Regular(pa.clone()),
        static_packing_for_b: Packer::Regular(pb.clone()),
        scenarios: vec![(mmm, packing, Packer::Identity, Packer::Identity)],
    })
}
