//! Tiny deterministic interchange fixtures for downstream inference integration.
use std::{fs::File, io::Write, path::PathBuf, sync::Arc};
use tract_ndarray::Dimension;
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::{
    array::{Gather, TypedConcat},
    konst::Const,
};
use tract_nnef::tract_core::tract_linalg::block_quant::{
    BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0,
};
use tract_transformers::{
    WithTractTransformers,
    ops::moe_ffn::{ExpertLayout, GateMode, MoeFfn},
};

fn weight(shape: &[usize], salt: usize) -> Tensor {
    tract_ndarray::ArrayD::from_shape_fn(shape, |ix| {
        let index = ix.slice().iter().fold(salt, |a, b| a * 37 + b);
        ((index % 101) as f32 - 50.0) / 100.0
    })
    .into_tensor()
}

fn expert(model: &mut TypedModel, name: &str, data: Tensor, quant: bool) -> TractResult<OutletId> {
    if !quant {
        return model.add_const(name, data);
    }
    let shape = data.shape().to_vec();
    let k = *shape.last().unwrap();
    let m = shape[..shape.len() - 1].iter().product();
    let bytes = Q4_0.quant_f32(data.try_as_plain_ram()?.as_slice::<f32>()?)?;
    let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(bytes))?;
    let tensor = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), &shape));
    let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
    Ok(model.wire_node(name, Const::new_with_exotic_fact(tensor, Box::new(fact))?, &[])?[0])
}

fn main() -> TractResult<()> {
    let dir = PathBuf::from(std::env::args().nth(1).context("output directory required")?);
    std::fs::create_dir_all(&dir)?;
    for variant in ["plain", "q40", "mixed", "clamped"] {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let p = model.symbols.sym("P");
        let tokens = model.add_source("tokens", i64::fact([1.to_dim(), s.to_dim()]))?;
        let cache = model.add_source("cache", f32::fact([1.to_dim(), p.to_dim(), 32.to_dim()]))?;
        let table = model.add_const("embedding", weight(&[32, 32], 1))?;
        let x = model.wire_node("embed", Gather::new(0), &[table, tokens])?[0];
        let wg = model.add_const("router", weight(&[2, 32], 2))?;
        let w1 = expert(&mut model, "w1", weight(&[2, 32, 32], 3), variant != "plain")?;
        let down = weight(&[2, 32, 32], 4);
        let down = if variant == "mixed" { down.cast_to::<f16>()?.into_owned() } else { down };
        let w2 = expert(&mut model, "w2", down, variant == "q40" || variant == "clamped")?;
        let w3 = expert(&mut model, "w3", weight(&[2, 32, 32], 5), variant != "plain")?;
        let clamped = variant == "clamped";
        let op = MoeFfn {
            k: 2,
            activation: "silu".into(),
            gate: GateMode::SoftmaxTopk,
            has_w3: true,
            has_wg_bias: clamped,
            has_w1_bias: clamped,
            has_w3_bias: clamped,
            has_w2_bias: clamped,
            act_alpha_bits: clamped.then_some(1.7f32.to_bits()),
            act_limit_bits: clamped.then_some(2.0f32.to_bits()),
            expert_layout: ExpertLayout::Linear,
        };
        let mut inputs = tvec!(x, wg, w1, w2, w3);
        if clamped {
            for (name, shape) in
                [("bg", vec![2]), ("b1", vec![2, 32]), ("b3", vec![2, 32]), ("b2", vec![2, 32])]
            {
                inputs.push(model.add_const(name, weight(&shape, 6))?);
            }
        }
        let logits = model.wire_node("moe", op, &inputs)?[0];
        // A history sum keeps cache feedback observable without modeling attention.
        let cache_out = model.wire_node("cache_out", TypedConcat { axis: 1 }, &[cache, x])?[0];
        let history = model.wire_node(
            "history",
            tract_nnef::tract_core::ops::nn::Reduce {
                axes: tvec!(1),
                reducer: tract_nnef::tract_core::ops::nn::Reducer::Sum,
            },
            &[cache_out],
        )?[0];
        let logits = model.wire_node(
            "logits",
            tract_nnef::tract_core::ops::math::add(),
            &[logits, history],
        )?[0];
        model.select_output_outlets(&[logits, cache_out])?;
        tract_nnef::nnef()
            .with_tract_transformers()
            .write_to_tar(&model, File::create(dir.join(format!("{variant}.nnef.tar")))?)?;
        let plan = SimplePlan::new(model)?;
        let tokens =
            tract_ndarray::Array2::from_shape_vec((1, 32), (0..32i64).collect())?.into_tensor();
        let cache = Tensor::zero::<f32>(&[1, 0, 32])?;
        let output = plan.run(tvec!(tokens.into_tvalue(), cache.into_tvalue()))?;
        let mut expected = File::create(dir.join(format!("{variant}.expected")))?;
        let embeddings = weight(&[32, 32], 1);
        let embeddings = embeddings.try_as_plain_ram()?;
        for (i, value) in output[0].try_as_plain_ram()?.as_slice::<f32>()?.iter().enumerate() {
            // Store per-token MoE logits; consumers add their own history sum.
            let sum: f32 = embeddings.as_slice::<f32>()?.chunks_exact(32).map(|r| r[i % 32]).sum();
            let value = value - sum;
            writeln!(expected, "{value:.9}")?;
        }
        println!("{variant}: exported tiny model and reference logits");
    }
    Ok(())
}
