use std::sync::Arc;

use super::*;
use tract_nnef::tract_core::tract_linalg::block_quant::{
    BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0,
};

fn make_moe_model(
    t_tokens: usize,
    d_model: usize,
    d_hidden: usize,
    num_experts: usize,
    k: usize,
    has_w3: bool,
) -> TractResult<(TypedModel, Tensor)> {
    let mut model = TypedModel::default();

    let x = model.add_source("x", f32::datum_type().fact([t_tokens, d_model]))?;

    // Deterministic pseudo-random weights
    let mut rng_state: u64 = 42;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng()).collect();
        tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
    };

    let wg_data = make_tensor(&[num_experts, d_model], &mut next_f32);
    let w1_data = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
    let w2_data = make_tensor(&[num_experts, d_hidden, d_model], &mut next_f32);

    let wg = model.add_const("wg", wg_data)?;
    let w1 = model.add_const("w1", w1_data)?;
    let w2 = model.add_const("w2", w2_data)?;

    let mut inputs = vec![x, wg, w1, w2];

    if has_w3 {
        let w3_data = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
        let w3 = model.add_const("w3", w3_data)?;
        inputs.push(w3);
    }

    let op = MoeFfn::basic(k, "silu", GateMode::SoftmaxTopk, has_w3);
    let outputs = model.wire_node("moe", op, &inputs)?;
    model.select_output_outlets(&outputs)?;

    // Create input tensor
    let x_data = make_tensor(&[t_tokens, d_model], &mut next_f32);

    Ok((model, x_data))
}

fn add_q40_const(model: &mut TypedModel, name: &str, tensor: Tensor) -> TractResult<OutletId> {
    let shape = tensor.shape().to_vec();
    let k = *shape.last().context("Q40 tensor has no last axis")?;
    ensure!(k % Q4_0.block_len() == 0, "Q40 K axis must be a multiple of 32");
    let m = shape[..shape.len() - 1].iter().product();
    let quant = Q4_0.quant_f32(tensor.try_as_plain_ram()?.as_slice::<f32>()?)?;
    let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant))?;
    let packed = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), &shape));
    let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
    Ok(model.wire_node(name, Const::new_with_exotic_fact(packed, Box::new(fact))?, &[])?[0])
}

fn q40_tensor(tensor: Tensor) -> TractResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let k = *shape.last().context("Q40 tensor has no last axis")?;
    ensure!(k % Q4_0.block_len() == 0, "Q40 K axis must be a multiple of 32");
    let m = shape[..shape.len() - 1].iter().product();
    let quant = Q4_0.quant_f32(tensor.try_as_plain_ram()?.as_slice::<f32>()?)?;
    let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant))?;
    Ok(storage.into_tensor_with_shape(f32::datum_type(), &shape))
}

fn q40_roundtrip(tensor: &Tensor) -> TractResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let k = *shape.last().context("Q40 tensor has no last axis")?;
    ensure!(k % Q4_0.block_len() == 0, "Q40 K axis must be a multiple of 32");
    let quant = Q4_0.quant_f32(tensor.try_as_plain_ram()?.as_slice::<f32>()?)?;
    Q4_0.dequant_f32(&quant)?.into_shape(&shape)
}

fn transpose_expert_last2(tensor: &Tensor) -> TractResult<Tensor> {
    let view = tensor.to_plain_array_view::<f32>()?;
    let transposed = view.permuted_axes(tract_ndarray::IxDyn(&[0, 2, 1]));
    Ok(transposed.into_owned().into_tensor())
}

#[test]
fn test_concat_block_quant_rows_preserves_q40_storage() -> TractResult<()> {
    let lhs_data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 17.0).collect();
    let rhs_data: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) / 19.0).collect();
    let lhs_plain = Tensor::from_shape(&[2, 32], &lhs_data)?;
    let rhs_plain = Tensor::from_shape(&[3, 32], &rhs_data)?;
    let lhs_q40 = q40_tensor(lhs_plain)?;
    let rhs_q40 = q40_tensor(rhs_plain)?;

    let fused = concat_block_quant_rows(&lhs_q40, &rhs_q40)?;
    assert_eq!(fused.shape(), &[5, 32]);
    assert!(fused.storage_as::<BlockQuantStorage>().is_some());

    let expected_plain =
        Tensor::from_shape(&[5, 32], &[lhs_data.as_slice(), rhs_data.as_slice()].concat())?;
    let expected = q40_roundtrip(&expected_plain)?;
    let fused_dequant = block_quant_group_as_2d(&fused, 0)?;
    fused_dequant.close_enough(&expected, Approximation::Approximate)?;
    Ok(())
}

#[test]
fn test_linear_expert_layout_matches_canonical_layout() -> TractResult<()> {
    let mut rng_state: u64 = 9001;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng()).collect();
        tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
    };

    let t_tokens = 8;
    let d_model = 16;
    let d_hidden = 32;
    let num_experts = 4;
    let wg_data = make_tensor(&[num_experts, d_model], &mut next_f32);
    let w1_canonical = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
    let w2_canonical = make_tensor(&[num_experts, d_hidden, d_model], &mut next_f32);
    let w3_canonical = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
    let x_data = make_tensor(&[t_tokens, d_model], &mut next_f32);

    let build = |layout: ExpertLayout,
                 wg_data: Tensor,
                 w1_data: Tensor,
                 w2_data: Tensor,
                 w3_data: Tensor|
     -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact([t_tokens, d_model]))?;
        let wg = model.add_const("wg", wg_data)?;
        let w1 = model.add_const("w1", w1_data)?;
        let w2 = model.add_const("w2", w2_data)?;
        let w3 = model.add_const("w3", w3_data)?;
        let op = MoeFfn::basic_with_layout(2, "silu", GateMode::SoftmaxTopk, true, layout);
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
        model.select_output_outlets(&outputs)?;
        model.into_optimized()
    };

    let canonical = build(
        ExpertLayout::Canonical,
        wg_data.clone(),
        w1_canonical.clone(),
        w2_canonical.clone(),
        w3_canonical.clone(),
    )?;
    let linear = build(
        ExpertLayout::Linear,
        wg_data,
        transpose_expert_last2(&w1_canonical)?,
        transpose_expert_last2(&w2_canonical)?,
        transpose_expert_last2(&w3_canonical)?,
    )?;

    let canonical_result =
        SimplePlan::new(canonical)?.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;
    let linear_result = SimplePlan::new(linear)?.spawn()?.run(tvec![x_data.into_tvalue()])?;

    canonical_result[0].close_enough(&linear_result[0], Approximation::Approximate)?;
    Ok(())
}

#[test]
fn test_codegen_keeps_q40_expert_constants_on_reference_eval() -> TractResult<()> {
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::datum_type().fact([2, 32]))?;
    let mut ref_model = TypedModel::default();
    let ref_x = ref_model.add_source("x", f32::datum_type().fact([2, 32]))?;

    let mut rng_state: u64 = 77;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng()).collect();
        tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
    };

    let wg_data = make_tensor(&[2, 32], &mut next_f32);
    let w1_data = make_tensor(&[2, 32, 32], &mut next_f32);
    let w2_data = make_tensor(&[2, 32, 32], &mut next_f32);
    let w3_data = make_tensor(&[2, 32, 32], &mut next_f32);

    let wg = model.add_const("wg", wg_data.clone())?;
    let w1 = add_q40_const(&mut model, "w1", w1_data.clone())?;
    let w2 = add_q40_const(&mut model, "w2", w2_data.clone())?;
    let w3 = add_q40_const(&mut model, "w3", w3_data.clone())?;

    let op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, true);
    let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
    model.select_output_outlets(&outputs)?;

    let ref_wg = ref_model.add_const("wg", wg_data)?;
    let ref_w1 = ref_model.add_const("w1", q40_roundtrip(&w1_data)?)?;
    let ref_w2 = ref_model.add_const("w2", q40_roundtrip(&w2_data)?)?;
    let ref_w3 = ref_model.add_const("w3", q40_roundtrip(&w3_data)?)?;
    let ref_op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, true);
    let ref_outputs =
        ref_model.wire_node("moe", ref_op, &[ref_x, ref_wg, ref_w1, ref_w2, ref_w3])?;
    ref_model.select_output_outlets(&ref_outputs)?;

    let opt_model = model.into_optimized()?;
    let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
    assert!(has_moe, "Expected Q40 experts to stay on MoeFfn reference eval");
    let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
    assert!(!has_opt, "Q40 experts should not lower to OptMoeFfn yet");
    let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
    assert!(!has_routed, "Q40 experts should not lower to RoutedMatMul yet");

    let x_data = make_tensor(&[2, 32], &mut next_f32);
    let plan = SimplePlan::new(opt_model)?;
    let result = plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;
    let ref_result = SimplePlan::new(ref_model)?.spawn()?.run(tvec![x_data.into_tvalue()])?;
    result[0].close_enough(&ref_result[0], Approximation::Approximate)?;

    Ok(())
}

#[test]
fn test_q40_linear_expert_layout_matches_dequantized_reference() -> TractResult<()> {
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::datum_type().fact([4, 32]))?;
    let mut ref_model = TypedModel::default();
    let ref_x = ref_model.add_source("x", f32::datum_type().fact([4, 32]))?;

    let mut rng_state: u64 = 2026;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng()).collect();
        tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
    };

    let wg_data = make_tensor(&[4, 32], &mut next_f32);
    let w1_data = make_tensor(&[4, 64, 32], &mut next_f32);
    let w2_data = make_tensor(&[4, 32, 64], &mut next_f32);
    let w3_data = make_tensor(&[4, 64, 32], &mut next_f32);

    let wg = model.add_const("wg", wg_data.clone())?;
    let w1 = add_q40_const(&mut model, "w1", w1_data.clone())?;
    let w2 = add_q40_const(&mut model, "w2", w2_data.clone())?;
    let w3 = add_q40_const(&mut model, "w3", w3_data.clone())?;
    let op =
        MoeFfn::basic_with_layout(2, "silu", GateMode::SoftmaxTopk, true, ExpertLayout::Linear);
    let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
    model.select_output_outlets(&outputs)?;

    let ref_wg = ref_model.add_const("wg", wg_data)?;
    let ref_w1 = ref_model.add_const("w1", q40_roundtrip(&w1_data)?)?;
    let ref_w2 = ref_model.add_const("w2", q40_roundtrip(&w2_data)?)?;
    let ref_w3 = ref_model.add_const("w3", q40_roundtrip(&w3_data)?)?;
    let ref_op =
        MoeFfn::basic_with_layout(2, "silu", GateMode::SoftmaxTopk, true, ExpertLayout::Linear);
    let ref_outputs =
        ref_model.wire_node("moe", ref_op, &[ref_x, ref_wg, ref_w1, ref_w2, ref_w3])?;
    ref_model.select_output_outlets(&ref_outputs)?;

    let opt_model = model.into_optimized()?;
    let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
    assert!(!has_moe, "Expected Q40 linear experts to lower to OptMoeFfn");
    let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
    assert!(has_opt, "Q40 linear experts should use OptMoeFfn");
    let uses_direct_q40 = opt_model
        .nodes()
        .iter()
        .filter_map(|n| n.op_as::<OptMoeFfn>())
        .any(|op| op.q40_linear_plan.is_some());
    assert!(uses_direct_q40, "Q40 linear experts should use the direct Q40 plan");
    let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
    assert!(!has_routed, "Q40 linear constants should use the direct optimized plan");

    let x_data = make_tensor(&[4, 32], &mut next_f32);
    let result = SimplePlan::new(opt_model)?.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;
    let ref_result =
        SimplePlan::new(ref_model.into_optimized()?)?.spawn()?.run(tvec![x_data.into_tvalue()])?;
    result[0].close_enough(&ref_result[0], Approximation::Approximate)?;

    Ok(())
}

/// Square experts (`d_model == d_hidden`) with mixed precision, the exact
/// gpt-oss-20b shape: 2880 x 2880 experts, Q40 gate/up, f16 down.
///
/// Every other MoE test uses non-square experts, so a layout mix-up shows
/// up as a shape error. When the two dims are equal nothing catches it,
/// and the wrong orientation silently produces plausible-looking garbage.
/// This runs a long enough sequence that every expert is exercised, and
/// compares the optimized path against the reference evaluator.
#[test]
fn test_opt_moe_ffn_square_mixed_precision_long_sequence() -> TractResult<()> {
    const DIM: usize = 64; // square: d_model == d_hidden
    const EXPERTS: usize = 8;
    const TOKENS: usize = 96; // >> experts, so every expert gets routed

    let mut rng_state: u64 = 20260728;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng()).collect();
        tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
    };

    // Linear layout: w1/w3 [E, H, D], w2 [E, D, H]. With DIM square these
    // are indistinguishable by shape, which is the whole point.
    let wg_data = make_tensor(&[EXPERTS, DIM], &mut next_f32);
    let w1_data = make_tensor(&[EXPERTS, DIM, DIM], &mut next_f32);
    let w2_data = make_tensor(&[EXPERTS, DIM, DIM], &mut next_f32);
    let w3_data = make_tensor(&[EXPERTS, DIM, DIM], &mut next_f32);
    let x_data = make_tensor(&[TOKENS, DIM], &mut next_f32);

    // Optimized model: Q40 w1/w3, plain f16 w2 (the mixed export shape).
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::datum_type().fact([TOKENS, DIM]))?;
    let wg = model.add_const("wg", wg_data.clone())?;
    let w1 = add_q40_const(&mut model, "w1", w1_data.clone())?;
    let w2 = model.add_const("w2", w2_data.clone().cast_to::<f16>()?.into_owned())?;
    let w3 = add_q40_const(&mut model, "w3", w3_data.clone())?;
    let op =
        MoeFfn::basic_with_layout(2, "silu", GateMode::SoftmaxTopk, true, ExpertLayout::Linear);
    let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
    model.select_output_outlets(&outputs)?;

    // Reference: same values, dequantized, on the un-optimized evaluator.
    let mut ref_model = TypedModel::default();
    let ref_x = ref_model.add_source("x", f32::datum_type().fact([TOKENS, DIM]))?;
    let ref_wg = ref_model.add_const("wg", wg_data)?;
    let ref_w1 = ref_model.add_const("w1", q40_roundtrip(&w1_data)?)?;
    let ref_w2 =
        ref_model.add_const("w2", w2_data.cast_to::<f16>()?.cast_to::<f32>()?.into_owned())?;
    let ref_w3 = ref_model.add_const("w3", q40_roundtrip(&w3_data)?)?;
    let ref_op =
        MoeFfn::basic_with_layout(2, "silu", GateMode::SoftmaxTopk, true, ExpertLayout::Linear);
    let ref_outputs =
        ref_model.wire_node("moe", ref_op, &[ref_x, ref_wg, ref_w1, ref_w2, ref_w3])?;
    ref_model.select_output_outlets(&ref_outputs)?;

    let opt_model = model.into_optimized()?;
    let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
    assert!(has_opt, "mixed-precision square experts should reach OptMoeFfn");
    // Mixed precision cannot use the all-Q40 routed plan; it must lower to
    // per-expert subplans instead of silently dropping to the reference op.
    let uses_direct_q40 = opt_model
        .nodes()
        .iter()
        .filter_map(|n| n.op_as::<OptMoeFfn>())
        .any(|op| op.q40_linear_plan.is_some());
    assert!(!uses_direct_q40, "a float w2 rules out the direct Q40 plan");

    let result = SimplePlan::new(opt_model)?.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;
    let ref_result = SimplePlan::new(ref_model)?.spawn()?.run(tvec![x_data.into_tvalue()])?;
    result[0].close_enough(&ref_result[0], Approximation::Approximate)?;

    Ok(())
}

#[test]
fn test_opt_moe_ffn_matches_reference() -> TractResult<()> {
    // Test with SwiGLU (has_w3=true)
    let (model, x_data) = make_moe_model(8, 16, 32, 4, 2, true)?;

    // Run reference (unoptimized)
    let ref_plan = SimplePlan::new(model.clone())?;
    let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

    // Run optimized
    let opt_model = model.into_optimized()?;

    // Verify constant expert weights keep the fast plan-based path.
    let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
    assert!(has_opt, "Expected OptMoeFfn in optimized model");
    let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
    assert!(!has_routed, "Constant expert weights should use OptMoeFfn");

    let opt_plan = SimplePlan::new(opt_model)?;
    let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

    // Compare outputs
    ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;

    Ok(())
}

#[test]
fn test_opt_moe_ffn_no_w3() -> TractResult<()> {
    // Test without SwiGLU (has_w3=false)
    let (model, x_data) = make_moe_model(8, 16, 32, 4, 2, false)?;

    let ref_plan = SimplePlan::new(model.clone())?;
    let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

    let opt_model = model.into_optimized()?;
    let opt_plan = SimplePlan::new(opt_model)?;
    let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

    ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;

    Ok(())
}

#[test]
fn test_opt_moe_ffn_top1() -> TractResult<()> {
    let (model, x_data) = make_moe_model(16, 8, 16, 8, 1, true)?;

    let ref_plan = SimplePlan::new(model.clone())?;
    let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

    let opt_model = model.into_optimized()?;
    let opt_plan = SimplePlan::new(opt_model)?;
    let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

    ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;

    Ok(())
}

#[test]
fn test_codegen_lowers_non_const_weights() -> TractResult<()> {
    // The routed primitive lowering does not require constant weights.
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::datum_type().fact([4, 8]))?;
    let wg = model.add_source("wg", f32::datum_type().fact([2, 8]))?;
    let w1 = model.add_source("w1", f32::datum_type().fact([2, 8, 16]))?;
    let w2 = model.add_source("w2", f32::datum_type().fact([2, 16, 8]))?;

    let op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, false);
    let outputs = model.wire_node("moe", op, &[x, wg, w1, w2])?;
    model.select_output_outlets(&outputs)?;

    let opt_model = model.into_optimized()?;

    let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
    assert!(!has_moe, "Expected MoeFfn to lower even when weights are not constants");
    let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
    assert!(has_routed, "Expected RoutedMatMul in optimized model");

    Ok(())
}

#[test]
fn test_codegen_keeps_non_const_linear_layout_on_moe_ffn() -> TractResult<()> {
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::datum_type().fact([4, 8]))?;
    let wg = model.add_source("wg", f32::datum_type().fact([2, 8]))?;
    let w1 = model.add_source("w1", f32::datum_type().fact([2, 16, 8]))?;
    let w2 = model.add_source("w2", f32::datum_type().fact([2, 8, 16]))?;

    let op =
        MoeFfn::basic_with_layout(1, "silu", GateMode::SoftmaxTopk, false, ExpertLayout::Linear);
    let outputs = model.wire_node("moe", op, &[x, wg, w1, w2])?;
    model.select_output_outlets(&outputs)?;

    let opt_model = model.into_optimized()?;

    let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
    assert!(has_moe, "Expected linear-layout non-const experts to remain on MoeFfn");
    let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
    assert!(!has_routed, "RoutedMatMul currently models canonical expert layout");

    Ok(())
}

#[test]
fn test_routed_matmul_groups_by_expert_and_preserves_route_order() -> TractResult<()> {
    let input = Tensor::from_shape(&[3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let weights = Tensor::from_shape(
        &[2, 2, 2],
        &[
            1.0f32, 0.0, 0.0, 1.0, // expert 0: identity
            10.0, 0.0, 0.0, 100.0, // expert 1: scale columns differently
        ],
    )?;
    let route_token_ids = Tensor::from_shape(&[4], &[2i64, 0, 1, 2])?;
    let route_expert_ids = Tensor::from_shape(&[4], &[1i64, 0, 1, 0])?;

    let op = RoutedMatMul { input_mode: RoutedInputMode::TokenRows, cache_weights: false };
    let result = op.eval(
        &EvalContext::out_of_plan(),
        tvec![
            input.into_tvalue(),
            weights.into_tvalue(),
            route_token_ids.into_tvalue(),
            route_expert_ids.into_tvalue(),
        ],
    )?;

    let expected = Tensor::from_shape(
        &[4, 2],
        &[
            50.0f32, 600.0, // route 0: token 2 through expert 1
            1.0, 2.0, // route 1: token 0 through expert 0
            30.0, 400.0, // route 2: token 1 through expert 1
            5.0, 6.0, // route 3: token 2 through expert 0
        ],
    )?;
    result[0].close_enough(&expected, Approximation::Approximate)?;

    Ok(())
}

#[test]
fn test_moe_ffn_runs_with_q40_expert_constants() -> TractResult<()> {
    let mut model = TypedModel::default();
    let t_tokens = 4;
    let d_model = 32;
    let d_hidden = 64;
    let num_experts = 4;

    let x = model.add_source("x", f32::datum_type().fact([t_tokens, d_model]))?;

    let mut rng_state: u64 = 1337;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng()).collect();
        tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
    };

    let wg = model.add_const("wg", make_tensor(&[num_experts, d_model], &mut next_f32))?;
    let w1 = add_q40_const(
        &mut model,
        "w1",
        make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32),
    )?;
    let w2 = add_q40_const(
        &mut model,
        "w2",
        make_tensor(&[num_experts, d_hidden, d_model], &mut next_f32),
    )?;
    let w3 = add_q40_const(
        &mut model,
        "w3",
        make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32),
    )?;

    let op = MoeFfn::basic(2, "silu", GateMode::SoftmaxTopk, true);
    let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
    model.select_output_outlets(&outputs)?;

    let opt_model = model.into_optimized()?;
    let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
    let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
    assert!(has_moe, "Expected Q40 experts to stay on MoeFfn reference eval");
    assert!(!has_opt, "Q40 experts should not use OptMoeFfn yet");

    let x_data = make_tensor(&[t_tokens, d_model], &mut next_f32);
    let result = SimplePlan::new(opt_model)?.spawn()?.run(tvec![x_data.into_tvalue()])?;
    let output = result[0].to_plain_array_view::<f32>()?;
    assert!(output.iter().all(|v| v.is_finite()));

    Ok(())
}

/// Findings: the routed fall-through lowering models neither biases nor
/// the clamped activation, so ops carrying either must keep the reference
/// evaluator when their weights are not constants.
fn non_const_codegen_patch(
    op: MoeFfn,
    extra_bias_input: bool,
) -> TractResult<Option<TypedModelPatch>> {
    let (t_tokens, d_model, d_hidden, num_experts) = (4, 16, 32, 2);
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::datum_type().fact([t_tokens, d_model]))?;
    let wg = model.add_source("wg", f32::datum_type().fact([num_experts, d_model]))?;
    let w1 = model.add_source("w1", f32::datum_type().fact([num_experts, d_model, d_hidden]))?;
    let w2 = model.add_source("w2", f32::datum_type().fact([num_experts, d_hidden, d_model]))?;
    let mut inputs = vec![x, wg, w1, w2];
    if op.has_w3 {
        let w3 =
            model.add_source("w3", f32::datum_type().fact([num_experts, d_model, d_hidden]))?;
        inputs.push(w3);
    }
    if extra_bias_input {
        let w2_bias =
            model.add_source("w2_bias", f32::datum_type().fact([num_experts, d_model]))?;
        inputs.push(w2_bias);
    }
    let outputs = model.wire_node("moe", op.clone(), &inputs)?;
    model.select_output_outlets(&outputs)?;
    op.codegen(&model, model.node(outputs[0].node))
}

#[test]
fn test_biased_moe_non_const_weights_stays_reference() -> TractResult<()> {
    let mut op = MoeFfn::basic(2, "swiglu", GateMode::SoftmaxTopk, true);
    op.has_w2_bias = true;
    let patch = non_const_codegen_patch(op, true)?;
    assert!(patch.is_none(), "biased MoE with non-const weights must not lower");
    Ok(())
}

#[test]
fn test_clamped_moe_non_const_weights_stays_reference() -> TractResult<()> {
    let mut op = MoeFfn::basic(2, "swiglu", GateMode::SoftmaxTopk, true);
    op.act_alpha_bits = Some(1.702f32.to_bits());
    op.act_limit_bits = Some(7.0f32.to_bits());
    let patch = non_const_codegen_patch(op, false)?;
    assert!(patch.is_none(), "clamped-activation MoE with non-const weights must not lower");
    Ok(())
}

#[test]
fn test_plain_moe_non_const_weights_still_lowers_routed() -> TractResult<()> {
    let op = MoeFfn::basic(2, "swiglu", GateMode::SoftmaxTopk, true);
    let patch = non_const_codegen_patch(op, false)?;
    assert!(patch.is_some(), "bias-free plain MoE should still get the routed lowering");
    Ok(())
}

/// Finding: reference evaluators used to hardcode silu whatever
/// `self.activation` declared. Single expert, k = 1, so the softmax gate
/// weight is exactly 1.0 and the expected output is gelu(x @ w1) @ w2.
#[test]
fn test_reference_eval_applies_gelu_activation() -> TractResult<()> {
    let (t_tokens, d_model, d_hidden) = (3, 4, 8);
    let mut rng_state: u64 = 7;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let make = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng()).collect();
        tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
    };
    let x = make(&[t_tokens, d_model], &mut next_f32);
    let wg = make(&[1, d_model], &mut next_f32);
    let w1 = make(&[1, d_model, d_hidden], &mut next_f32);
    let w2 = make(&[1, d_hidden, d_model], &mut next_f32);

    let op = MoeFfn::basic(1, "gelu", GateMode::SoftmaxTopk, false);
    let got = op.eval(
        &EvalContext::out_of_plan(),
        tvec![
            x.clone().into_tvalue(),
            wg.into_tvalue(),
            w1.clone().into_tvalue(),
            w2.clone().into_tvalue(),
        ],
    )?;

    let x_a: tract_ndarray::Array2<f32> =
        x.to_plain_array_view::<f32>()?.into_dimensionality()?.to_owned();
    let w1_a: tract_ndarray::Array2<f32> = w1
        .to_plain_array_view::<f32>()?
        .into_shape_with_order((d_model, d_hidden))?
        .into_dimensionality()?
        .to_owned();
    let w2_a: tract_ndarray::Array2<f32> = w2
        .to_plain_array_view::<f32>()?
        .into_shape_with_order((d_hidden, d_model))?
        .into_dimensionality()?
        .to_owned();
    let mut h = x_a.dot(&w1_a);
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    h.iter_mut().for_each(|v| {
        let x = *v;
        *v = 0.5 * x * (1.0 + f32::tanh(sqrt_2_over_pi * (x + 0.044715 * x.powi(3))));
    });
    let expected = h.dot(&w2_a).into_tensor();

    got[0].close_enough(&expected, Approximation::Approximate)?;
    Ok(())
}

#[test]
fn test_e2e_nnef_qwen3_moe() -> TractResult<()> {
    use crate::WithTractTransformers;
    use std::io::Cursor;

    // Load the Qwen3 MoE model exported from transformers
    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../harness/nnef-test-cases/moe-ffn/qwen3-tiny");

    let nnef = tract_nnef::nnef().with_tract_transformers();
    let model = nnef.model_for_path(&model_path)?;
    let model = model.into_optimized()?;

    // Verify constant expert weights use the fast plan-based path.
    let has_opt = model.nodes().iter().any(|n: &TypedNode| n.op_is::<OptMoeFfn>());
    assert!(has_opt, "Expected OptMoeFfn in optimized model");
    let has_routed = model.nodes().iter().any(|n: &TypedNode| n.op_is::<RoutedMatMul>());
    assert!(!has_routed, "Constant expert weights should use OptMoeFfn");

    let plan = SimplePlan::new(model)?;

    // Load input and expected output from io.npz
    let npz_path = model_path.join("io.npz");
    let npz_bytes = std::fs::read(&npz_path)?;
    let mut npz = ndarray_npy::NpzReader::new(Cursor::new(npz_bytes))?;

    let input: tract_ndarray::ArrayD<f32> = npz.by_name("input_0.npy")?;
    let expected_output: tract_ndarray::ArrayD<f32> = npz.by_name("output_0.npy")?;

    // Run inference
    let result = plan.spawn()?.run(tvec![input.into_tensor().into_tvalue()])?;

    // Compare against PyTorch reference output
    result[0].close_enough(&expected_output.into_tensor(), Approximation::Approximate)?;

    Ok(())
}

#[test]
fn moe_rejects_invalid_facts() -> TractResult<()> {
    let op =
        MoeFfn::basic_with_layout(2, "silu", GateMode::SoftmaxTopk, false, ExpertLayout::Canonical);
    let facts =
        vec![f32::fact([2, 4]), f32::fact([3, 4]), f32::fact([3, 4, 8]), f32::fact([3, 8, 4])];
    op.validate_facts(&facts.iter().collect::<Vec<_>>())?;
    assert!(op.validate_facts(&[]).is_err());
    for (index, bad) in [
        (0, f32::fact([4])),
        (0, i32::fact([2, 4])),
        (1, f32::fact([2, 3, 4])),
        (1, f32::fact([3, 5])),
        (2, f32::fact([2, 4, 8])),
        (2, f32::fact([3, 5, 8])),
        (3, f32::fact([3, 9, 4])),
        (3, f32::fact([3, 8, 5])),
    ] {
        let mut invalid = facts.clone();
        invalid[index] = bad;
        assert!(op.validate_facts(&invalid.iter().collect::<Vec<_>>()).is_err());
    }
    for k in [0, 4, usize::MAX] {
        let mut invalid = op.clone();
        invalid.k = k;
        assert!(invalid.validate_facts(&facts.iter().collect::<Vec<_>>()).is_err());
    }
    for (slot, shape) in [(0, vec![4]), (1, vec![3, 9]), (2, vec![3, 9]), (3, vec![3, 5])] {
        let mut biased = op.clone();
        biased.has_w3 = true;
        biased.has_wg_bias = slot == 0;
        biased.has_w1_bias = slot == 1;
        biased.has_w3_bias = slot == 2;
        biased.has_w2_bias = slot == 3;
        let mut invalid = facts.clone();
        invalid.push(facts[2].clone());
        invalid.push(f32::fact(shape));
        assert!(biased.validate_facts(&invalid.iter().collect::<Vec<_>>()).is_err());
    }
    let mut invalid = op.clone();
    invalid.has_w3_bias = true;
    let mut invalid_facts = facts.clone();
    invalid_facts.push(f32::fact([3, 8]));
    assert!(invalid.validate_facts(&invalid_facts.iter().collect::<Vec<_>>()).is_err());
    for (alpha, limit) in [(Some(f32::NAN), None), (None, Some(0.0)), (None, Some(f32::INFINITY))] {
        let mut invalid = op.clone();
        invalid.act_alpha_bits = alpha.map(f32::to_bits);
        invalid.act_limit_bits = limit.map(f32::to_bits);
        assert!(invalid.validate_facts(&facts.iter().collect::<Vec<_>>()).is_err());
    }
    let symbols = SymbolScope::default();
    let mut symbolic = facts.clone();
    symbolic[0] = f32::fact([symbols.sym("T").to_dim(), symbols.sym("D").to_dim()]);
    op.validate_facts(&symbolic.iter().collect::<Vec<_>>())?;
    let inputs = tvec![
        Tensor::zero::<f32>(&[2, 5])?.into_tvalue(),
        Tensor::zero::<f32>(&[3, 4])?.into_tvalue(),
        Tensor::zero::<f32>(&[3, 4, 8])?.into_tvalue(),
        Tensor::zero::<f32>(&[3, 8, 4])?.into_tvalue()
    ];
    assert!(op.eval(&EvalContext::out_of_plan(), inputs).is_err());
    Ok(())
}

#[test]
fn routing_ties_have_one_order() {
    for k in [1, 2, 3, 5] {
        let mut scores = vec![(4, 0.0), (3, -0.0), (2, 1.0), (1, 1.0), (0, 1.0)];
        select_routes(&mut scores, k);
        assert_eq!(scores.iter().map(|s| s.0).collect::<Vec<_>>(), (0..k).collect::<Vec<_>>());
    }
}

#[test]
fn distinct_moe_branches_survive_codegen() -> TractResult<()> {
    for change_weights in [false, true] {
        let (mut model, input) = make_moe_model(2, 4, 8, 3, 2, true)?;
        let first = model.output_outlets()?[0];
        let node = model.node(first.node);
        let mut op = node.op_as::<MoeFfn>().unwrap().clone();
        let mut inputs = node.inputs.clone();
        if change_weights {
            inputs[3] = model.add_const("other_w2", Tensor::zero::<f32>(&[3, 8, 4])?)?;
        } else {
            op.activation = "relu".into();
        }
        let second = model.wire_node("other_moe", op, &inputs)?[0];
        let combined = model.wire_node("sum", add(), &[first, second])?;
        model.select_output_outlets(&combined)?;
        let expected = SimplePlan::new(model.clone())?.run(tvec![input.clone().into()])?;
        let optimized = model.into_optimized()?;
        assert_eq!(optimized.nodes().iter().filter(|n| n.op_is::<OptMoeFfn>()).count(), 2);
        let op = optimized.nodes().iter().find_map(|n| n.op_as::<OptMoeFfn>()).unwrap();
        let cloned = op.clone();
        assert_eq!(op, &cloned);
        let hash = |op: &OptMoeFfn| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            op.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(op), hash(&cloned));
        let mut biased = cloned.clone();
        biased.wg_bias = Some(Tensor::zero::<f32>(&[3])?);
        assert_ne!(op, &biased);
        let mut state = op.state(&EvalContext::out_of_plan())?.unwrap();
        assert!(
            state
                .eval(
                    &EvalContext::out_of_plan(),
                    op,
                    tvec![Tensor::zero::<f32>(&[2, 5])?.into_tvalue()]
                )
                .is_err()
        );
        let actual = SimplePlan::new(optimized)?.run(tvec![input.into()])?;
        actual[0].close_enough(&expected[0], Approximation::Approximate)?;
    }
    Ok(())
}

#[test]
fn nnef_moe_roundtrip_variants() -> TractResult<()> {
    use crate::WithTractTransformers;

    let nnef = tract_nnef::nnef().with_tract_transformers();
    for layout in [ExpertLayout::Canonical, ExpertLayout::Linear] {
        for gate in [GateMode::SoftmaxTopk, GateMode::SoftmaxAll, GateMode::Sigmoid, GateMode::Raw]
        {
            for variant in 0..32 {
                let has_w3 = variant & 1 != 0;
                if !has_w3 && variant & 8 != 0 {
                    continue;
                }
                let mut op = MoeFfn::basic_with_layout(2, "silu", gate.clone(), has_w3, layout);
                op.has_wg_bias = variant & 2 != 0;
                op.has_w1_bias = variant & 4 != 0;
                op.has_w3_bias = variant & 8 != 0;
                op.has_w2_bias = variant & 16 != 0;
                if variant == 31 {
                    op.act_alpha_bits = Some(1.702f32.to_bits());
                    op.act_limit_bits = Some(0.5f32.to_bits());
                }
                let mut model = TypedModel::default();
                let x = model.add_source("x", f32::fact([2, 4]))?;
                let mut inputs = vec![x];
                let (w1, w2) = match layout {
                    ExpertLayout::Canonical => (vec![3, 4, 8], vec![3, 8, 4]),
                    ExpertLayout::Linear => (vec![3, 8, 4], vec![3, 4, 8]),
                };
                let mut tensors = vec![("wg", vec![3, 4]), ("w1", w1.clone()), ("w2", w2)];
                if has_w3 {
                    tensors.push(("w3", w1));
                }
                for (name, enabled, shape) in [
                    ("wg_bias", op.has_wg_bias, vec![3]),
                    ("w1_bias", op.has_w1_bias, vec![3, 8]),
                    ("w3_bias", op.has_w3_bias, vec![3, 8]),
                    ("w2_bias", op.has_w2_bias, vec![3, 4]),
                ] {
                    if enabled {
                        tensors.push((name, shape));
                    }
                }
                for (i, (name, shape)) in tensors.iter().enumerate() {
                    let values: Vec<f32> = (0..shape.iter().product())
                        .map(|j| ((j * 7 + i * 3) % 19) as f32 / 10.0 - 0.9)
                        .collect();
                    inputs.push(model.add_const(*name, Tensor::from_shape(shape, &values)?)?);
                }
                let output = model.wire_node("moe", op.clone(), &inputs)?;
                model.select_output_outlets(&output)?;
                let mut bytes = vec![];
                nnef.write_to_tar(&model, &mut bytes)?;
                let reloaded = nnef.model_for_read(&mut bytes.as_slice())?;
                let node = reloaded
                    .nodes()
                    .iter()
                    .find(|n| n.op_is::<MoeFfn>())
                    .context("Missing MoeFfn")?;
                assert_eq!(node.op_as::<MoeFfn>(), Some(&op));
                for (original, loaded) in inputs.iter().skip(1).zip(node.inputs.iter().skip(1)) {
                    assert_eq!(
                        model.outlet_fact(*original)?.konst,
                        reloaded.outlet_fact(*loaded)?.konst
                    );
                }
                let input =
                    Tensor::from_shape(&[2, 4], &[0.2f32, -0.3, 0.7, 0.1, 0.8, 0.4, -0.5, 0.6])?;
                let expected = SimplePlan::new(model)?.run(tvec![input.clone().into()])?;
                let actual = SimplePlan::new(reloaded)?.run(tvec![input.into()])?;
                actual[0].close_enough(&expected[0], Approximation::Exact)?;
            }
        }
    }
    Ok(())
}

#[test]
fn nnef_moe_roundtrip_q40_storage() -> TractResult<()> {
    use crate::WithTractTransformers;

    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 32]))?;
    let wg = model.add_const("wg", Tensor::from_shape(&[2, 32], &vec![0.25f32; 64])?)?;
    let weights = Tensor::from_shape(&[2, 32, 32], &vec![0.125f32; 2048])?;
    let w1 = add_q40_const(&mut model, "w1", weights.clone())?;
    let w2 = add_q40_const(&mut model, "w2", weights)?;
    let op =
        MoeFfn::basic_with_layout(1, "silu", GateMode::SoftmaxTopk, false, ExpertLayout::Linear);
    let output = model.wire_node("moe", op, &[x, wg, w1, w2])?;
    model.select_output_outlets(&output)?;
    let nnef = tract_nnef::nnef().with_tract_transformers();
    let mut bytes = vec![];
    nnef.write_to_tar(&model, &mut bytes)?;
    let reloaded = nnef.model_for_read(&mut bytes.as_slice())?;
    let node = reloaded.nodes().iter().find(|n| n.op_is::<MoeFfn>()).context("Missing MoeFfn")?;
    for (original, loaded) in [w1, w2].iter().zip(node.inputs[2..].iter()) {
        let original = model
            .outlet_fact(*original)?
            .konst
            .as_ref()
            .unwrap()
            .try_storage_as::<BlockQuantStorage>()?;
        let loaded = reloaded
            .outlet_fact(*loaded)?
            .konst
            .as_ref()
            .unwrap()
            .try_storage_as::<BlockQuantStorage>()?;
        assert_eq!(original.value(), loaded.value());
    }
    let input = Tensor::from_shape(&[1, 32], &[0.1f32; 32])?;
    let expected = SimplePlan::new(model)?.run(tvec![input.clone().into()])?;
    let actual = SimplePlan::new(reloaded)?.run(tvec![input.into()])?;
    actual[0].close_enough(&expected[0], Approximation::Exact)?;
    Ok(())
}

#[test]
fn nnef_optimized_moe_reports_export_boundary() -> TractResult<()> {
    use crate::WithTractTransformers;

    let (model, _) = make_moe_model(2, 4, 8, 3, 2, true)?;
    let model = model.into_optimized()?;
    let error =
        tract_nnef::nnef().with_tract_transformers().write_to_tar(&model, &mut vec![]).unwrap_err();
    assert!(format!("{error:#}").contains("serialize MoeFfn before CPU codegen"));
    Ok(())
}
