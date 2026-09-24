use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::math::add;

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
fn codegen_selects_q40_execution_by_layout() -> TractResult<()> {
    for layout in [ExpertLayout::Canonical, ExpertLayout::Linear] {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([2, 32]))?;
        let wg = model.add_const("wg", Tensor::from_shape(&[2, 32], &[0.25f32; 64])?)?;
        let (up_shape, down_shape) = match layout {
            ExpertLayout::Canonical => ([2, 32, 64], [2, 64, 32]),
            ExpertLayout::Linear => ([2, 64, 32], [2, 32, 64]),
        };
        let w1 = add_q40_const(&mut model, "w1", Tensor::from_shape(&up_shape, &[0.25f32; 4096])?)?;
        let w2 =
            add_q40_const(&mut model, "w2", Tensor::from_shape(&down_shape, &[0.25f32; 4096])?)?;
        let w3 = add_q40_const(&mut model, "w3", Tensor::from_shape(&up_shape, &[0.25f32; 4096])?)?;
        let op = MoeFfn::basic_with_layout(1, "silu", GateMode::SoftmaxTopk, true, layout);
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
        model.select_output_outlets(&outputs)?;
        let model = model.into_optimized()?;
        let optimized: Vec<_> =
            model.nodes().iter().filter_map(|n| n.op_as::<OptMoeFfn>()).collect();
        let reference_count = model.nodes().iter().filter(|n| n.op_is::<MoeFfn>()).count();
        match layout {
            ExpertLayout::Canonical => {
                assert_eq!(reference_count, 1);
                assert!(optimized.is_empty());
            }
            ExpertLayout::Linear => {
                assert_eq!(reference_count, 0);
                assert_eq!(optimized.len(), 1);
                assert!(optimized[0].uses_direct_q40());
            }
        }
        assert!(!model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>()));
    }
    Ok(())
}

/// Square Q40 gate/up and f16 down projections expose layout errors that
/// shape checks cannot catch. A long token batch exercises every expert
/// against the reference evaluator.
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
        .any(|op| op.uses_direct_q40());
    assert!(!uses_direct_q40, "a float w2 rules out the direct Q40 plan");

    let result = SimplePlan::new(opt_model)?.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;
    let ref_result = SimplePlan::new(ref_model)?.spawn()?.run(tvec![x_data.into_tvalue()])?;
    result[0].close_enough(&ref_result[0], Approximation::Approximate)?;

    Ok(())
}

#[test]
fn codegen_uses_subplans_for_float_constants() -> TractResult<()> {
    for has_w3 in [false, true] {
        for k in [1, 2] {
            let (model, _) = make_moe_model(8, 16, 32, 4, k, has_w3)?;
            let model = model.into_optimized()?;
            assert!(model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>()));
            assert!(!model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>()));
        }
    }
    Ok(())
}

#[test]
fn test_codegen_lowers_non_const_weights() -> TractResult<()> {
    // The routed primitive lowering does not require constant weights.
    let mut model = TypedModel::default();
    let tokens = model.symbols.sym("tokens");
    let x = model.add_source("x", f32::fact([tokens.to_dim(), 8.to_dim()]))?;
    let wg = model.add_source("wg", f32::datum_type().fact([2, 8]))?;
    let w1 = model.add_source("w1", f32::datum_type().fact([2, 8, 16]))?;
    let w2 = model.add_source("w2", f32::datum_type().fact([2, 16, 8]))?;

    let op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, false);
    let outputs = model.wire_node("moe", op, &[x, wg, w1, w2])?;
    model.select_output_outlets(&outputs)?;

    let opt_model = model.clone().into_optimized()?;

    let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
    assert!(!has_moe, "Expected MoeFfn to lower even when weights are not constants");
    let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
    assert!(has_routed, "Expected RoutedMatMul in optimized model");

    let reference = SimplePlan::new(model)?;
    let optimized = SimplePlan::new(opt_model)?;
    for tokens in [1, 4] {
        let inputs = tvec![
            Tensor::from_shape(&[tokens, 8], &vec![0.5f32; tokens * 8])?.into_tvalue(),
            Tensor::from_shape(&[2, 8], &[0.25f32; 16])?.into_tvalue(),
            Tensor::from_shape(&[2, 8, 16], &[0.25f32; 256])?.into_tvalue(),
            Tensor::from_shape(&[2, 16, 8], &[0.25f32; 256])?.into_tvalue(),
        ];
        let expected = reference.run(inputs.clone())?;
        let actual = optimized.run(inputs)?;
        actual[0].close_enough(&expected[0], Approximation::Approximate)?;
    }

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
fn optimized_moe_bias_equality_implies_equal_hashes() -> TractResult<()> {
    let (model, _) = make_moe_model(2, 4, 8, 3, 2, true)?;
    let optimized = model.into_optimized()?;
    let op = optimized.nodes().iter().find_map(|n| n.op_as::<OptMoeFfn>()).unwrap();
    let hash = |op: &OptMoeFfn| {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        op.hash(&mut hasher);
        hasher.finish()
    };
    for (a, b) in
        [(0.0, -0.0), (f32::from_bits(0x7fc00001), f32::from_bits(0x7fc00002)), (1.0, 1.0)]
    {
        let mut lhs = op.clone();
        let mut rhs = op.clone();
        lhs.wg_bias = Some(Tensor::from_shape(&[3], &[a, 0.0, 1.0])?);
        rhs.wg_bias = Some(Tensor::from_shape(&[3], &[b, 0.0, 1.0])?);
        assert_eq!(lhs, rhs);
        assert_eq!(hash(&lhs), hash(&rhs));
        rhs.wg_bias = Some(Tensor::from_shape(&[3], &[2.0, 0.0, 1.0])?);
        assert_ne!(lhs, rhs);
    }
    Ok(())
}

#[test]
fn symbolic_moe_contract_survives_codegen() -> TractResult<()> {
    for symbolic_up in [false, true] {
        let mut model = TypedModel::default();
        let e = model.symbols.sym("router_experts");
        let h = model.symbols.sym("up_hidden");
        let x = model.add_source("x", f32::fact([1, 2]))?;
        let wg = model.add_source(
            "wg",
            f32::fact([if symbolic_up { 3.to_dim() } else { e.to_dim() }, 2.to_dim()]),
        )?;
        let w1 = model.add_const("w1", Tensor::from_shape(&[3, 2, 2], &[0.5f32; 12])?)?;
        let w2 = model.add_const("w2", Tensor::from_shape(&[3, 2, 2], &[0.25f32; 12])?)?;
        let mut inputs = tvec![x, wg, w1, w2];
        if symbolic_up {
            inputs.push(model.add_source("w3", f32::fact([3.to_dim(), 2.to_dim(), h.to_dim()]))?);
        }
        let op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, symbolic_up);
        let out = model.wire_node("moe", op, &inputs)?;
        model.select_output_outlets(&out)?;
        let optimized = model.clone().into_optimized()?;
        for valid in [false, true] {
            let router_e = if valid || symbolic_up { 3 } else { 2 };
            let mut values = tvec![
                Tensor::from_shape(&[1, 2], &[1.0f32, 2.0])?.into_tvalue(),
                Tensor::zero::<f32>(&[router_e, 2])?.into_tvalue()
            ];
            if symbolic_up {
                let width = if valid { 2 } else { 1 };
                values.push(
                    Tensor::from_shape(&[3, 2, width], &vec![0.5f32; 6 * width])?.into_tvalue(),
                );
            }
            let reference = SimplePlan::new(model.clone())?.run(values.clone());
            let actual = SimplePlan::new(optimized.clone())?.run(values);
            if valid {
                actual?[0].close_enough(&reference?[0], Approximation::Approximate)?;
            } else {
                assert!(reference.is_err());
                assert!(actual.is_err(), "codegen lost the symbolic MoE contract");
            }
        }
        assert!(optimized.nodes().iter().any(|n| n.op_is::<MoeFfn>()));
    }
    Ok(())
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
        let other = optimized
            .nodes()
            .iter()
            .filter_map(|n| n.op_as::<OptMoeFfn>())
            .find(|other| !std::ptr::eq(*other, op))
            .unwrap();
        let error = state
            .eval(&EvalContext::out_of_plan(), other, tvec![input.clone().into_tvalue()])
            .unwrap_err();
        assert!(error.to_string().contains("different plan"));
        state.eval(&EvalContext::out_of_plan(), &cloned, tvec![input.clone().into_tvalue()])?;
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
