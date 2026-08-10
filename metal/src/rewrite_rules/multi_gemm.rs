//! Horizontal sibling-gemm fusion: batches the matmuls that share one
//! activation outlet (attention q/k/v, GDN in-projections, shared-expert
//! gate/up, ...) into a single [`MetalMultiGemm`] dispatch at decode. The
//! sibling weight constants are concatenated along their output-row axis at
//! transform time (Q4_0 rows concatenate bytewise), so the fused mat-vec
//! reads one weight tensor and the shared activation exactly once.

use std::collections::HashMap;

use crate::kernels::matmul::GgmlGemm;
use crate::ops::{MetalGemm, MetalMultiGemm};
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::{
    BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0,
};
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::{as_q40_tensor, as_quant_fact};

/// Weight family of one candidate: all siblings of a group must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum WeightClass {
    Q40,
    Plain(DatumType),
}

struct Candidate {
    node: usize,
    w_outlet: OutletId,
    class: WeightClass,
    n: usize,
}

fn log_enabled() -> bool {
    std::env::var_os("TRACT_METAL_LOG_MULTI_GEMM").is_some()
}

pub fn fuse_sibling_gemms(model: &mut TypedModel) -> TractResult<()> {
    if std::env::var_os("TRACT_METAL_DISABLE_MULTI_GEMM").is_some() {
        return Ok(());
    }
    let mut changed = false;
    loop {
        let Some(patch) = find_sibling_group(model)? else { break };
        patch.apply(model)?;
        changed = true;
    }
    if changed {
        // Drop the now-dead per-sibling weight consts so their device
        // buffers are released instead of doubling resident weight memory.
        model.compact()?;
    }
    Ok(())
}

fn outlet_is_const(model: &TypedModel, outlet: OutletId) -> bool {
    outlet.slot == 0 && model.node(outlet.node).op_is::<Const>()
}

/// The logical fact behind a possibly device-wrapped fact.
fn inner_fact(fact: &TypedFact) -> &TypedFact {
    fact.as_device_fact().map(|df| &**df).unwrap_or(fact)
}

fn find_sibling_group(model: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
    let mut groups: HashMap<(OutletId, bool), Vec<usize>> = HashMap::new();
    let mut order: Vec<(OutletId, bool)> = vec![];
    for node in model.nodes() {
        let Some(gemm) = node.op_as::<MetalGemm<GgmlGemm>>() else { continue };
        if gemm.kernel.transpose_a || !gemm.kernel.transpose_b {
            continue;
        }
        if node.inputs.len() != 2 || node.outputs.len() != 1 {
            continue;
        }
        let (a, b) = (node.inputs[0], node.inputs[1]);
        let (x, weight_first) = match (outlet_is_const(model, a), outlet_is_const(model, b)) {
            (false, true) => (a, false),
            (true, false) => (b, true),
            _ => continue,
        };
        let key = (x, weight_first);
        groups
            .entry(key)
            .or_insert_with(|| {
                order.push(key);
                vec![]
            })
            .push(node.id);
    }
    for key in order {
        let nodes = &groups[&key];
        if nodes.len() < 2 {
            continue;
        }
        if let Some(patch) = try_build_group_patch(model, key.0, key.1, nodes)? {
            return Ok(Some(patch));
        }
    }
    Ok(None)
}

/// Validates one candidate node of a group; None when it cannot join.
fn candidate(
    model: &TypedModel,
    node_id: usize,
    weight_first: bool,
    x_dt: DatumType,
    k: usize,
) -> TractResult<Option<Candidate>> {
    let node = model.node(node_id);
    let w_outlet = node.inputs[if weight_first { 0 } else { 1 }];
    let w_fact = inner_fact(model.outlet_fact(w_outlet)?);
    let (class, dims): (WeightClass, TVec<usize>) =
        if let Some(bqf) = as_quant_fact(w_fact, &Q4_0) {
            (WeightClass::Q40, bqf.shape().into())
        } else {
            let Some(conc) = w_fact.shape.as_concrete() else { return Ok(None) };
            (WeightClass::Plain(w_fact.datum_type), conc.into())
        };
    // Logical weight shape must be [n, k] after squeezing leading 1s.
    if dims.len() < 2
        || dims[dims.len() - 1] != k
        || dims[..dims.len() - 2].iter().any(|&d| d != 1)
    {
        return Ok(None);
    }
    let n = dims[dims.len() - 2];
    // Dtype combinations the ggml kernels accept, with the fused output
    // dtype (follows the activation) equal to the original output dtype.
    let supported = match class {
        WeightClass::Q40 => {
            !weight_first
                && matches!(x_dt, DatumType::F16 | DatumType::F32)
                && k % Q4_0.block_len() == 0
        }
        WeightClass::Plain(w_dt) => {
            if weight_first {
                // The original output followed the weight dtype; the fused
                // gemv follows the activation dtype, so they must agree.
                w_dt == x_dt && matches!(x_dt, DatumType::F16 | DatumType::F32)
            } else {
                matches!(
                    (x_dt, w_dt),
                    (DatumType::F32, DatumType::F32)
                        | (DatumType::F16, DatumType::F16)
                        | (DatumType::F32, DatumType::F16)
                )
            }
        }
    };
    if !supported {
        return Ok(None);
    }
    Ok(Some(Candidate { node: node_id, w_outlet, class, n }))
}

fn try_build_group_patch(
    model: &TypedModel,
    x: OutletId,
    weight_first: bool,
    nodes: &[usize],
) -> TractResult<Option<TypedModelPatch>> {
    let x_fact_outer = model.outlet_fact(x)?;
    if x_fact_outer.as_device_fact().is_none() {
        return Ok(None);
    }
    let x_fact = inner_fact(x_fact_outer).clone();
    let x_rank = x_fact.rank();
    if x_rank < 2 {
        return Ok(None);
    }
    let x_dt = x_fact.datum_type;
    let Ok(k) = x_fact.shape[x_rank - 1].to_usize() else { return Ok(None) };
    if weight_first && !x_fact.shape[..x_rank - 2].iter().all(|d| d.is_one()) {
        // The per-sibling prefill dispatch writes [n_i, rows]; batch dims
        // folded into rows would not match the declared layout.
        return Ok(None);
    }

    // Bucket the group by weight class and keep the first fusable bucket.
    let mut sorted_nodes: Vec<usize> = nodes.to_vec();
    sorted_nodes.sort();
    let mut buckets: Vec<(WeightClass, Vec<Candidate>)> = vec![];
    for &n in &sorted_nodes {
        let Some(cand) = candidate(model, n, weight_first, x_dt, k)? else { continue };
        match buckets.iter_mut().find(|(class, _)| *class == cand.class) {
            Some((_, v)) => v.push(cand),
            None => buckets.push((cand.class, vec![cand])),
        }
    }
    let Some((class, group)) = buckets.into_iter().find(|(_, v)| v.len() >= 2) else {
        return Ok(None);
    };

    // Verify each sibling's declared output fact matches what the fused op
    // will produce for its slot.
    let splits: TVec<usize> = group.iter().map(|c| c.n).collect();
    let op = MetalMultiGemm { splits: splits.clone(), weight_first };
    for cand in &group {
        let out_fact = inner_fact(model.outlet_fact(OutletId::new(cand.node, 0))?);
        let mut expected = x_fact.shape.to_tvec();
        if weight_first {
            let m = expected[x_rank - 2].clone();
            expected[x_rank - 2] = cand.n.to_dim();
            expected[x_rank - 1] = m;
        } else {
            expected[x_rank - 1] = cand.n.to_dim();
        }
        if out_fact.datum_type != x_dt || out_fact.shape.to_tvec() != expected {
            if log_enabled() {
                eprintln!(
                    "multi-gemm skip group at {}: sibling {} fact {:?} != expected {:?} {:?}",
                    model.node(x.node).name,
                    model.node(cand.node).name,
                    out_fact,
                    x_dt,
                    expected
                );
            }
            return Ok(None);
        }
    }

    // Concatenate the sibling weights along output rows (bytewise: both
    // plain row-major [n, k] tensors and Q4_0 row blocks are contiguous
    // per-row layouts).
    let n_total: usize = splits.iter().sum();
    let row_bytes = match class {
        WeightClass::Q40 => (k / Q4_0.block_len()) * Q4_0.block_bytes(),
        WeightClass::Plain(w_dt) => k * w_dt.size_of(),
    };
    let mut bytes: Vec<u8> = Vec::with_capacity(n_total * row_bytes);
    let mut w_dt = x_dt;
    for cand in &group {
        let const_op = model
            .node(cand.w_outlet.node)
            .op_as::<Const>()
            .context("expected a Const weight node")?;
        w_dt = inner_fact(model.outlet_fact(cand.w_outlet)?).datum_type;
        let host: Arc<Tensor> = match const_op.val().as_device_tensor() {
            Some(dev) => dev.to_host()?,
            None => Arc::clone(const_op.val()),
        };
        match class {
            WeightClass::Q40 => {
                let bqs = as_q40_tensor(&host).context("expected a Q4_0 host tensor")?;
                ensure!(
                    bqs.value().len() == cand.n * row_bytes,
                    "Q4_0 blob size {} != expected {} for [{}, {k}]",
                    bqs.value().len(),
                    cand.n * row_bytes,
                    cand.n
                );
                bytes.extend_from_slice(bqs.value());
            }
            WeightClass::Plain(_) => {
                ensure!(host.datum_type().is_copy());
                let blob = unsafe { host.as_bytes() };
                ensure!(blob.len() == cand.n * row_bytes);
                bytes.extend_from_slice(blob);
            }
        }
    }
    let cat_const = match class {
        WeightClass::Q40 => {
            let fact = BlockQuantFact::new(Box::new(Q4_0), tvec![n_total, k]);
            let storage = BlockQuantStorage::new(
                Box::new(Q4_0),
                n_total,
                k,
                Arc::new(Blob::from_bytes(&bytes)?),
            )?;
            Const::new_with_exotic_fact(
                Arc::new(storage.into_tensor_with_shape(w_dt, &[n_total, k])),
                Box::new(fact),
            )?
        }
        WeightClass::Plain(dt) => {
            Const::new(Arc::new(unsafe { Tensor::from_raw_dt(dt, &[n_total, k], &bytes)? }))?
        }
    };
    let device_const = crate::transform::convert_const(&cat_const)?;

    let mut patch = TypedModelPatch::default();
    let x_tap = patch.tap_model(model, x)?;
    let name = format!("{}.multi_gemm", model.node(group[0].node).name);
    let w_tap = patch.wire_node(format!("{name}.weights"), device_const, &[])?[0];
    let outs = patch.wire_node(&name, op, &[x_tap, w_tap])?;
    for (i, cand) in group.iter().enumerate() {
        patch.shunt_outside(model, OutletId::new(cand.node, 0), outs[i])?;
    }
    if log_enabled() {
        eprintln!(
            "multi-gemm fused {} siblings of {} (weight_first={weight_first}, class={class:?}, k={k}, splits={:?})",
            group.len(),
            model.node(x.node).name,
            splits
        );
    }
    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalRuntime;
    use crate::kernels::matmul::MetalGemmImplKind;
    use crate::transform::MetalTransform;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::element_wise::ElementWiseOp;
    use tract_core::ops::nn::Silu;
    use tract_core::transform::ModelTransform;
    use tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0};

    fn add_q40_const(model: &mut TypedModel, name: &str, tensor: Tensor) -> TractResult<OutletId> {
        let shape = tensor.shape().to_vec();
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0);
        let m: usize = shape[..shape.len() - 1].iter().product();
        let quant = Q4_0.quant_f32(tensor.try_as_plain()?.as_slice::<f32>()?)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant))?;
        let packed = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), &shape));
        let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
        Ok(model.wire_node(
            name,
            Const::new_with_exotic_fact(packed, Box::new(fact))?,
            &[],
        )?[0])
    }

    fn make(rng_state: &mut u64, shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        tract_ndarray::ArrayD::from_shape_vec(shape.to_vec(), data).unwrap().into_tensor()
    }

    fn matmul() -> PrefixMatMul {
        PrefixMatMul {
            transpose_a: false,
            transpose_b: true,
            transpose_c: false,
            quantize_output: None,
            operating_dt: Some(DatumType::F32),
        }
    }

    fn check_fused(
        model: TypedModel,
        x_data: Tensor,
        gemm_impl: Option<MetalGemmImplKind>,
    ) -> TractResult<()> {
        let mut transformed = model.clone();
        MetalTransform { gemm_impl }.transform(&mut transformed)?;
        ensure!(
            transformed.nodes().iter().any(|n| n.op_is::<MetalMultiGemm>()),
            "Metal transform did not fuse sibling gemms (no MetalMultiGemm)"
        );
        let expected =
            DefaultRuntime.prepare(model.clone())?.run(tvec![x_data.clone().into_tvalue()])?;
        // MetalRuntime applies its own (ggml-defaulted) MetalTransform.
        let actual = MetalRuntime.prepare(model)?.run(tvec![x_data.into_tvalue()])?;
        for (a, e) in actual.iter().zip(expected.iter()) {
            let a = a.clone().into_tensor().cast_to::<f32>()?.into_owned();
            let e = e.clone().into_tensor().cast_to::<f32>()?.into_owned();
            a.close_enough(&e, Approximation::VeryApproximate)?;
        }
        Ok(())
    }

    /// Activation-first siblings with Q4_0 weights (the qkv / in-proj shape).
    fn check_q40_siblings(m: usize) -> TractResult<()> {
        let (k, n1, n2, n3) = (64, 48, 32, 16);
        let mut rng = 4242u64;
        let mut model = TypedModel::default();
        let x = model.add_source("x", f16::datum_type().fact([1, m, k]))?;
        let x_f32 = model.wire_node(
            "x_f32",
            tract_core::ops::cast::cast(DatumType::F32),
            &[x],
        )?[0];
        let mut outputs = tvec![];
        for (i, n) in [n1, n2, n3].into_iter().enumerate() {
            let w = add_q40_const(&mut model, &format!("w{i}"), make(&mut rng, &[1, n, k]))?;
            outputs.push(model.wire_node(format!("mm{i}"), matmul(), &[x_f32, w])?[0]);
        }
        model.select_output_outlets(&outputs)?;
        let x_data = make(&mut rng, &[1, m, k]).cast_to::<f16>()?.into_owned();
        check_fused(model, x_data, None)
    }

    #[test]
    fn q40_siblings_decode() -> TractResult<()> {
        check_q40_siblings(1)
    }

    #[test]
    fn q40_siblings_prefill() -> TractResult<()> {
        check_q40_siblings(17)
    }

    /// Activation-first siblings with plain f16 weights (the GDN
    /// in-proj-b/a shape). The activation passes through a device op so the
    /// siblings share one device outlet, as in a real model.
    fn check_f16_siblings(m: usize) -> TractResult<()> {
        let (k, n1, n2) = (64, 8, 8);
        let mut rng = 777u64;
        let mut model = TypedModel::default();
        let x = model.add_source("x", f16::datum_type().fact([1, m, k]))?;
        let h = model.wire_node("act", ElementWiseOp(Box::new(Silu {}), None), &[x])?[0];
        let mm = PrefixMatMul { operating_dt: None, ..matmul() };
        let mut outputs = tvec![];
        for (i, n) in [n1, n2].into_iter().enumerate() {
            let w = model.add_const(
                format!("w{i}"),
                make(&mut rng, &[1, n, k]).cast_to::<f16>()?.into_owned(),
            )?;
            outputs.push(model.wire_node(format!("mm{i}"), mm.clone(), &[h, w])?[0]);
        }
        model.select_output_outlets(&outputs)?;
        let x_data = make(&mut rng, &[1, m, k]).cast_to::<f16>()?.into_owned();
        check_fused(model, x_data, Some(MetalGemmImplKind::Ggml))
    }

    #[test]
    fn f16_siblings_decode() -> TractResult<()> {
        check_f16_siblings(1)
    }

    #[test]
    fn f16_siblings_prefill() -> TractResult<()> {
        check_f16_siblings(9)
    }

    /// Weight-first siblings (W @ x, f16 everywhere), forced onto the ggml
    /// backend as the qwen linear-attention in-projections are.
    fn check_weight_first_siblings(m: usize) -> TractResult<()> {
        let (k, n1, n2) = (64, 8, 12);
        let mut rng = 999u64;
        let mut model = TypedModel::default();
        let x = model.add_source("x", f16::datum_type().fact([1, m, k]))?;
        let h = model.wire_node("act", ElementWiseOp(Box::new(Silu {}), None), &[x])?[0];
        let mm = PrefixMatMul {
            transpose_a: false,
            transpose_b: true,
            transpose_c: false,
            quantize_output: None,
            operating_dt: Some(DatumType::F16),
        };
        let mut outputs = tvec![];
        for (i, n) in [n1, n2].into_iter().enumerate() {
            let w = model.add_const(
                format!("w{i}"),
                make(&mut rng, &[1, n, k]).cast_to::<f16>()?.into_owned(),
            )?;
            outputs.push(model.wire_node(format!("mm{i}"), mm.clone(), &[w, h])?[0]);
        }
        model.select_output_outlets(&outputs)?;
        let x_data = make(&mut rng, &[1, m, k]).cast_to::<f16>()?.into_owned();
        check_fused(model, x_data, Some(MetalGemmImplKind::Ggml))
    }

    #[test]
    fn weight_first_siblings_decode() -> TractResult<()> {
        check_weight_first_siblings(1)
    }

    #[test]
    fn weight_first_siblings_prefill() -> TractResult<()> {
        check_weight_first_siblings(13)
    }
}
