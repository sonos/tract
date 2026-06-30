use tract_core::ops::einsum::EinSum;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};
use tract_nnef::internal::*;

/// Routes an `activation · weightᵀ` matmul with a rank-2 f32 constant weight to [`W4A8MatMul`],
/// quantizing the weight to `Q4_0` in `[N, K]` layout so the int8-dot decode kernel replaces the
/// f32 GEMM. Matches the contraction (the axis shared by both inputs and absent from the output)
/// on the activation's last axis; bails on any other shape or when `K % 32 != 0`.
pub fn w4a8_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    prefix: &str,
    op: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(node.inputs.len() == 2 && op.q_params.is_none());
    let facts = model.node_input_facts(node.id)?;
    let Some(wslot) = facts
        .iter()
        .position(|f| f.konst.as_ref().is_some_and(|t| t.datum_type().is_float()) && f.rank() == 2)
    else {
        return Ok(None);
    };
    let aslot = 1 - wslot;

    let Some(k_axis) = op
        .axes
        .iter_all_axes()
        .find(|ax| ax.inputs[0].len() == 1 && ax.inputs[1].len() == 1 && ax.outputs[0].is_empty())
    else {
        return Ok(None);
    };
    let k_on_weight = k_axis.inputs[wslot][0];
    let k_on_act = k_axis.inputs[aslot][0];

    let wt = facts[wslot].konst.as_ref().unwrap();
    let (d0, d1) = (wt.shape()[0], wt.shape()[1]);
    let (n, k) = if k_on_weight == 1 { (d0, d1) } else { (d1, d0) };
    if k % 32 != 0 {
        return Ok(None);
    }
    // The kernel contracts the activation's last axis and emits N last; the output may carry the
    // remaining dims in a different (size-1-padded) rank, so require equal volume and reshape.
    let a_shape = facts[aslot].shape.to_tvec();
    let out_shape = node.outputs[0].fact.shape.to_tvec();
    let nd = n.to_dim();
    let a_free_vol = a_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != k_on_act)
        .fold(1.to_dim(), |acc, (_, d)| acc * d);
    let out_lead_vol = out_shape[..out_shape.len() - 1].iter().fold(1.to_dim(), |acc, d| acc * d);
    if k_on_act + 1 != a_shape.len() || out_shape.last() != Some(&nd) || a_free_vol != out_lead_vol
    {
        return Ok(None);
    }

    let wt = wt.cast_to::<f32>()?;
    let wv = wt.to_plain_array_view::<f32>()?;
    let wflat = wv.as_slice().unwrap();
    // arrange weight as row-major [N, K]
    let mut wnk = vec![0f32; n * k];
    if k_on_weight == 1 {
        wnk.copy_from_slice(wflat);
    } else {
        for ni in 0..n {
            for ki in 0..k {
                wnk[ni * k + ki] = wflat[ki * n + ni];
            }
        }
    }
    let bytes = Q4_0.quant_f32(&wnk)?.to_vec();

    let mut patch = TypedModelPatch::default();
    let a = patch.tap_model(model, node.inputs[aslot])?;
    let w = patch.add_const(format!("{prefix}.w4a8.weight"), tensor1(&bytes))?;
    let mut out = patch.wire_node(prefix, W4A8MatMul { n, k }, &[a, w])?[0];
    // The op keeps the activation's dims with N last; reshape to the einsum's output dims.
    let mut op_shape = a_shape;
    *op_shape.last_mut().unwrap() = nd;
    if op_shape != out_shape {
        out = patch.wire_node(
            format!("{prefix}.w4a8.reshape"),
            AxisOp::Reshape(0, op_shape, out_shape),
            &[out],
        )?[0];
    }
    patch.shunt_outside(model, node.id.into(), out)?;
    Ok(Some(patch))
}

/// CPU W4A8 matmul: `Y = A · dequant(W)ᵀ` computed as per-block int8 dot products (W4A8 — the
/// activation is quantized to int8 on the fly). `W` is a flat `Q4_0` byte tensor describing an
/// `[N, K]` weight; `A` is `[.., K]` and the output is `[.., N]`, each leading row of `A` an
/// independent activation row. Rows are computed by a single GEMM call so each weight block is
/// decoded once and reused across them.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct W4A8MatMul {
    pub n: usize,
    pub k: usize,
}

impl Op for W4A8MatMul {
    fn name(&self) -> StaticName {
        "W4A8MatMul".into()
    }
    op_as_typed_op!();
}

impl EvalOp for W4A8MatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        use rayon::prelude::*;
        let a = inputs[0].cast_to::<f32>()?;
        let av = a.to_plain_array_view::<f32>()?;
        let af = av.as_slice().unwrap();
        let w = inputs[1].cast_to::<u8>()?;
        let wv = w.to_plain_array_view::<u8>()?;
        let wb = wv.as_slice().unwrap();
        let (n, k) = (self.n, self.k);
        let rows = af.len() / k;
        let mut out = vec![0f32; rows * n];
        // Parallelize over output columns N: each task decodes its weight rows once and reuses
        // them across all activation rows, so the per-block decode stays amortized.
        let row_bytes = (k / Q4_0.block_len()) * Q4_0.block_bytes();
        let nchunk = n.div_ceil(rayon::current_num_threads().max(1)).max(1);
        if n <= nchunk {
            Q4_0.w4a8_gemm(wb, n, k, af, rows, &mut out)?;
        } else {
            let parts: TractResult<Vec<(usize, usize, Vec<f32>)>> = (0..n)
                .step_by(nchunk)
                .par_bridge()
                .map(|n0| {
                    let nc = (n0 + nchunk).min(n) - n0;
                    let mut oc = vec![0f32; rows * nc];
                    Q4_0.w4a8_gemm(
                        &wb[n0 * row_bytes..(n0 + nc) * row_bytes],
                        nc,
                        k,
                        af,
                        rows,
                        &mut oc,
                    )?;
                    Ok((n0, nc, oc))
                })
                .collect();
            for (n0, nc, oc) in parts? {
                for mi in 0..rows {
                    out[mi * n + n0..mi * n + n0 + nc].copy_from_slice(&oc[mi * nc..(mi + 1) * nc]);
                }
            }
        }
        let mut shape: TVec<usize> = a.shape().into();
        *shape.last_mut().unwrap() = n;
        Ok(tvec!(Tensor::from_shape(&shape, &out)?.into_tvalue()))
    }
}

impl TypedOp for W4A8MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.to_tvec();
        *shape.last_mut().unwrap() = self.n.to_dim();
        Ok(tvec!(f32::fact(shape)))
    }
    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::tract_linalg::block_quant::BlockQuant;

    // Both ONNX orientations: weight [K,N] ("mk,kn->mn") and [N,K] ("mk,nk->mn").
    fn check_wiring(axes: &str, w_shape: &[usize]) -> TractResult<()> {
        use tract_core::transform::ModelTransform;
        let (m, k) = (8usize, 128usize);
        let n = w_shape.iter().product::<usize>() / k;
        let wf: Vec<f32> =
            (0..w_shape.iter().product()).map(|i| ((i * 31 % 47) as f32 - 23.0) / 11.0).collect();
        let a = Tensor::from_shape(
            &[m, k],
            &(0..m * k).map(|i| ((i * 23 % 43) as f32 - 21.0) / 8.0).collect::<Vec<f32>>(),
        )?;

        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([m, k]))?;
        let w = model.add_const("w", Tensor::from_shape(w_shape, &wf)?)?;
        let out = model.wire_node(
            "mm",
            EinSum { axes: axes.parse()?, operating_dt: f32::datum_type(), q_params: None },
            &[x, w],
        )?;
        model.select_output_outlets(&out)?;
        let model = model.into_decluttered()?;

        let reference = model.clone().into_runnable()?.run(tvec!(a.clone().into_tvalue()))?;
        let reference = reference[0].to_plain_array_view::<f32>()?;
        let reference = reference.as_slice().unwrap();

        let mut wired = model;
        crate::rewriter::BlockQuantW4A8Transform.transform(&mut wired)?;
        assert!(
            wired.nodes().iter().any(|nd| nd.op.name() == "W4A8MatMul"),
            "W4A8MatMul was not wired in for {axes}"
        );

        let got = wired.into_runnable()?.run(tvec!(a.into_tvalue()))?;
        let got = got[0].to_plain_array_view::<f32>()?;
        let got = got.as_slice().unwrap();
        // W4A8 perturbs individual elements (esp. near-zero ones), so assert the result keeps
        // the same direction rather than per-element closeness.
        let dot: f64 = got.iter().zip(reference).map(|(&g, &r)| g as f64 * r as f64).sum();
        let ng: f64 = got.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt();
        let nr: f64 = reference.iter().map(|&r| (r as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (ng * nr);
        assert!(cosine > 0.95, "{axes}: cosine {cosine}");
        Ok(())
    }

    #[test]
    fn w4a8_rule_wires_weight_kn() -> TractResult<()> {
        check_wiring("mk,kn->mn", &[128, 256])
    }

    #[test]
    fn w4a8_rule_wires_weight_nk() -> TractResult<()> {
        check_wiring("mk,nk->mn", &[256, 128])
    }

    #[test]
    #[ignore = "perf: cargo test -p tract-transformers --release w4a8_op_bench -- --ignored --nocapture"]
    fn w4a8_op_bench() -> TractResult<()> {
        use std::time::Instant;
        let (m, n, k) = (64usize, 1536usize, 384usize); // MiniLM intermediate, prefill
        let wf: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 53) as f32 - 26.0) / 9.0).collect();
        let qbytes = Q4_0.quant_f32(&wf)?;
        let a = Tensor::from_shape(
            &[m, k],
            &(0..m * k).map(|i| ((i * 19 % 41) as f32 - 20.0) / 7.0).collect::<Vec<f32>>(),
        )?;
        let w = Tensor::from_shape(&[qbytes.len()], &qbytes)?;
        let op = W4A8MatMul { n, k };
        let inputs = tvec!(a.into_tvalue(), w.into_tvalue());
        for _ in 0..5 {
            op.eval(inputs.clone())?;
        }
        let iters = 100;
        let t = Instant::now();
        for _ in 0..iters {
            op.eval(inputs.clone())?;
        }
        eprintln!(
            "w4a8 op prefill m={m} n={n} k={k} ({} threads): {} us/iter",
            rayon::current_num_threads(),
            t.elapsed().as_micros() as usize / iters
        );
        Ok(())
    }

    #[test]
    fn w4a8_matmul_op_matches_dequant() -> TractResult<()> {
        let (m, n, k) = (6usize, 8usize, 64usize);
        let wf: Vec<f32> = (0..n * k).map(|i| ((i * 31 % 47) as f32 - 23.0) / 11.0).collect();
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 23 % 43) as f32 - 21.0) / 8.0).collect();
        let qbytes = Q4_0.quant_f32(&wf)?;

        let mut model = TypedModel::default();
        let ai = model.add_source("a", f32::fact([m, k]))?;
        let wi = model.add_const("w", Tensor::from_shape(&[qbytes.len()], &qbytes)?)?;
        let out = model.wire_node("mm", W4A8MatMul { n, k }, &[ai, wi])?;
        model.select_output_outlets(&out)?;

        let res =
            model.into_runnable()?.run(tvec!(Tensor::from_shape(&[m, k], &a)?.into_tvalue()))?;
        let got = res[0].to_plain_array_view::<f32>()?;
        let got = got.as_slice().unwrap();

        let wdeq = Q4_0.dequant_f32(&qbytes)?;
        let wdeq = wdeq.try_as_plain()?.as_slice::<f32>()?;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += wdeq[ni * k + ki] * a[mi * k + ki];
                }
                let g = got[mi * n + ni];
                assert!((g - acc).abs() <= 0.15 * acc.abs() + 1.0, "[{mi},{ni}] {g} vs {acc}");
            }
        }
        Ok(())
    }
}
