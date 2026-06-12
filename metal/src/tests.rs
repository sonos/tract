#[cfg(test)]
mod tests {
    use crate::MetalTransform;
    use crate::utils::with_borrowed_metal_stream;
    use tract_core::internal::*;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::math::{add, mul};
    use tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};
    use tract_core::transform::ModelTransform;
    use tract_gpu::memory::DeviceMemSchema;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn test_alloc_zero() -> TractResult<()> {
        with_borrowed_metal_stream(|_| Tensor::from_shape::<f32>(&[0], &[])?.into_device())?;
        Ok(())
    }

    fn wire_sdpa_layer(
        model: &mut TypedModel,
        name: impl ToString,
        q: OutletId,
        k: OutletId,
        v: OutletId,
    ) -> TractResult<TVec<OutletId>> {
        let name = name.to_string();

        // Reshape Q
        let q_shape = model.outlet_fact(q)?.shape.to_tvec();
        let embed_dim: TDim = q_shape[1].clone();
        let head_dim: TDim = q_shape[3].clone();
        let batch: TDim = q_shape[0].clone();
        let seq_len: TDim = q_shape[2].clone();
        ensure!(batch.to_i64()? == 1, "Input 'q' shape is {:?} (expect batch = 1)", q_shape);
        ensure!(q_shape.len() == 4, "Input 'q' shape is {:?} (expect 4D)", q_shape);
        let q_reshaped = model.wire_node(
            format!("q_reshape_{}", name),
            AxisOp::Reshape(
                0,
                q_shape.clone(),
                tvec![embed_dim.clone(), batch.clone(), seq_len.clone(), head_dim.clone(),],
            ),
            &[q],
        )?[0];

        // Reshape K
        let k_shape = model.outlet_fact(k)?.shape.to_tvec();
        ensure!(k_shape.len() == 4, "Input 'k' shape is {:?} (expect 4D)", k_shape);
        let seq_plus_prompt_len: TDim = k_shape[2].clone();

        let k_reshaped = model.wire_node(
            format!("k_reshape_{}", name),
            AxisOp::Reshape(
                0,
                k_shape.clone(),
                tvec![
                    embed_dim.clone(),
                    batch.clone(),
                    seq_plus_prompt_len.clone(),
                    head_dim.clone(),
                ],
            ),
            &[k],
        )?[0];

        // Compute Q * K^T
        let qk = model.wire_node(
            format!("qk_{}", name),
            PrefixMatMul {
                transpose_a: false,
                transpose_b: true,
                transpose_c: false,
                quantize_output: None,
                operating_dt: Some(DatumType::F32),
            },
            &[q_reshaped, k_reshaped],
        )?[0];

        let qk_squeezed = model.wire_node(
            format!("qk_squeezed_{}", name),
            AxisOp::Reshape(
                0,
                tvec![
                    embed_dim.clone(),
                    batch.clone(),
                    seq_len.clone(),
                    seq_plus_prompt_len.clone(),
                ],
                tvec![embed_dim.clone(), seq_len.clone(), seq_plus_prompt_len.clone(),],
            ),
            &[qk],
        )?[0];

        // Scale factor for attention
        let scale = model.add_const(
            format!("scale_{}", name),
            tensor3(&[[[1.0f32 / (head_dim.to_i64()? as f32).sqrt()]]]),
        )?;
        let qk_scaled =
            model.wire_node(format!("qk_scaled_{}", name), mul(), &[qk_squeezed, scale])?[0];

        // Mask QK
        let mask = model.add_const("mask", tensor3(&[[[1.0f32]]]))?;
        let qk_scaled_masked =
            model.wire_node(format!("qk_scaled_masked_{}", name), add(), &[qk_scaled, mask])?[0];

        // Apply softmax
        let attention = model.wire_node(
            format!("attention_weights_{}", name),
            Softmax::new(tvec![2], None, SoftmaxKind::Softmax(SoftmaxExp::Libc)),
            &[qk_scaled_masked],
        )?[0];

        // Reshape V
        let v_reshaped = model.wire_node(
            format!("v_reshape_{}", name),
            AxisOp::Reshape(
                0,
                k_shape,
                tvec![embed_dim.clone(), seq_plus_prompt_len.clone(), head_dim.clone(),],
            ),
            &[v],
        )?[0];

        // Multiply with V
        let output = model.wire_node(
            format!("attention_output_{}", name),
            PrefixMatMul {
                transpose_a: false,
                transpose_b: false,
                transpose_c: false,
                quantize_output: None,
                operating_dt: Some(DatumType::F32),
            },
            &[attention, v_reshaped],
        )?[0];

        // Reshape output
        let output_reshaped = model.wire_node(
            format!("output_reshape_{}", name),
            AxisOp::Reshape(
                0,
                tvec![embed_dim.clone(), seq_len.clone(), head_dim.clone(),],
                q_shape,
            ),
            &[output],
        )?;
        Ok(output_reshaped)
    }

    #[test]
    fn test_build_schema_from_model() -> TractResult<()> {
        // Given
        const EMBED_DIM: i64 = 32;
        const HEAD_DIM: i64 = 64;
        const SEQUENCE_LENGTH: i64 = 1;
        const PAST_SEQUENCE_LENGTH: i64 = 8;
        const EXPECTED_PEAK_SIZE: i64 = 9344;
        const EXPECTED_USAGE: f32 = 0.89;

        // Build a model with Scaled Dot-Product Attention (SDPA) layers
        let mut model = TypedModel::default();

        // Input shapes for Q, K, V
        let s = TDim::Sym(model.sym("S"));
        let p = TDim::Sym(model.sym("P"));
        let q_fact = f32::fact(tvec![1.into(), EMBED_DIM.into(), s.clone(), HEAD_DIM.into()]);
        let k_fact = f32::fact(tvec![1.into(), EMBED_DIM.into(), s + p, HEAD_DIM.into()]);
        let v_fact = k_fact.clone();

        // Create inputs for Q, K, V
        let q = model.add_source("q", q_fact)?;
        let k = model.add_source("k", k_fact)?;
        let v = model.add_source("v", v_fact)?;

        let outputs = wire_sdpa_layer(&mut model, "0", q, k, v)?;
        let outputs = wire_sdpa_layer(&mut model, "1", outputs[0], k, v)?;

        model.select_output_outlets(&outputs)?;

        // Transform model for Metal execution
        let model = MetalTransform::default().transform_into(model)?;

        // Get execution order
        let order = model.eval_order()?;

        // Hint symbol values
        let mut symbol_values = SymbolValues::default();
        symbol_values.set(&model.symbols.get("S").context("Missing symbol S")?, SEQUENCE_LENGTH);
        symbol_values
            .set(&model.symbols.get("P").context("Missing symbol P")?, PAST_SEQUENCE_LENGTH);

        // Build memory schema
        let schema = DeviceMemSchema::build(&model, &order, &symbol_values)?;

        // Verify number of nodes
        assert!(schema.model_num_nodes > 1, "Schema should contain at least 2 nodes");

        // Verify number of partitions
        assert!(schema.by_partition.len() > 1, "Schema should contain at least 2 partitions");

        // Verify steps
        assert_eq!(schema.by_steps.len(), order.len());
        for step in 0..schema.by_steps.len() {
            for partition in schema.by_partition.iter() {
                let partition_size = partition.eval_size_to_i64(&symbol_values)?;

                // No empty partition
                assert!(!partition.nodes.is_empty());

                if let Some(this) = partition.find_node_alive_at_step(step) {
                    // Node memory requirement should be <= the partition size
                    let node_size = this.mem_size.eval_to_i64(&symbol_values)?;
                    assert!(node_size <= partition_size);
                    assert!(node_size > 0);

                    // All nodes should have a valid lifetime
                    assert!(this.lifetime.start < this.lifetime.end);

                    // No other node in the partition should be alive at this step
                    for other in partition.nodes.iter().filter(|it| it.outlet_id != this.outlet_id)
                    {
                        assert!(
                            !other.lifetime.is_alive_at_step(step)
                                && other.lifetime.is_disjoint(&this.lifetime),
                            "Lifetime conflict @ step {}\n{:?}\n{:?}",
                            step,
                            this,
                            other
                        );
                    }

                    // This node should not be alive in another partition at the same step
                    for p in schema.by_partition.iter().filter(|it| it != &partition) {
                        if let Some(other) = p.find_node_alive_at_step(step) {
                            assert!(other.outlet_id != this.outlet_id);
                        }
                    }
                }
            }
        }

        // Verify schema usage
        let usage = schema.eval_usage(&symbol_values)?;
        assert!(usage >= EXPECTED_USAGE, "Usage {}, expected >= {}", usage, EXPECTED_USAGE);

        // Verify peak memory size
        let peak_memory_size = schema.eval_peak_memory_size(&symbol_values)?;
        assert_eq!(peak_memory_size, EXPECTED_PEAK_SIZE, "Peak memory size mismatch");

        Ok(())
    }

    // Slice C e2e: a real `Sdpa` op routes to MetalMfaSdpa via the metal transform
    // and matches the CPU explode path.
    #[test]
    fn sdpa_routes_to_mfa_and_matches_cpu() -> TractResult<()> {
        use crate::kernels::matmul::mfa::MetalMfaSdpa;
        let (h, s, d) = (4usize, 32usize, 64usize); // B=1, D%8 -> fusable
        let fact = f32::fact(tvec![
            TDim::from(1i64),
            TDim::from(h as i64),
            TDim::from(s as i64),
            TDim::from(d as i64)
        ]);
        let mut model = TypedModel::default();
        let q = model.add_source("q", fact.clone())?;
        let k = model.add_source("k", fact.clone())?;
        let v = model.add_source("v", fact.clone())?;
        let out = model.wire_node(
            "sdpa",
            tract_transformers::ops::sdpa::Sdpa {
                scale: None,
                datum_type: f32::datum_type(),
                acc_datum_type: f32::datum_type(),
                is_causal: false,
            },
            &[q, k, v],
        )?;
        model.select_output_outlets(&out)?;

        let mk = |seed: i64| -> TractResult<TValue> {
            let n = h * s * d;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i as i64 * 2654435761 + seed).rem_euclid(1000)) as f32 / 1000.0) - 0.5)
                .collect();
            Ok(Tensor::from_shape(&[1, h, s, d], &data)?.into_tvalue())
        };
        let (qt, kt, vt) = (mk(1)?, mk(2)?, mk(3)?);

        let cpu = model.clone().into_runnable()?;
        let cpu_out = cpu.run(tvec![qt.clone(), kt.clone(), vt.clone()])?;

        let metal = MetalTransform::default().transform_into(model)?;
        assert!(
            metal.nodes().iter().any(|n| n.op_is::<MetalMfaSdpa>()),
            "expected the Sdpa node to route to MetalMfaSdpa"
        );

        let metal_out = metal.into_runnable()?.run(tvec![qt, kt, vt])?;
        cpu_out[0]
            .clone()
            .into_tensor()
            .close_enough(&metal_out[0].clone().into_tensor(), Approximation::Approximate)?;
        Ok(())
    }

    // GQA (H_kv < H_q) predates the MFA kernel: the translator must decline and
    // the explode fallback must still match the CPU reference.
    #[test]
    fn gqa_sdpa_declines_mfa_and_matches_cpu() -> TractResult<()> {
        use crate::kernels::matmul::mfa::MetalMfaSdpa;
        let (hq, hkv, s, d) = (4usize, 2usize, 32usize, 64usize);
        let fact = |h: usize| {
            f32::fact(tvec![
                TDim::from(1i64),
                TDim::from(h as i64),
                TDim::from(s as i64),
                TDim::from(d as i64)
            ])
        };
        let mut model = TypedModel::default();
        let q = model.add_source("q", fact(hq))?;
        let k = model.add_source("k", fact(hkv))?;
        let v = model.add_source("v", fact(hkv))?;
        let out = model.wire_node(
            "sdpa",
            tract_transformers::ops::sdpa::Sdpa {
                scale: None,
                datum_type: f32::datum_type(),
                acc_datum_type: f32::datum_type(),
                is_causal: false,
            },
            &[q, k, v],
        )?;
        model.select_output_outlets(&out)?;

        let mk = |h: usize, seed: i64| -> TractResult<TValue> {
            let n = h * s * d;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i as i64 * 2654435761 + seed).rem_euclid(1000)) as f32 / 1000.0) - 0.5)
                .collect();
            Ok(Tensor::from_shape(&[1, h, s, d], &data)?.into_tvalue())
        };
        let (qt, kt, vt) = (mk(hq, 1)?, mk(hkv, 2)?, mk(hkv, 3)?);

        let cpu = model.clone().into_runnable()?;
        let cpu_out = cpu.run(tvec![qt.clone(), kt.clone(), vt.clone()])?;

        let metal = MetalTransform::default().transform_into(model)?;
        assert!(
            !metal.nodes().iter().any(|n| n.op_is::<MetalMfaSdpa>()),
            "GQA Sdpa must not route to MetalMfaSdpa"
        );

        let metal_out = metal.into_runnable()?.run(tvec![qt, kt, vt])?;
        cpu_out[0]
            .clone()
            .into_tensor()
            .close_enough(&metal_out[0].clone().into_tensor(), Approximation::Approximate)?;
        Ok(())
    }

    // Model-level A/B through the metal transform: a fused Sdpa (3 inputs ->
    // MetalMfaSdpa) vs the explode path (4-input Sdpa w/ neutral zero mask ->
    // gemm+softmax+gemm). The zero mask is neutral so both compute the same attn.
    //   cargo test -p tract-metal bench_sdpa_model_fused_vs_explode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_sdpa_model_fused_vs_explode() -> TractResult<()> {
        use crate::kernels::matmul::mfa::MetalMfaSdpa;
        use std::time::Instant;
        let (h, s, d) = (8usize, 1024usize, 64usize);
        let dim = |x: usize| TDim::from(x as i64);
        let qf = f32::fact(tvec![dim(1), dim(h), dim(s), dim(d)]);
        let mk = |with_mask: bool| -> TractResult<TypedModel> {
            let mut m = TypedModel::default();
            let q = m.add_source("q", qf.clone())?;
            let k = m.add_source("k", qf.clone())?;
            let v = m.add_source("v", qf.clone())?;
            let mut ins = vec![q, k, v];
            if with_mask {
                ins.push(m.add_source("mask", f32::fact(tvec![dim(1), dim(1), dim(s), dim(s)]))?);
            }
            let out = m.wire_node(
                "sdpa",
                tract_transformers::ops::sdpa::Sdpa {
                    scale: None,
                    datum_type: f32::datum_type(),
                    acc_datum_type: f32::datum_type(),
                    is_causal: false,
                },
                &ins,
            )?;
            m.select_output_outlets(&out)?;
            MetalTransform::default().transform_into(m)
        };
        let fused_m = mk(false)?;
        let explode_m = mk(true)?;
        assert!(
            fused_m.nodes().iter().any(|n| n.op_is::<MetalMfaSdpa>()),
            "3-input Sdpa should fuse to MetalMfaSdpa"
        );
        assert!(
            !explode_m.nodes().iter().any(|n| n.op_is::<MetalMfaSdpa>()),
            "4-input Sdpa should take the explode path"
        );
        let fused = fused_m.into_runnable()?;
        let explode = explode_m.into_runnable()?;
        let z =
            |sh: &[usize]| -> TractResult<TValue> { Ok(Tensor::zero::<f32>(sh)?.into_tvalue()) };
        let qkv: TVec<TValue> = tvec![z(&[1, h, s, d])?, z(&[1, h, s, d])?, z(&[1, h, s, d])?];
        let qkvm: TVec<TValue> =
            tvec![z(&[1, h, s, d])?, z(&[1, h, s, d])?, z(&[1, h, s, d])?, z(&[1, 1, s, s])?];
        let bench = |f: &dyn Fn() -> TractResult<()>| -> TractResult<f64> {
            for _ in 0..3 {
                f()?;
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..20 {
                    f()?;
                }
                best = best.min(t.elapsed().as_secs_f64() / 20.0);
            }
            Ok(best)
        };
        let ft = bench(&|| {
            fused.run(qkv.clone())?;
            Ok(())
        })?;
        let et = bench(&|| {
            explode.run(qkvm.clone())?;
            Ok(())
        })?;
        println!("\n  model-level Sdpa through the metal transform, f32, B=1 H={h} S={s} D={d}:");
        println!("  fused  (MetalMfaSdpa)        : {:.3} ms/run", ft * 1e3);
        println!("  explode (gemm+softmax+gemm)  : {:.3} ms/run", et * 1e3);
        println!("  end-to-end GAIN explode/fused = {:.2}x", et / ft);
        Ok(())
    }

    // Multi-layer A/B: N stacked Sdpa layers so input/output sync amortizes across
    // them -> isolates the real per-attention end-to-end gain (the single-op bench is
    // dominated by per-run sync/alloc overhead). All-attention, so this measures the
    // ATTENTION-portion gain (a real model dilutes it by the FFN share).
    //   cargo test -p tract-metal bench_sdpa_multilayer_fused_vs_explode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_sdpa_multilayer_fused_vs_explode() -> TractResult<()> {
        use crate::kernels::matmul::mfa::MetalMfaSdpa;
        use std::time::Instant;
        let (n, h, s, d) = (8usize, 8usize, 512usize, 64usize);
        let dim = |x: usize| TDim::from(x as i64);
        let qf = f32::fact(tvec![dim(1), dim(h), dim(s), dim(d)]);
        let mk = |with_mask: bool| -> TractResult<TypedModel> {
            let mut m = TypedModel::default();
            let mut cur = m.add_source("q", qf.clone())?;
            let k = m.add_source("k", qf.clone())?;
            let v = m.add_source("v", qf.clone())?;
            let mask = if with_mask {
                Some(m.add_source("mask", f32::fact(tvec![dim(1), dim(1), dim(s), dim(s)]))?)
            } else {
                None
            };
            for i in 0..n {
                let mut ins = vec![cur, k, v];
                if let Some(msk) = mask {
                    ins.push(msk);
                }
                cur = m.wire_node(
                    format!("sdpa{i}"),
                    tract_transformers::ops::sdpa::Sdpa {
                        scale: None,
                        datum_type: f32::datum_type(),
                        acc_datum_type: f32::datum_type(),
                        is_causal: false,
                    },
                    &ins,
                )?[0];
            }
            m.select_output_outlets(&[cur])?;
            MetalTransform::default().transform_into(m)
        };
        let fused_m = mk(false)?;
        let explode_m = mk(true)?;
        let n_fused = fused_m.nodes().iter().filter(|x| x.op_is::<MetalMfaSdpa>()).count();
        assert_eq!(n_fused, n, "all {n} layers should fuse");
        assert_eq!(explode_m.nodes().iter().filter(|x| x.op_is::<MetalMfaSdpa>()).count(), 0);
        let fused = fused_m.into_runnable()?;
        let explode = explode_m.into_runnable()?;
        let z =
            |sh: &[usize]| -> TractResult<TValue> { Ok(Tensor::zero::<f32>(sh)?.into_tvalue()) };
        let qkv: TVec<TValue> = tvec![z(&[1, h, s, d])?, z(&[1, h, s, d])?, z(&[1, h, s, d])?];
        let qkvm: TVec<TValue> =
            tvec![z(&[1, h, s, d])?, z(&[1, h, s, d])?, z(&[1, h, s, d])?, z(&[1, 1, s, s])?];
        let bench = |f: &dyn Fn() -> TractResult<()>| -> TractResult<f64> {
            for _ in 0..3 {
                f()?;
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..10 {
                    f()?;
                }
                best = best.min(t.elapsed().as_secs_f64() / 10.0);
            }
            Ok(best)
        };
        let ft = bench(&|| {
            fused.run(qkv.clone())?;
            Ok(())
        })?;
        let et = bench(&|| {
            explode.run(qkvm.clone())?;
            Ok(())
        })?;
        println!("\n  {n}-layer Sdpa stack, f32, B=1 H={h} S={s} D={d}:");
        println!("  fused  : {:.3} ms/run  ({:.3} ms/layer)", ft * 1e3, ft * 1e3 / n as f64);
        println!("  explode: {:.3} ms/run  ({:.3} ms/layer)", et * 1e3, et * 1e3 / n as f64);
        println!("  attention-portion GAIN explode/fused = {:.2}x", et / ft);
        Ok(())
    }
}
