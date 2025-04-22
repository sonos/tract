#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;
    use crate::MetalTransform;
    use tract_core::internal::*;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::math::{add, mul};
    use tract_core::ops::nn::{Softmax, SoftmaxExp};
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
            Softmax::new(tvec![2], None, SoftmaxExp::Libc),
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

        model.set_output_outlets(&outputs)?;

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
                    for other in partition.nodes.iter().filter(|it| it.node != this.node) {
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
                            assert!(other.node != this.node);
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
}
