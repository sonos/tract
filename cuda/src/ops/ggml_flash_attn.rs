use crate::context::CUDA_STREAM;
use crate::kernels::ggml_flash_attn::GgmlFlashAttn;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, new)]
pub struct CudaFlashAttention {
    scale: f32,
    _is_causal: bool,
}

impl Op for CudaFlashAttention {
    fn name(&self) -> StaticName {
        "CudaFlashAttention".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaFlashAttention {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        CUDA_STREAM.with(|stream| {
            ensure!(inputs.len() == 4, "flash-attn expects [q, k, v, mask]");

            let q = inputs[0].to_device_tensor()?;
            let k = inputs[1].to_device_tensor()?;
            let v = inputs[2].to_device_tensor()?;
            let mask = inputs[3].to_device_tensor()?;

            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                q.datum_type(),
                &GgmlFlashAttn.output_shape(q.shape(), k.shape(), v.shape())?,
            )?;
            GgmlFlashAttn.dispatch_eval(stream, q, k, v, mask, self.scale, &output)?;
            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for CudaFlashAttention {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 4);
            let dt = facts[0].datum_type;

            ensure!(facts.iter().all(|f| f.rank() == 4));
            let shape =
                GgmlFlashAttn.output_shape(&facts[0].shape, &facts[1].shape, &facts[2].shape)?;
            let fact = dt.fact(shape);
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

// Transform code copied here to keep transform.rs clean

//.with_rule_for("causal_mask_as_extern", causal_mask_as_extern)
//.with_rule_for("full_attn_mask_as_neutral", neutral_mask_for_full_attn)

//fn convert_sdpa_to_cuda_flash_attn(
//    model: &TypedModel,
//    node: &TypedNode,
//    target: &mut TypedModel,
//    inputs: &mut [OutletId],
//    op: &Sdpa,
//) -> TractResult<TVec<OutletId>> {
//    // ---- Facts & quick guards -------------------------------------------------
//    let facts = model.node_input_facts(node.id)?;
//    ensure!(!op.is_causal && facts.len() == 4, "FlashAttn requires non-causal SDPA with 4 inputs");
//
//    let [qf, kf, vf, mf] = [facts[0], facts[1], facts[2], facts[3]];
//    ensure!(kf.datum_type() == vf.datum_type(), "K/V dtypes must match");
//
//    // split inputs as (q, k, v, m)
//    let [q, k, v, m, ..] = &mut inputs[..] else {
//        bail!("need at least 4 inputs");
//    };
//
//    // ---- Small helpers --------------------------------------------------------
//    fn name(base: &str, suffix: &str) -> String {
//        format!("{base}{suffix}")
//    }
//
//    fn mut_cast(
//        target: &mut TypedModel,
//        node_name: &str,
//        dst: &mut OutletId,
//        have: DatumType,
//        want: DatumType,
//        suffix: &str,
//    ) -> TractResult<()> {
//        if have != want {
//            *dst = target.wire_node(
//                name(node_name, suffix),
//                ops::CudaCast::new(want).unwrap(),
//                &[*dst],
//            )?[0];
//        }
//        Ok(())
//    }
//
//    fn add_head_axis_if_rank3(
//        target: &mut TypedModel,
//        node_name: &str,
//        dst: &mut OutletId,
//        fact: &TypedFact,
//        suffix: &str,
//    ) -> TractResult<bool> {
//        if fact.rank() == 3 {
//            let ax = ops::CudaAxisOp::from_tract_core(AxisOp::Add(1));
//            *dst = target.wire_node(name(node_name, suffix), ax, &[*dst])?[0];
//            Ok(true)
//        } else {
//            ensure!(fact.rank() == 4, "Q/K/V must be rank 3 or 4");
//            Ok(false)
//        }
//    }
//
//    fn pad_last_two_dims(
//        target: &mut TypedModel,
//        node_name: &str,
//        dst: &mut OutletId,
//        pad_s: TDim,
//        pad_sp: TDim,
//        fill: Tensor,
//        suffix: &str,
//    ) -> TractResult<()> {
//        let mut pads = vec![(TDim::Val(0), TDim::Val(0)); 4];
//        pads[2].1 = pad_s;
//        pads[3].1 = pad_sp;
//        *dst = target.wire_node(
//            name(node_name, suffix),
//            ops::CudaPad::new(pads, PadMode::Constant(fill.into()))?,
//            &[*dst],
//        )?[0];
//        Ok(())
//    }
//
//    // ----- casts
//    let q_dt = qf.datum_type().unwrap();
//    let kv_dt = kf.datum_type().unwrap();
//    mut_cast(target, &node.name, k, kv_dt, DatumType::F16, ".cast_k")?;
//    mut_cast(target, &node.name, v, kv_dt, DatumType::F16, ".cast_v")?;
//    mut_cast(target, &node.name, q, q_dt, DatumType::F32, ".cast_q")?;
//
//    // ----- rank normalize (sequential to avoid overlapping borrows)
//    let mut added_head_axis = false;
//    added_head_axis |= add_head_axis_if_rank3(target, &node.name, q, qf, ".reshape_q")?;
//    added_head_axis |= add_head_axis_if_rank3(target, &node.name, k, kf, ".reshape_k")?;
//    added_head_axis |= add_head_axis_if_rank3(target, &node.name, v, vf, ".reshape_v")?;
//
//    let out_dim = kf.shape[kf.rank() - 1].to_i64()?;
//    ensure!(
//        matches!(out_dim, 64 | 80 | 96 | 112 | 128 | 256),
//        "Unsupported head dim (D): {out_dim}"
//    );
//    ensure!(kf.shape == vf.shape, "K and V shapes must be identical");
//
//    // ----- pad K/V seq to multiple of 256
//    let s_plus_p = kf.shape.dims()[qf.rank() - 2].clone();
//    let s_plus_p_to_256 = ((s_plus_p.clone() + 255) / 256) * 256 - s_plus_p;
//
//    let zero_f16: Arc<Tensor> = tensor0(f16::from_f32(0.0)).into();
//    // Only pad dim=2 (S+P) for K/V
//    let mut pads_kv = vec![(TDim::Val(0), TDim::Val(0)); 4];
//    pads_kv[2].1 = s_plus_p_to_256.clone();
//    *k = target.wire_node(
//        name(&node.name, ".pad_k"),
//        ops::CudaPad::new(pads_kv.clone(), PadMode::Constant(zero_f16.clone()))?,
//        &[*k],
//    )?[0];
//    *v = target.wire_node(
//        name(&node.name, ".pad_v"),
//        ops::CudaPad::new(pads_kv, PadMode::Constant(zero_f16))?,
//        &[*v],
//    )?[0];
//
//    // ----- mask: cast→reshape→pad
//    mut_cast(target, &node.name, m, mf.datum_type().unwrap(), DatumType::F16, ".cast_m")?;
//    if mf.rank() != 4 {
//        let add = 4 - mf.rank();
//        let ax =
//            ops::CudaAxisOp::from_tract_core(AxisOp::Reshape(0, tvec![], tvec![TDim::Val(1); add]));
//        *m = target.wire_node(name(&node.name, ".reshape_m"), ax, &[*m])?[0];
//    }
//    let s = qf.shape.dims()[qf.rank() - 2].clone();
//    let pad_s_to_16 = ((s.clone() + 15) / 16) * 16 - s;
//    let neg_inf_f16 = tensor0(-f16::infinity());
//    pad_last_two_dims(target, &node.name, m, pad_s_to_16, s_plus_p_to_256, neg_inf_f16, ".pad_m")?;
//
//    // ----- scale & op
//    let scale = op
//        .scale
//        .as_ref()
//        .map(|s| *s.to_scalar::<f32>().unwrap())
//        .unwrap_or(1.0 / (out_dim as f32).sqrt());
//    let sdpa = ops::CudaFlashAttention::new(scale, false);
//
//    let mut out = target.wire_node(node.name.clone(), sdpa, inputs)?;
//
//    if added_head_axis {
//        out = target.wire_node(
//            name(&node.name, ".reshape_out"),
//            ops::CudaAxisOp::from_tract_core(AxisOp::Rm(1)),
//            &out,
//        )?;
//    }
//    if q_dt != DatumType::F32 {
//        out = target.wire_node(
//            name(&node.name, ".cast_out"),
//            ops::CudaCast::new(q_dt).unwrap(),
//            &out,
//        )?;
//    }
//
//    Ok(out)
//}
