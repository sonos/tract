use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::array::TypedConcat;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::math::add;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_transformers::ops::sdpa::Sdpa;

pub fn attention(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let softcap = node.get_attr_opt::<f32>("softcap")?.unwrap_or(0.0);
    if softcap != 0.0 {
        bail!("Attention: softcap is not supported");
    }
    let qk_matmul_output_mode = node.get_attr_opt::<i64>("qk_matmul_output_mode")?.unwrap_or(0);
    if qk_matmul_output_mode != 0 {
        bail!("Attention: qk_matmul_output_mode is not supported");
    }

    let q_num_heads = node.get_attr_opt::<i64>("q_num_heads")?.map(|v| v as usize);
    let kv_num_heads = node.get_attr_opt::<i64>("kv_num_heads")?.map(|v| v as usize);
    let is_causal = node.get_attr_opt::<i64>("is_causal")?.unwrap_or(0) != 0;
    let scale = node.get_attr_opt::<f32>("scale")?;

    let have_nonpad_kv_seqlen = node.input.len() > 6 && !node.input[6].is_empty();
    if have_nonpad_kv_seqlen {
        bail!("Attention: nonpad_kv_seqlen input is not supported");
    }
    let have_mask = node.input.len() > 3 && !node.input[3].is_empty();
    let have_past_key = node.input.len() > 4 && !node.input[4].is_empty();
    let have_past_value = node.input.len() > 5 && !node.input[5].is_empty();

    let have_present_key = node.output.len() > 1 && !node.output[1].is_empty();
    let have_present_value = node.output.len() > 2 && !node.output[2].is_empty();

    Ok((
        expand(AttentionOp {
            q_num_heads,
            kv_num_heads,
            is_causal,
            scale,
            have_mask,
            have_past_key,
            have_past_value,
            have_present_key,
            have_present_value,
        }),
        vec![],
    ))
}

#[derive(Debug, Clone)]
struct AttentionOp {
    q_num_heads: Option<usize>,
    kv_num_heads: Option<usize>,
    is_causal: bool,
    scale: Option<f32>,
    have_mask: bool,
    have_past_key: bool,
    have_past_value: bool,
    have_present_key: bool,
    have_present_value: bool,
}

impl AttentionOp {
    fn mask_input_idx(&self) -> Option<usize> {
        self.have_mask.then_some(3)
    }

    fn past_key_input_idx(&self) -> Option<usize> {
        self.have_past_key.then_some(3 + self.have_mask as usize)
    }

    fn past_value_input_idx(&self) -> Option<usize> {
        self.have_past_value.then_some(3 + self.have_mask as usize + self.have_past_key as usize)
    }
}

fn wire_3d_to_4d(
    prefix: &str,
    model: &mut TypedModel,
    x: OutletId,
    total_dim: TDim,
    num_heads: usize,
) -> TractResult<OutletId> {
    let head_dim = total_dim.clone() / num_heads;
    let after_reshape = model.wire_node(
        format!("{prefix}.reshape"),
        AxisOp::Reshape(2, tvec![total_dim], tvec![num_heads.to_dim(), head_dim]),
        &[x],
    )?[0];
    // (B, S, H, D) → Move(2, 1) → (B, H, S, D)
    model
        .wire_node(format!("{prefix}.transpose"), AxisOp::Move(2, 1), &[after_reshape])
        .map(|v| v[0])
}

impl Expansion for AttentionOp {
    fn name(&self) -> StaticName {
        "OnnxAttention".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.have_present_key as usize + self.have_present_value as usize)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let n_in = 3
            + self.have_mask as usize
            + self.have_past_key as usize
            + self.have_past_value as usize;
        let n_out = 1 + self.have_present_key as usize + self.have_present_value as usize;
        check_input_arity(inputs, n_in)?;
        check_output_arity(outputs, n_out)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        // Output Y has same rank and same batch/seq dims as Q; head dim may differ in
        // diff-head-sizes attention (V head dim != Q head dim), so we only propagate rank.
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if self.have_present_key {
            s.equals(&inputs[0].datum_type, &outputs[1].datum_type)?;
        }
        if self.have_present_value {
            let pv_idx = 1 + self.have_present_key as usize;
            s.equals(&inputs[0].datum_type, &outputs[pv_idx].datum_type)?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let q_fact = model.outlet_fact(inputs[0])?.clone();
        let is_4d = q_fact.rank() == 4;
        let dt = q_fact.datum_type;
        let acc_dt = DatumType::F32;

        // Build 4D Q, K, V for Sdpa: (B, heads, S, head_dim)
        let (q4, k4, v4, q_hdim_3d) = if is_4d {
            (inputs[0], inputs[1], inputs[2], None)
        } else {
            let q_hdim = q_fact.shape[2].clone();
            let k_hdim = model.outlet_fact(inputs[1])?.shape[2].clone();
            let v_hdim = model.outlet_fact(inputs[2])?.shape[2].clone();
            let q_num_heads = self.q_num_heads.context("q_num_heads required for 3D Attention")?;
            let kv_num_heads =
                self.kv_num_heads.context("kv_num_heads required for 3D Attention")?;
            let q4 = wire_3d_to_4d(
                &format!("{prefix}.q"),
                model,
                inputs[0],
                q_hdim.clone(),
                q_num_heads,
            )?;
            let k4 = wire_3d_to_4d(&format!("{prefix}.k"), model, inputs[1], k_hdim, kv_num_heads)?;
            let v4 = wire_3d_to_4d(&format!("{prefix}.v"), model, inputs[2], v_hdim, kv_num_heads)?;
            (q4, k4, v4, Some(q_hdim))
        };

        // Handle KV cache: concat [past, current] along the sequence axis (2)
        let (k_for_attn, v_for_attn, present_k, present_v) =
            if self.have_past_key || self.have_present_key {
                let k_full = if self.have_past_key {
                    let past_k = inputs[self.past_key_input_idx().unwrap()];
                    model.wire_node(
                        format!("{prefix}.concat_k"),
                        TypedConcat { axis: 2 },
                        &[past_k, k4],
                    )?[0]
                } else {
                    k4
                };
                let v_full = if self.have_past_value {
                    let past_v = inputs[self.past_value_input_idx().unwrap()];
                    model.wire_node(
                        format!("{prefix}.concat_v"),
                        TypedConcat { axis: 2 },
                        &[past_v, v4],
                    )?[0]
                } else {
                    v4
                };
                let pk = self.have_present_key.then_some(k_full);
                let pv = self.have_present_value.then_some(v_full);
                (k_full, v_full, pk, pv)
            } else {
                (k4, v4, None, None)
            };

        // Build explicit ONNX mask from input (if provided); pad rank to 4
        let explicit_mask = if self.have_mask {
            let m = inputs[self.mask_input_idx().unwrap()];
            let m_rank = model.outlet_fact(m)?.rank();
            let mut m = m;
            for i in m_rank..4 {
                m = model.wire_node(format!("{prefix}.mask_add_axis_{i}"), AxisOp::Add(0), &[m])?
                    [0];
            }
            Some(m)
        } else {
            None
        };

        // Build causal mask: ONNX semantics are Q[i] sees K[j] iff j <= i (simple lower-tri,
        // no offset). When shapes are concrete, materialise an explicit (1,1,q_seq,kv_seq)
        // additive mask and let Sdpa run without is_causal so both branches agree.
        // When shapes are symbolic (e.g. dynamic seq len at runtime), fall back to Sdpa's
        // own is_causal flag, which is exact when q_seq == kv_seq (the normal training case).
        let (causal_mask, sdpa_is_causal) = if self.is_causal {
            let q_seq = model.outlet_fact(q4)?.shape[2].to_usize().ok();
            let kv_seq = model.outlet_fact(k_for_attn)?.shape[2].to_usize().ok();
            if let (Some(qs), Some(ks)) = (q_seq, kv_seq) {
                let arr = tract_ndarray::Array2::<f32>::from_shape_fn((qs, ks), |(i, j)| {
                    if j <= i { 0.0f32 } else { f32::NEG_INFINITY }
                });
                let mask_tensor: Tensor = arr.into();
                let c = model.add_const(format!("{prefix}.causal_mask"), mask_tensor)?;
                let mut m = c;
                for i in 0..2 {
                    m = model.wire_node(
                        format!("{prefix}.causal_mask_unsqueeze_{i}"),
                        AxisOp::Add(0),
                        &[m],
                    )?[0];
                }
                (Some(m), false)
            } else {
                (None, true)
            }
        } else {
            (None, false)
        };

        // Combine explicit mask + causal mask (both are additive bias terms)
        let mask = match (explicit_mask, causal_mask) {
            (Some(em), Some(cm)) => Some(
                wire_with_rank_broadcast(
                    format!("{prefix}.mask_combined"),
                    model,
                    add(),
                    &[em, cm],
                )?[0],
            ),
            (m, None) | (None, m) => m,
        };

        // Wire Sdpa
        let mut sdpa_inputs = tvec![q4, k_for_attn, v_for_attn];
        if let Some(m) = mask {
            sdpa_inputs.push(m);
        }
        let sdpa = Sdpa {
            scale: self.scale.map(tensor0),
            datum_type: dt,
            acc_datum_type: acc_dt,
            is_causal: sdpa_is_causal,
        };
        let y4 = model.wire_node(format!("{prefix}.sdpa"), sdpa, &sdpa_inputs)?[0];

        // For 3D output: Move(1,2) then merge head dims back.
        // Output shape is (B, S, q_heads * v_head_dim) — note: v_head_dim may differ from
        // q_head_dim in diff-head-sizes (MLA-style) attention.
        let y = if q_hdim_3d.is_some() {
            // (B, q_heads, S, v_head_dim) → Move(1,2) → (B, S, q_heads, v_head_dim)
            let y_transposed =
                model.wire_node(format!("{prefix}.y_transpose"), AxisOp::Move(1, 2), &[y4])?[0];
            let y4_fact = model.outlet_fact(y4)?.clone();
            let q_heads_dim = y4_fact.shape[1].clone();
            let v_head_dim = y4_fact.shape[3].clone();
            let y_hdim = q_heads_dim.clone() * v_head_dim.clone();
            // (B, S, q_heads, v_head_dim) → (B, S, q_heads*v_head_dim)
            model.wire_node(
                format!("{prefix}.y_reshape"),
                AxisOp::Reshape(2, tvec![q_heads_dim, v_head_dim], tvec![y_hdim]),
                &[y_transposed],
            )?[0]
        } else {
            y4
        };

        let mut result = tvec![y];
        if let Some(pk) = present_k {
            result.push(pk);
        }
        if let Some(pv) = present_v {
            result.push(pv);
        }
        Ok(result)
    }
}
