use crate::model::{ParsingContext, optional_outputs};
use crate::pb::NodeProto;
use tract_core::ops::change_axes::AxisOp;
use tract_hir::internal::*;
use tract_transformers::ops::sdpa::Sdpa;

// com.microsoft MultiHeadAttention (scoped: unpacked Q/K/V, bidirectional).
//   inputs:  query(0), key(1), value(2), bias(3?), key_padding_mask(4?), attention_bias(5?),
//            past_key(6?), past_value(7?)
//   outputs: output(0), present_key(1?), present_value(2?)
// Standard (non-causal) multi-head attention lowered onto Sdpa. Bias, masks, packed QKV and
// past KV cache are rejected with clear errors.
pub fn multi_head_attention(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let num_heads: usize = node.get_attr("num_heads")?;
    let scale = node.get_attr_opt::<f32>("scale")?;
    ensure!(
        node.input.len() >= 3 && !node.input[1].is_empty() && !node.input[2].is_empty(),
        "MultiHeadAttention: requires unpacked query, key and value inputs"
    );
    for i in 3..node.input.len() {
        ensure!(
            node.input[i].is_empty(),
            "MultiHeadAttention: optional input #{i} (bias / mask / past KV) is unsupported"
        );
    }
    let mut oo = optional_outputs(node).skip(1);
    let present_k = oo.next().unwrap().is_some();
    let present_v = oo.next().unwrap().is_some();
    Ok((expand(MultiHeadAttention { num_heads, scale, present_k, present_v }), vec![]))
}

#[derive(Debug, Clone)]
struct MultiHeadAttention {
    num_heads: usize,
    scale: Option<f32>,
    present_k: bool,
    present_v: bool,
}

// [B, S, heads*head_size] -> [B, heads, S, head_size]
fn to_4d(
    model: &mut TypedModel,
    prefix: &str,
    x: OutletId,
    total: TDim,
    heads: usize,
) -> TractResult<OutletId> {
    let head_dim = total.clone() / heads;
    let reshaped = model.wire_node(
        format!("{prefix}.reshape"),
        AxisOp::Reshape(2, tvec![total], tvec![heads.to_dim(), head_dim]),
        &[x],
    )?[0];
    Ok(model.wire_node(format!("{prefix}.transpose"), AxisOp::Move(2, 1), &[reshaped])?[0])
}

impl Expansion for MultiHeadAttention {
    fn name(&self) -> StaticName {
        "MultiHeadAttention".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.present_k as usize + self.present_v as usize)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, self.nboutputs()?)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        let nh = self.num_heads;
        if self.present_k {
            s.equals(&inputs[0].datum_type, &outputs[1].datum_type)?;
            s.given(&inputs[1].shape, move |s, ks| {
                s.equals(
                    &outputs[1].shape,
                    tvec![ks[0].clone(), nh.to_dim(), ks[1].clone(), ks[2].clone() / nh],
                )
            })?;
        }
        if self.present_v {
            let vi = 1 + self.present_k as usize;
            s.equals(&inputs[0].datum_type, &outputs[vi].datum_type)?;
            s.given(&inputs[2].shape, move |s, vs| {
                s.equals(
                    &outputs[vi].shape,
                    tvec![vs[0].clone(), nh.to_dim(), vs[1].clone(), vs[2].clone() / nh],
                )
            })?;
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
        let dt = q_fact.datum_type;
        ensure!(q_fact.rank() == 3, "MultiHeadAttention: expected 3D query [B, S, hidden]");
        let q_hidden = q_fact.shape[2].clone();
        let k_hidden = model.outlet_fact(inputs[1])?.shape[2].clone();
        let v_hidden = model.outlet_fact(inputs[2])?.shape[2].clone();

        let q4 = to_4d(model, &format!("{prefix}.q"), inputs[0], q_hidden, self.num_heads)?;
        let k4 = to_4d(model, &format!("{prefix}.k"), inputs[1], k_hidden, self.num_heads)?;
        let v4 = to_4d(model, &format!("{prefix}.v"), inputs[2], v_hidden, self.num_heads)?;

        // Bidirectional attention (no causal mask). Sdpa default scale is 1/sqrt(head_dim).
        let sdpa = Sdpa {
            scale: self.scale.map(tensor0),
            datum_type: dt,
            acc_datum_type: DatumType::F32,
            is_causal: false,
        };
        let y4 = model.wire_node(format!("{prefix}.sdpa"), sdpa, &[q4, k4, v4])?[0];

        let y_t = model.wire_node(format!("{prefix}.y_transpose"), AxisOp::Move(1, 2), &[y4])?[0];
        let yf = model.outlet_fact(y4)?.clone();
        let (heads_dim, head_dim) = (yf.shape[1].clone(), yf.shape[3].clone());
        let y = model.wire_node(
            format!("{prefix}.y_reshape"),
            AxisOp::Reshape(
                2,
                tvec![heads_dim.clone(), head_dim.clone()],
                tvec![heads_dim * head_dim],
            ),
            &[y_t],
        )?[0];

        let mut out = tvec!(y);
        if self.present_k {
            out.push(k4);
        }
        if self.present_v {
            out.push(v4);
        }
        Ok(out)
    }
}
