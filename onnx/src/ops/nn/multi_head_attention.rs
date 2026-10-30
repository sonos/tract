use crate::model::{ParsingContext, optional_inputs, optional_outputs};
use crate::pb::NodeProto;
use tract_core::ops::change_axes::AxisOp;
use tract_hir::internal::*;
use tract_transformers::ops::sdpa::Sdpa;

// com.microsoft MultiHeadAttention (scoped: unpacked Q/K/V, bidirectional).
//   inputs:  query(0), key(1), value(2), bias(3?), key_padding_mask(4?), attention_bias(5?),
//            past_key(6?), past_value(7?)
//   outputs: output(0), present_key(1?), present_value(2?)
// Standard (non-causal) multi-head attention lowered onto Sdpa, with attention_bias as Sdpa's
// additive mask. The projection bias, key_padding_mask, packed QKV and past KV cache are
// rejected with clear errors.
const ATTENTION_BIAS: usize = 5;

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
            i == ATTENTION_BIAS || node.input[i].is_empty(),
            "MultiHeadAttention: optional input #{i} (bias / mask / past KV) is unsupported"
        );
    }
    let has_bias = optional_inputs(node).nth(ATTENTION_BIAS).unwrap().is_some();
    let mut oo = optional_outputs(node).skip(1);
    let present_k = oo.next().unwrap().is_some();
    let present_v = oo.next().unwrap().is_some();
    Ok((expand(MultiHeadAttention { num_heads, scale, has_bias, present_k, present_v }), vec![]))
}

#[derive(Debug, Clone)]
struct MultiHeadAttention {
    num_heads: usize,
    scale: Option<f32>,
    has_bias: bool,
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
        check_input_arity(inputs, 3 + self.has_bias as usize)?;
        check_output_arity(outputs, self.nboutputs()?)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        if self.has_bias {
            s.equals(&inputs[3].rank, 4)?;
        }
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
        let mut sdpa_inputs = tvec![q4, k4, v4];
        if self.has_bias {
            sdpa_inputs.push(inputs[3]);
        }
        let y4 = model.wire_node(format!("{prefix}.sdpa"), sdpa, &sdpa_inputs)?[0];

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

#[cfg(test)]
mod tests {
    use crate::pb::*;
    use prost::Message;
    use tract_hir::internal::*;

    fn float_value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        let dim = dims
            .iter()
            .map(|d| tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*d)),
            })
            .collect();
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto { dim }),
                })),
            }),
            doc_string: String::new(),
        }
    }

    fn values(len: usize, seed: f32) -> Vec<f32> {
        (0..len).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
    }

    #[test]
    fn attention_bias_is_added_to_the_scaled_scores() -> TractResult<()> {
        let (seq, heads, head_dim) = (3usize, 2usize, 4usize);
        let hidden = heads * head_dim;
        let node = NodeProto {
            name: "mha".to_string(),
            op_type: "MultiHeadAttention".to_string(),
            domain: "com.microsoft".to_string(),
            input: ["q", "k", "v", "", "", "bias"].map(String::from).to_vec(),
            output: vec!["y".to_string()],
            attribute: vec![AttributeProto {
                name: "num_heads".to_string(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: heads as i64,
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        };
        let x_shape = [1, seq as i64, hidden as i64];
        let bias_shape = [1, heads as i64, seq as i64, seq as i64];
        let graph = GraphProto {
            node: vec![node],
            name: "g".to_string(),
            input: vec![
                float_value_info("q", &x_shape),
                float_value_info("k", &x_shape),
                float_value_info("v", &x_shape),
                float_value_info("bias", &bias_shape),
            ],
            output: vec![float_value_info("y", &x_shape)],
            ..GraphProto::default()
        };
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![
                OperatorSetIdProto { domain: String::new(), version: 18 },
                OperatorSetIdProto { domain: "com.microsoft".to_string(), version: 1 },
            ],
            graph: Some(graph),
            ..ModelProto::default()
        };
        let mut buf = vec![];
        model.encode(&mut buf).unwrap();
        let runnable =
            crate::onnx().model_for_read(&mut &*buf)?.into_optimized()?.into_runnable()?;

        let (q, k, v) =
            (values(seq * hidden, 0.), values(seq * hidden, 5.), values(seq * hidden, 9.));
        let mut bias = values(heads * seq * seq, 2.);
        bias[1] = -1e4;
        let scale = (head_dim as f32).sqrt().recip();
        let mut expected = vec![0f32; seq * hidden];
        for h in 0..heads {
            for i in 0..seq {
                let scores: Vec<f32> = (0..seq)
                    .map(|j| {
                        let dot: f32 = (0..head_dim)
                            .map(|d| {
                                q[i * hidden + h * head_dim + d] * k[j * hidden + h * head_dim + d]
                            })
                            .sum();
                        dot * scale + bias[(h * seq + i) * seq + j]
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..head_dim {
                    expected[i * hidden + h * head_dim + d] =
                        (0..seq).map(|j| exps[j] / sum * v[j * hidden + h * head_dim + d]).sum();
                }
            }
        }

        let shape = [1, seq, hidden];
        let out = runnable.run(tvec!(
            tensor1(&q).into_shape(&shape)?.into_tvalue(),
            tensor1(&k).into_shape(&shape)?.into_tvalue(),
            tensor1(&v).into_shape(&shape)?.into_tvalue(),
            tensor1(&bias).into_shape(&[1, heads, seq, seq])?.into_tvalue(),
        ))?;
        out[0].close_enough(&tensor1(&expected).into_shape(&shape)?, Approximation::Approximate)?;
        Ok(())
    }
}
