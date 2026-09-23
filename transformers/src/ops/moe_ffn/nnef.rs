use super::*;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_moe_ffn);
    registry.register_dumper(ser_opt_moe_ffn);
    registry.register_primitive(
        "tract_moe_ffn",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Scalar.tensor().named("wg"),
            TypeName::Scalar.tensor().named("w1"),
            TypeName::Scalar.tensor().named("w2"),
            TypeName::Scalar.tensor().named("w3"),
            // Optional biases (e.g. gpt-oss): router bias [E], gate bias
            // (w1) [E, H], up bias (w3) [E, H], down bias (w2) [E, D].
            TypeName::Scalar.tensor().named("wg_bias"),
            TypeName::Scalar.tensor().named("w1_bias"),
            TypeName::Scalar.tensor().named("w3_bias"),
            TypeName::Scalar.tensor().named("w2_bias"),
            TypeName::Integer.named("k"),
            TypeName::String.named("activation"),
            // How router logits become the top-k gate weights:
            //   softmax_topk: softmax over the top-k logits (Mixtral, Qwen w/
            //                 norm_topk_prob, gpt-oss)
            //   softmax_all:  softmax over ALL experts, gather top-k, no
            //                 renormalization (OLMoE, Qwen w/o norm_topk_prob)
            //   sigmoid:      per-expert sigmoid of the top-k logits (Llama4)
            //   raw:          raw top-k logits as weights
            TypeName::String.named("gate"),
            // Optional clamped-SwiGLU params (gpt-oss): when act_limit is set
            // the activation becomes gate.clamp(max=limit) /
            // up.clamp(+-limit) / glu = gate*sigmoid(alpha*gate) /
            // out = (up + 1) * glu.
            TypeName::Scalar.named("act_alpha"),
            TypeName::Scalar.named("act_limit"),
            // Expert tensor layout:
            //   canonical: w1/w3 [E,D,H], w2 [E,H,D] (existing exports)
            //   linear:    w1/w3 [E,H,D], w2 [E,D,H] (nn.Linear layout)
            TypeName::String.named("expert_layout"),
        ],
        // single output: tract is inference-only, so router_logits (a
        // training-time load-balancing-loss signal) is computed internally for
        // expert selection but never surfaced as an output.
        &[("output", TypeName::Scalar.tensor())],
        deser_moe_ffn,
    );
}

fn ser_moe_ffn(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &MoeFfn,
) -> TractResult<Option<Arc<RValue>>> {
    let indexes = op.input_idx();
    let optional = [
        ("w3", indexes.w3),
        ("wg_bias", indexes.wg_bias),
        ("w1_bias", indexes.w1_bias),
        ("w3_bias", indexes.w3_bias),
        ("w2_bias", indexes.w2_bias),
    ];
    ensure!(node.inputs.len() == 4 + optional.iter().filter(|(_, ix)| ix.is_some()).count());
    let gate = match op.gate {
        GateMode::SoftmaxTopk => "softmax_topk",
        GateMode::SoftmaxAll => "softmax_all",
        GateMode::Sigmoid => "sigmoid",
        GateMode::Raw => "raw",
    };
    let layout = match op.expert_layout {
        ExpertLayout::Canonical => "canonical",
        ExpertLayout::Linear => "linear",
    };
    let mut args = vec![
        ("k", numeric(i64::try_from(op.k)?)),
        ("activation", string(&op.activation)),
        ("gate", string(gate)),
        ("expert_layout", string(layout)),
    ];
    for (name, ix) in optional {
        if let Some(ix) = ix {
            args.push((name, ast.mapping[&node.inputs[ix]].as_ref().clone()));
        }
    }
    for (name, bits) in [("act_alpha", op.act_alpha_bits), ("act_limit", op.act_limit_bits)] {
        if let Some(bits) = bits {
            let value = f32::from_bits(bits);
            ensure!(
                value.is_finite(),
                "tract_moe_ffn {name} must be finite for NNEF serialization"
            );
            args.push((name, numeric(value)));
        }
    }
    let inputs = node.inputs[..4].iter().map(|o| ast.mapping[o].clone()).collect::<Vec<_>>();
    Ok(Some(invocation("tract_moe_ffn", &inputs, &args)))
}

fn ser_opt_moe_ffn(
    _ast: &mut IntoAst,
    _node: &TypedNode,
    _op: &OptMoeFfn,
) -> TractResult<Option<Arc<RValue>>> {
    bail!("OptMoeFfn contains packed execution plans; serialize MoeFfn before CPU codegen")
}

fn deser_moe_ffn(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let wg = invocation.named_arg_as(builder, "wg")?;
    let w1 = invocation.named_arg_as(builder, "w1")?;
    let w2 = invocation.named_arg_as(builder, "w2")?;
    let w3: Option<OutletId> = invocation.get_named_arg_as(builder, "w3")?;
    let wg_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "wg_bias")?;
    let w1_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w1_bias")?;
    let w3_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w3_bias")?;
    let w2_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w2_bias")?;
    let k: i64 = invocation.named_arg_as(builder, "k")?;
    ensure!(k > 0, "tract_moe_ffn k must be positive");
    let activation: String = invocation.named_arg_as(builder, "activation")?;
    let gate_str: String = invocation.named_arg_as(builder, "gate")?;
    let gate = GateMode::parse(&gate_str)?;
    let act_alpha: Option<f32> = invocation.get_named_arg_as(builder, "act_alpha")?;
    let act_limit: Option<f32> = invocation.get_named_arg_as(builder, "act_limit")?;
    let expert_layout_str: Option<String> =
        invocation.get_named_arg_as(builder, "expert_layout")?;
    let expert_layout = ExpertLayout::parse(expert_layout_str.as_deref().unwrap_or("canonical"))?;

    // Inputs are pushed in a fixed order so eval can recover their positions
    // from the has_* flags: x, wg, w1, w2, [w3], [wg_bias], [w1_bias],
    // [w3_bias], [w2_bias].
    let mut inputs = vec![x, wg, w1, w2];
    let has_w3 = w3.is_some();
    let has_wg_bias = wg_bias.is_some();
    let has_w1_bias = w1_bias.is_some();
    let has_w3_bias = w3_bias.is_some();
    let has_w2_bias = w2_bias.is_some();
    for opt in [w3, wg_bias, w1_bias, w3_bias, w2_bias].into_iter().flatten() {
        inputs.push(opt);
    }

    builder.wire(
        MoeFfn {
            k: usize::try_from(k)?,
            activation,
            gate,
            has_w3,
            has_wg_bias,
            has_w1_bias,
            has_w3_bias,
            has_w2_bias,
            act_alpha_bits: act_alpha.map(f32::to_bits),
            act_limit_bits: act_limit.map(f32::to_bits),
            expert_layout,
        },
        &inputs,
    )
}
