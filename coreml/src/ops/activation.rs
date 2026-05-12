//! tract `ElementWiseOp` (pointwise activations) → MIL `sigmoid`/`tanh`/
//! `silu`/`gelu`/`hard_swish` translator.
//!
//! tract wraps pointwise activations in `ElementWiseOp(Box<dyn ElementWiseMiniOp>)`
//! where the mini-op's `name()` returns the activation name. We dispatch by
//! string match on that name — keeps one translator file for the whole family
//! and adapts naturally as tract adds new pointwise ops.
//!
//! Phase 3 (continuing): unblocks segmentation models like MediaPipe Selfie
//! (HardSwish + Sigmoid), MODNet (Sigmoid), and any model with attention
//! (GeLU). Pure pointwise so no shape complications — input shape == output
//! shape, dtype preserved.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::nn::LeakyRelu;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{
    DataType, tensor_type, tensor_type_scalar, tv_floats, tv_ints, tv_strings,
};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum ActivationAnalysis {
    Translatable(ActivationPlan),
    Skip(String),
}

pub struct ActivationPlan {
    /// MLPackage-side input/output shape (rank as-is — pointwise preserves rank).
    pub shape: Vec<i64>,
    /// MIL op name: "sigmoid", "tanh", "silu", "gelu", "hard_swish", "leaky_relu", ...
    pub mil_op: &'static str,
    /// Optional mode argument (e.g. for `gelu(mode="EXACT")`); empty means none.
    pub mode: Option<&'static str>,
    /// Optional fp32 scalar parameter (currently used for `leaky_relu`'s
    /// `alpha`). Wired into the MIL op's `alpha` named input when set.
    pub alpha: Option<f32>,
    pub output_fact: TypedFact,
}

pub fn analyse_activation(model: &TypedModel, node: &TypedNode) -> Result<ActivationAnalysis> {
    let Some(ew) = node.op_as::<ElementWiseOp>() else {
        return Ok(ActivationAnalysis::Skip("not an ElementWiseOp".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(ActivationAnalysis::Skip(format!(
            "ElementWise has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let name = ew.0.name();

    // LeakyRelu has a runtime parameter (`alpha`); handle it specially before
    // the no-parameter dispatch table. Surfaces in Real-ESRGAN (279× of them
    // — every Conv has a LeakyRelu after it; without translation the model
    // fragments into 375 small CoremlOps).
    if name.as_str() == "LeakyRelu" {
        let leaky = ew
            .0
            .downcast_ref::<LeakyRelu>()
            .ok_or_else(|| anyhow::anyhow!("ElementWise name='LeakyRelu' but downcast failed"))?;
        let fact = model.outlet_fact(node.inputs[0])?;
        if fact.datum_type != DatumType::F16 {
            return Ok(ActivationAnalysis::Skip(format!(
                "LeakyRelu input dtype {:?} (need F16)",
                fact.datum_type
            )));
        }
        let raw_in = match shape_to_concrete_i64(&fact.shape) {
            Some(s) => s,
            None => {
                return Ok(ActivationAnalysis::Skip(format!(
                    "LeakyRelu input symbolic shape: {:?}",
                    fact.shape
                )));
            }
        };
        if !(1..=5).contains(&raw_in.len()) {
            return Ok(ActivationAnalysis::Skip(format!(
                "LeakyRelu input rank {} (only 1..=5 supported)",
                raw_in.len()
            )));
        }
        let shape = super::rank::pad_to_rank_4(&raw_in);
        return Ok(ActivationAnalysis::Translatable(ActivationPlan {
            shape,
            mil_op: "leaky_relu",
            mode: None,
            alpha: Some(leaky.alpha),
            output_fact: node.outputs[0].fact.clone(),
        }));
    }

    let (mil_op, mode) = match name.as_str() {
        "Sigmoid" => ("sigmoid", None),
        "Tanh" => ("tanh", None),
        "Silu" => ("silu", None),
        "HardSwish" => ("hard_swish", None),
        // tract's GeluApproximate is the only Gelu variant in tract-core; it
        // uses the tanh approximation. Map to MIL `gelu(mode="TANH_APPROXIMATION")`.
        "GeluApproximate" => ("gelu", Some("TANH_APPROXIMATION")),
        "Gelu" => ("gelu", Some("EXACT")),
        // Erf surfaces in some BERT-class models — map directly.
        "Erf" => ("erf", None),
        // Rsqrt = 1 / sqrt(x) — appears in InstanceNorm/LayerNorm chains.
        "Rsqrt" => ("rsqrt", None),
        // Sqrt for completeness — pairs with Square in some normalization variants.
        "Sqrt" => ("sqrt", None),
        // Square = x * x — standalone form (also used inside Reduce<MeanOfSquares>'s
        // square+reduce_mean lowering). Surfaces in RVM's foreground-residual chain.
        "Square" => ("square", None),
        // Ln = natural log. Surfaces in audio preprocessors (log-mel features).
        "Ln" => ("log", None),
        // Recip = 1 / x. Surfaces in normalization chains and audio preprocessors.
        // MIL `inverse` is the elementwise reciprocal in the iOS17 opset (NOT
        // matrix inverse, which doesn't exist in MIL). `reciprocal` is the
        // older alias documented in some Apple docs but `inverse` is what
        // ships in iOS17+.
        "Recip" => ("inverse", None),
        // Cos / Sin — RoPE position-encoding primitives. Each surfaces once
        // per SmolLM2-class LLM (the cos/sin tables are computed at the
        // model entry and broadcast to every attention layer). MIL has them
        // direct (`cos`, `sin` in iOS17 opset).
        "Cos" => ("cos", None),
        "Sin" => ("sin", None),
        // Neg = -x. Surfaces 60× per SmolLM2-class transformer (one per
        // attention layer × the two RoPE halves, where each half is rotated
        // by negating one of its components: `[x, y]` → `[-y, x]`). Without
        // a translator, every Neg becomes a CPU residual that fragments the
        // graph at every attention layer; SmolLM2-135M went 0 → 183
        // CoremlOps after fixing the unmapped-Source panic, then a Neg
        // translator should collapse it further by absorbing the per-layer
        // RoPE residuals.
        "Neg" => ("neg", None),
        // Softplus, Selu, etc. would go here.
        other => {
            return Ok(ActivationAnalysis::Skip(format!(
                "ElementWiseOp '{other}' not yet translated"
            )));
        }
    };

    let fact = model.outlet_fact(node.inputs[0])?;
    if fact.datum_type != DatumType::F16 {
        return Ok(ActivationAnalysis::Skip(format!(
            "{mil_op} input dtype {:?} (need F16)",
            fact.datum_type
        )));
    }
    let raw_in = match shape_to_concrete_i64(&fact.shape) {
        Some(s) => s,
        None => {
            return Ok(ActivationAnalysis::Skip(format!(
                "{mil_op} input symbolic shape: {:?}",
                fact.shape
            )));
        }
    };
    if !(1..=5).contains(&raw_in.len()) {
        return Ok(ActivationAnalysis::Skip(format!(
            "{mil_op} input rank {} (only 1..=5 supported)",
            raw_in.len()
        )));
    }
    // Rank-padding correction (same convention as reduce.rs / rms_norm.rs /
    // binop.rs): the in-MLPackage tensor is rank-4-padded, even when tract's
    // own fact is rank 3 (CHW). Activation is pointwise so any size-1
    // padding is no-op semantically — we just need rank consistency with
    // the MLPackage value flowing in from a rank-4-padding upstream op.
    let shape = super::rank::pad_to_rank_4(&raw_in);

    Ok(ActivationAnalysis::Translatable(ActivationPlan {
        shape,
        mil_op,
        mode,
        alpha: None,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_activation_mil(
    plan: &ActivationPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let mut prelude: Vec<mil::Operation> = Vec::new();
    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();

    // Defensive input-rank normalization. analyse_activation pads the input
    // shape to rank-4 via `pad_to_rank_4` (same as binop/reduce/etc.). The
    // declared MIL output type (`plan.shape`) is therefore rank-4. But the
    // upstream MIL value may be rank-3 if it was emitted by an op that
    // doesn't pad (e.g. a general_matmul whose canonical-form output is
    // rank-3). MIL's element-wise activations (`tanh`, `sigmoid`, etc.)
    // preserve rank, so a rank-3 input would produce a rank-3 output but
    // we declared the output as rank-4 — MIL aborts with
    // "Output '0' has unexpected type ... Expected tensor<fp16, [...rank-3]>;
    //  got tensor<fp16, [1, ...rank-3]>". DFN3 surfaced this on a Tanh
    // consuming a rank-3 EinSum output.
    //
    // Fix: emit a leading `mb.reshape(x, plan.shape)` so the input is
    // materialized at the same rank as our declared output type. Metadata-
    // only when shapes already match. Same pattern as Conv/Concat/Slice.
    let in_rank4_name = format!("{output_name}_x_rank{}", plan.shape.len());
    let in_shape_n = format!("{output_name}_x_shape");
    let in_shape_op = op_const_immediate(
        &in_shape_n,
        tensor_type(DataType::Int32, &[plan.shape.len() as i64]),
        tv_ints(plan.shape.iter().map(|&v| v as i32).collect()),
    );
    let in_ty = tensor_type(DataType::Float16, &plan.shape);
    let mut reshape_inputs: HashMap<String, mil::Argument> = HashMap::new();
    reshape_inputs.insert("x".into(), arg_name(input_name));
    reshape_inputs.insert("shape".into(), arg_name(&in_shape_n));
    let in_reshape_op = mil::Operation {
        r#type: "reshape".into(),
        inputs: reshape_inputs,
        outputs: vec![mil::NamedValueType { name: in_rank4_name.clone(), r#type: Some(in_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(in_shape_op);
    prelude.push(in_reshape_op);
    inputs.insert("x".into(), arg_name(&in_rank4_name));

    // Optional const inputs (e.g., gelu's `mode`).
    if let Some(mode_str) = plan.mode {
        let mode_n = format!("{output_name}_mode");
        let mode_op = op_const_immediate(
            &mode_n,
            tensor_type_scalar(DataType::String),
            tv_strings(vec![mode_str.into()]),
        );
        inputs.insert("mode".into(), arg_name(&mode_n));
        prelude.push(mode_op);
    }

    // MIL `rsqrt`, `log`, and `inverse` take a required `epsilon` for
    // numerical stability:
    //   rsqrt:   y = 1 / sqrt(x + epsilon)
    //   log:     y = log(x + epsilon)
    //   inverse: y = 1 / (x + epsilon)
    // Default to 1e-12 — matches tract's pure semantics within FP16 noise.
    // Plain `sqrt` does NOT accept `epsilon` (CoreML rejects it as an
    // invalid param name); discovered via Parakeet preprocessor.
    if matches!(plan.mil_op, "rsqrt" | "log" | "inverse") {
        let eps_n = format!("{output_name}_epsilon");
        let eps_op = op_const_immediate(
            &eps_n,
            tensor_type_scalar(DataType::Float32),
            tv_floats(vec![1e-12]),
        );
        inputs.insert("epsilon".into(), arg_name(&eps_n));
        prelude.push(eps_op);
    }

    // LeakyRelu's `alpha` is a required fp32 scalar. tract carries it as
    // `LeakyRelu { alpha: f32 }` (default 0.01). MIL `leaky_relu(x, alpha)`
    // expects alpha as a scalar tensor input.
    if let Some(alpha) = plan.alpha {
        let alpha_n = format!("{output_name}_alpha");
        let alpha_op = op_const_immediate(
            &alpha_n,
            tensor_type_scalar(DataType::Float32),
            tv_floats(vec![alpha]),
        );
        inputs.insert("alpha".into(), arg_name(&alpha_n));
        prelude.push(alpha_op);
    }

    let out_ty = tensor_type(DataType::Float16, &plan.shape);

    // MIL has no `neg` op in our targeted opset (CoreML7 — `mb.neg` from the
    // Python helpers desugars to `mul(x, -1)` under the hood). Emit the
    // multiply-by-minus-one form directly. Same observable semantics.
    if plan.mil_op == "neg" {
        let neg_one_n = format!("{output_name}_neg_one");
        // F16 const must use RepeatedBytes (f16 byte representation), not
        // RepeatedFloats (which is f32 — 4 bytes/elem; CoreML rejects with
        // "Tensor storage and type have different number of elements").
        let neg_one_f16 = half::f16::from_f32(-1.0);
        let neg_one_bytes = neg_one_f16.to_le_bytes().to_vec();
        let neg_one_op = op_const_immediate(
            &neg_one_n,
            tensor_type(DataType::Float16, &[]),
            mil::TensorValue {
                value: Some(mil::tensor_value::Value::Bytes(mil::tensor_value::RepeatedBytes {
                    values: neg_one_bytes,
                })),
            },
        );
        let mut mul_inputs = inputs.clone();
        mul_inputs.insert("y".into(), arg_name(&neg_one_n));
        // `inputs` already has "x" pointing at the rank-padded reshape we
        // emitted earlier in this function, so reuse it.
        let mul_op = mil::Operation {
            r#type: "mul".into(),
            inputs: mul_inputs,
            outputs: vec![mil::NamedValueType {
                name: output_name.to_string(),
                r#type: Some(out_ty),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        };
        prelude.push(neg_one_op);
        prelude.push(mul_op);
        return Ok(prelude);
    }

    let act_op = mil::Operation {
        r#type: plan.mil_op.into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(act_op);
    Ok(prelude)
}
