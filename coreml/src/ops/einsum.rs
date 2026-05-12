//! tract `EinSum` (1×1 conv-shaped) → MIL `conv` translator.
//!
//! After tract's `into_decluttered` + `f32_to_f16`, MobileNet's 1×1
//! pointwise convolutions surface as `EinSum` ops with axes signature
//! `IHW,OI->OHW` or `NIHW,OI->OHW` (the latter when the batch dim hasn't
//! been stripped). These are mathematically identical to a 1×1 Conv with
//! weight reshaped from `[O, I]` to `[O, I, 1, 1]`. We translate by
//! materialising a synthetic `ConvPlan` and reusing
//! [`crate::ops::conv::emit_conv_mil`].
//!
//! Other EinSum patterns (general matmul, attention, etc.) are skipped for
//! now — Phase 2 follow-up adds them via `mb.matmul` directly.

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::einsum::EinSum;

use crate::ops::conv::{ConvAnalysis, ConvPlan};

use super::{const_tensor, shape_to_concrete_i64};

/// Detect whether a node is a 1×1-conv-shaped EinSum and, if so, return a
/// `ConvAnalysis::Translatable(ConvPlan)` that the existing Conv emitter
/// can consume. Returns `ConvAnalysis::Skip(reason)` for anything else.
pub fn analyse_einsum(model: &TypedModel, node: &TypedNode) -> Result<ConvAnalysis> {
    let Some(es) = node.op_as::<EinSum>() else {
        return Ok(ConvAnalysis::Skip("not an EinSum".into()));
    };
    if es.q_params.is_some() {
        return Ok(ConvAnalysis::Skip("quantized EinSum".into()));
    }
    if es.operating_dt != DatumType::F16 {
        return Ok(ConvAnalysis::Skip(format!(
            "EinSum operating_dt {:?} (need F16)",
            es.operating_dt
        )));
    }

    let axes_str = format!("{}", es.axes);
    // Patterns tract emits for 1×1 conv post-declutter. The first pair has
    // both sides CHW (tract dropped batch); the second pair carries N through.
    // The `->NOHW` variants surface in SqueezeNet when the EinSum output feeds
    // a Concat (Concat wants matching rank, so tract synthesises N=1 in the
    // output even when the data input is CHW).
    // (prepend_n_in, _prepend_n_out, weight_has_o)
    //
    // weight_has_o = true: weight is `OI` (rank 2). out_channels = weight[0].
    // weight_has_o = false: weight is `I` (rank 1). out_channels = 1
    //   — this surfaces in MODNet's final mask projection (NIHW,I->NOHW with
    //   O=1 in the output and the weight just being a per-input-channel
    //   scaling vector that gets summed into a single output channel).
    let (prepend_n_in, _prepend_n_out, weight_has_o) = match axes_str.as_str() {
        "IHW,OI->OHW" => (true, true, true),     // both sides CHW
        "NIHW,OI->OHW" => (false, true, true),   // input NCHW, output CHW (rare)
        "IHW,OI->NOHW" => (true, false, true),   // input CHW, output NCHW (SqueezeNet pre-Concat)
        "NIHW,OI->NOHW" => (false, false, true), // both sides NCHW
        "NIHW,I->NOHW" => (false, false, false), // weight rank 1, single output channel
        other => {
            return Ok(ConvAnalysis::Skip(format!("EinSum axes {other:?} not a 1×1-conv pattern")));
        }
    };
    let prepend_n = prepend_n_in;

    if node.inputs.len() != 2 {
        return Ok(ConvAnalysis::Skip(format!(
            "EinSum has {} inputs (need 2 for 1×1 conv pattern)",
            node.inputs.len()
        )));
    }

    let data_fact = model.outlet_fact(node.inputs[0])?;
    if data_fact.datum_type != DatumType::F16 {
        return Ok(ConvAnalysis::Skip(format!(
            "EinSum data dtype {:?} (need F16)",
            data_fact.datum_type
        )));
    }
    let raw_data_shape = match shape_to_concrete_i64(&data_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(ConvAnalysis::Skip(format!(
                "EinSum data symbolic shape: {:?}",
                data_fact.shape
            )));
        }
    };
    let expected_rank = if prepend_n { 3 } else { 4 };
    if raw_data_shape.len() != expected_rank {
        return Ok(ConvAnalysis::Skip(format!(
            "EinSum data rank {} (expected {} for {})",
            raw_data_shape.len(),
            expected_rank,
            axes_str
        )));
    }

    // The data input's MLPackage shape (rank 4, prepending N=1 if needed).
    let input_shape: Vec<i64> = if prepend_n {
        std::iter::once(1).chain(raw_data_shape.iter().copied()).collect()
    } else {
        raw_data_shape.clone()
    };

    // Weight must be a const [O, I] tensor.
    let weight_owned = match const_tensor(model, node.inputs[1])? {
        Some(t) => t.into_owned(),
        None => {
            let n = &model.nodes[node.inputs[1].node];
            return Ok(ConvAnalysis::Skip(format!(
                "EinSum weight (input slot 1) is not a Const \
                 — actual op: {}",
                n.op.name()
            )));
        }
    };
    if weight_owned.datum_type() != DatumType::F16 {
        return Ok(ConvAnalysis::Skip(format!(
            "EinSum weight dtype {:?} (need F16)",
            weight_owned.datum_type()
        )));
    }
    let expected_w_rank = if weight_has_o { 2 } else { 1 };
    if weight_owned.rank() != expected_w_rank {
        return Ok(ConvAnalysis::Skip(format!(
            "EinSum weight rank {} (need {} for {})",
            weight_owned.rank(),
            expected_w_rank,
            axes_str
        )));
    }
    let (out_channels, in_channels) = if weight_has_o {
        (weight_owned.shape()[0], weight_owned.shape()[1])
    } else {
        // Weight is `[I]`, output_channels is implicit 1.
        (1usize, weight_owned.shape()[0])
    };
    if input_shape[1] as usize != in_channels {
        return Ok(ConvAnalysis::Skip(format!(
            "EinSum I dimension mismatch: weight {}, data {}",
            in_channels, input_shape[1]
        )));
    }

    // Reshape weight to OIHW with kernel 1×1. From `[O, I]` if it had O, or
    // from `[I]` otherwise (broadcast to a single-output-channel filter).
    let weight_4d = weight_owned.into_shape(&[out_channels, in_channels, 1, 1])?;

    // Synthetic zero bias of shape [O].
    let zero_bias_data: Vec<f16> = vec![f16::ZERO; out_channels];
    let bias = Tensor::from_shape::<f16>(&[out_channels], &zero_bias_data)?;

    let h = input_shape[2];
    let w = input_shape[3];
    let output_shape = vec![1, out_channels as i64, h, w];

    Ok(ConvAnalysis::Translatable(ConvPlan {
        input_shape,
        output_shape,
        weight: weight_4d,
        weight_shape: vec![out_channels as i64, in_channels as i64, 1, 1],
        bias,
        pad: vec![0, 0, 0, 0],
        strides: vec![1, 1],
        dilations: vec![1, 1],
        groups: 1,
        output_fact: node.outputs[0].fact.clone(),
    }))
}
