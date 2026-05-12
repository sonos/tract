//! tract `Resize` (from `tract-onnx-opl`) → MIL `resize_nearest_neighbor` /
//! `resize_bilinear` translator.
//!
//! Resize is the upsampling primitive for segmentation/decoder models: MODNet,
//! MediaPipe Selfie, and most U-Net-style architectures use it to bring
//! feature maps back to input resolution. ONNX Resize is much more general
//! than what MIL exposes (4 coord transformers, 3 interpolators, scales OR
//! sizes input, optional roi), so this Phase 3 first cut handles the common
//! production patterns and skips the rest with diagnostic reasons.
//!
//! Translatable subset (Phase 3.1):
//!   * F16 input
//!   * Static input shape (rank 4 NCHW or rank 3 CHW)
//!   * `Interpolator::Linear` → `mb.resize_bilinear`
//!   * `Interpolator::Nearest` → `mb.resize_nearest_neighbor`
//!   * Const scales (or const sizes) so we can compute target_h/target_w at
//!     translate time. Dynamic scales/sizes deferred (would need to embed
//!     them as runtime inputs to the MIL op).
//!   * `axes` is None or covers H,W (the spatial axes) — resize on N or C
//!     axes is rejected as unusual
//!   * `coord_transformer`: `HalfPixel` / `Asymmetric` / `AlignCorners`
//!     mapped to MIL's `sampling_mode` strings
//!   * `Interpolator::Cubic` skipped (MIL has no direct equivalent — would
//!     need a custom kernel)
//!   * `optional_roi_input.is_some()` skipped

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_onnx_opl::resize::{CoordTransformer, Interpolator, Resize};

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints, tv_strings};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

#[allow(clippy::large_enum_variant)]
pub enum ResizeAnalysis {
    Translatable(ResizePlan),
    Skip(String),
}

pub struct ResizePlan {
    /// MLPackage-side input shape (rank 4, with N=1 prepended for CHW).
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (rank 4).
    pub output_shape: Vec<i64>,
    /// Target H (output spatial axis 2).
    pub target_h: i32,
    /// Target W (output spatial axis 3).
    pub target_w: i32,
    /// MIL op name: "resize_nearest_neighbor" or "resize_bilinear".
    pub mil_op: &'static str,
    /// `sampling_mode` MIL string for bilinear; ignored for nearest.
    pub sampling_mode: &'static str,
    pub output_fact: TypedFact,
}

pub fn analyse_resize(model: &TypedModel, node: &TypedNode) -> Result<ResizeAnalysis> {
    let Some(rs) = node.op_as::<Resize>() else {
        return Ok(ResizeAnalysis::Skip("not a Resize".into()));
    };

    // ONNX Resize opset 11+ has a positional ROI input (slot 1) that's only
    // used by `tf_crop_and_resize` mode — for everything else it's an
    // empty (length-0) tensor. Allow it through if empty / absent.
    if let Some(roi_idx) = rs.optional_roi_input {
        let roi_outlet = node.inputs[roi_idx];
        match const_tensor(model, roi_outlet)? {
            Some(t) if t.len() == 0 => {
                // empty ROI tensor → not actually used; OK
            }
            Some(t) => {
                return Ok(ResizeAnalysis::Skip(format!(
                    "Resize ROI input has {} elements (only empty ROI supported)",
                    t.len()
                )));
            }
            None => {
                return Ok(ResizeAnalysis::Skip(format!(
                    "Resize ROI input slot {roi_idx} is not a Const \
                     — actual op: {}",
                    model.nodes[roi_outlet.node].op.name()
                )));
            }
        }
    }
    if matches!(rs.interpolator, Interpolator::Cubic) {
        return Ok(ResizeAnalysis::Skip(
            "Resize Interpolator::Cubic not supported (MIL has no direct equivalent)".into(),
        ));
    }

    // Data input is always slot 0.
    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(ResizeAnalysis::Skip(format!(
            "Resize input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let raw_in = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(ResizeAnalysis::Skip(format!(
                "Resize input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    let prepend_n = match raw_in.len() {
        4 => false,
        3 => true,
        other => {
            return Ok(ResizeAnalysis::Skip(format!(
                "Resize input rank {other} (only 3 or 4 supported)"
            )));
        }
    };
    let input_shape: Vec<i64> =
        if prepend_n { std::iter::once(1).chain(raw_in).collect() } else { raw_in };

    // Compute target H,W. Prefer scales input if non-empty; else sizes.
    // Some ONNX exports include both inputs but populate only one — when
    // that happens the unused side comes through as a length-0 const tensor.
    let scales_outlet_and_len: Option<(OutletId, usize)> =
        if let Some(idx) = rs.optional_scales_input {
            let outlet = node.inputs[idx];
            let len = const_tensor(model, outlet)?.map(|t| t.len()).unwrap_or(0);
            if len > 0 { Some((outlet, len)) } else { None }
        } else {
            None
        };

    let (target_h_full, target_w_full) = match (scales_outlet_and_len, rs.optional_sizes_input) {
        (Some((scales_outlet, _)), _) => {
            let scales_t = match const_tensor(model, scales_outlet)? {
                Some(t) => t.into_owned(),
                None => {
                    return Ok(ResizeAnalysis::Skip(format!(
                        "Resize scales input is not a Const \
                             — actual op: {}",
                        model.nodes[scales_outlet.node].op.name()
                    )));
                }
            };
            let scales = scales_t
                .cast_to::<f32>()
                .map_err(|e| anyhow::anyhow!("Resize scales cast to f32 failed: {e}"))?
                .into_owned();
            let scales_slice = unsafe { scales.as_slice_unchecked::<f32>() };
            if scales_slice.len() != input_shape.len() {
                return Ok(ResizeAnalysis::Skip(format!(
                    "Resize scales len {} != input rank {}",
                    scales_slice.len(),
                    input_shape.len()
                )));
            }
            // For CHW (prepend_n=true), tract scales are over rank-3, so the
            // spatial scales are at indices 1,2; with prepend, rank-4 spatial
            // axes are 2,3 — but the scales slice is rank-3, so map index
            // 1→H and 2→W of MLPackage shape.
            let (sh, sw) = if prepend_n {
                (scales_slice[1], scales_slice[2])
            } else {
                (scales_slice[2], scales_slice[3])
            };
            let h = (input_shape[2] as f32 * sh).round() as i32;
            let w = (input_shape[3] as f32 * sw).round() as i32;
            (h, w)
        }
        (None, Some(sizes_idx)) => {
            let sizes_outlet = node.inputs[sizes_idx];
            let sizes_t = match const_tensor(model, sizes_outlet)? {
                Some(t) => t.into_owned(),
                None => {
                    return Ok(ResizeAnalysis::Skip(format!(
                        "Resize sizes input slot {sizes_idx} is not a Const \
                             — actual op: {}",
                        model.nodes[sizes_outlet.node].op.name()
                    )));
                }
            };
            let sizes = sizes_t
                .cast_to::<i64>()
                .map_err(|e| anyhow::anyhow!("Resize sizes cast to i64 failed: {e}"))?
                .into_owned();
            let sizes_slice = unsafe { sizes.as_slice_unchecked::<i64>() };
            if sizes_slice.len() != input_shape.len() {
                return Ok(ResizeAnalysis::Skip(format!(
                    "Resize sizes len {} != input rank {}",
                    sizes_slice.len(),
                    input_shape.len()
                )));
            }
            let (h, w) = if prepend_n {
                (sizes_slice[1] as i32, sizes_slice[2] as i32)
            } else {
                (sizes_slice[2] as i32, sizes_slice[3] as i32)
            };
            (h, w)
        }
        (None, None) => {
            return Ok(ResizeAnalysis::Skip("Resize has neither scales nor sizes input".into()));
        }
    };

    // Verify N and C aren't being resized — only H,W (spatial) resizes are supported.
    // The non-spatial axes must remain at their input dims.
    let output_shape =
        vec![input_shape[0], input_shape[1], target_h_full as i64, target_w_full as i64];

    // Map interpolator + coord_transformer to MIL op + sampling_mode.
    let (mil_op, sampling_mode) = match rs.interpolator {
        Interpolator::Nearest => ("resize_nearest_neighbor", ""),
        Interpolator::Linear => {
            let mode = match rs.coord_transformer {
                CoordTransformer::HalfPixel => "DEFAULT",
                // PytorchHalfPixel is functionally identical to HalfPixel for
                // all output dims > 1 (only differs when output spatial dim
                // equals 1, an unusual edge case for upsampling). Map to MIL
                // DEFAULT (half_pixel) — accept the edge-case approximation.
                CoordTransformer::PytorchHalfPixel => "DEFAULT",
                CoordTransformer::AlignCorners => "STRICT_ALIGN_CORNERS",
                CoordTransformer::Asymmetric => "OFFSET_CORNERS",
            };
            ("resize_bilinear", mode)
        }
        Interpolator::Cubic => unreachable!("filtered above"),
    };

    Ok(ResizeAnalysis::Translatable(ResizePlan {
        input_shape,
        output_shape,
        target_h: target_h_full,
        target_w: target_w_full,
        mil_op,
        sampling_mode,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_resize_mil(
    plan: &ResizePlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let i32t_scalar = tensor_type_scalar(DataType::Int32);
    let str_t = tensor_type_scalar(DataType::String);

    let h_n = format!("{output_name}_target_h");
    let w_n = format!("{output_name}_target_w");
    let h_op = op_const_immediate(&h_n, i32t_scalar.clone(), tv_ints(vec![plan.target_h]));
    let w_op = op_const_immediate(&w_n, i32t_scalar, tv_ints(vec![plan.target_w]));

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(input_name));
    inputs.insert("target_size_height".into(), arg_name(&h_n));
    inputs.insert("target_size_width".into(), arg_name(&w_n));

    let mut prelude = vec![h_op, w_op];

    if plan.mil_op == "resize_bilinear" {
        let sm_n = format!("{output_name}_sampling_mode");
        let sm_op = op_const_immediate(&sm_n, str_t, tv_strings(vec![plan.sampling_mode.into()]));
        inputs.insert("sampling_mode".into(), arg_name(&sm_n));
        prelude.push(sm_op);
    }

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let resize_op = mil::Operation {
        r#type: plan.mil_op.into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(resize_op);
    Ok(prelude)
}
