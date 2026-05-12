//! tract `Conv` → MIL `conv` translator + MLPackage builder.
//!
//! Phase 1 scope (anything else returns `None` and the transform leaves the
//! node on CPU):
//! - F16 input, F16 const weight, F16 const bias
//! - NCHW data format, OIHW kernel format
//! - 2D convolution (kernel_shape rank 2)
//! - group == 1
//! - q_params is None (no quantization)
//! - PaddingSpec::Explicit (rank-2) or PaddingSpec::Valid

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};
use prost::Message;

use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec};
use tract_core::ops::nn::DataFormat;

use super::{const_tensor, shape_to_concrete_i64};

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::program::single_function_program;
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints, tv_strings};
use crate::mlpackage;
use crate::proto::core_ml::specification as spec;
use crate::proto::core_ml::specification::mil_spec as mil;

const COREML_INPUT_NAME: &str = "input";
const COREML_OUTPUT_NAME: &str = "output";

/// Outcome of analysing a candidate `Conv` node for CoreML translation.
#[allow(clippy::large_enum_variant)]
pub enum ConvAnalysis {
    /// Translatable; carries the materialisation plan.
    Translatable(ConvPlan),
    /// Not translatable; carries a short human-readable reason. Caller leaves
    /// the node on CPU. Reasons are useful for diagnostics on real models —
    /// they show coverage gaps as a list of "this op was skipped because ...".
    Skip(String),
}

/// Inspect `node` (which must be a `Conv`) and report whether the Coreml
/// translator can handle it.
pub fn analyse_conv(source: &TypedModel, node: &TypedNode) -> Result<ConvAnalysis> {
    let Some(conv) = node.op_as::<Conv>() else {
        return Ok(ConvAnalysis::Skip("not a Conv".into()));
    };

    if conv.q_params.is_some() {
        return Ok(ConvAnalysis::Skip("quantized Conv (q_params=Some)".into()));
    }
    if conv.kernel_fmt != KernelFormat::OIHW {
        return Ok(ConvAnalysis::Skip(format!("kernel_fmt={:?} (need OIHW)", conv.kernel_fmt)));
    }
    // group > 1 (depthwise / grouped) is supported — MIL conv accepts `groups` directly,
    // and OIHW weight layout is the same in tract and MIL: [O, I/groups, kH, kW].
    // Both NCHW (rank-4) and CHW (rank-3, no batch axis) are supported. CHW
    // is what tract uses after declutter strips the implicit-N=1 dim. For
    // CHW we synthesise a rank-4 MIL conv by prepending N=1; CoremlOp.eval
    // reshapes between the two ranks at runtime (cheap — same byte layout).
    let prepend_n = match conv.pool_spec.data_format {
        DataFormat::NCHW => false,
        DataFormat::CHW => true,
        other => {
            return Ok(ConvAnalysis::Skip(format!("data_format={other:?} (need NCHW or CHW)")));
        }
    };
    if conv.pool_spec.kernel_shape.len() != 2 {
        return Ok(ConvAnalysis::Skip(format!(
            "kernel rank {} (need 2; only 2D conv supported)",
            conv.pool_spec.kernel_shape.len()
        )));
    }
    if node.inputs.len() != 3 {
        return Ok(ConvAnalysis::Skip(format!(
            "node has {} inputs (need 3 = data+weight+bias)",
            node.inputs.len()
        )));
    }

    let input_fact = source.outlet_fact(node.inputs[0])?;
    if input_fact.datum_type != DatumType::F16 {
        return Ok(ConvAnalysis::Skip(format!(
            "input dtype {:?} (need F16)",
            input_fact.datum_type
        )));
    }
    let expected_rank = if prepend_n { 3 } else { 4 };
    let raw_input_shape = match shape_to_concrete_i64(&input_fact.shape) {
        Some(s) if s.len() == expected_rank => s,
        Some(s) => {
            return Ok(ConvAnalysis::Skip(format!(
                "input rank {} (expected {} for {:?})",
                s.len(),
                expected_rank,
                conv.pool_spec.data_format
            )));
        }
        None => {
            return Ok(ConvAnalysis::Skip(format!(
                "input shape has symbolic dims: {:?}",
                input_fact.shape
            )));
        }
    };
    let input_shape: Vec<i64> = if prepend_n {
        std::iter::once(1).chain(raw_input_shape).collect()
    } else {
        raw_input_shape
    };

    let output_fact = node.outputs[0].fact.clone();
    let raw_output_shape = match shape_to_concrete_i64(&output_fact.shape) {
        Some(s) if s.len() == expected_rank => s,
        Some(s) => {
            return Ok(ConvAnalysis::Skip(format!(
                "output rank {} (expected {} for {:?})",
                s.len(),
                expected_rank,
                conv.pool_spec.data_format
            )));
        }
        None => {
            return Ok(ConvAnalysis::Skip(format!(
                "output shape has symbolic dims: {:?}",
                output_fact.shape
            )));
        }
    };
    let output_shape: Vec<i64> = if prepend_n {
        std::iter::once(1).chain(raw_output_shape).collect()
    } else {
        raw_output_shape
    };

    let weight = match const_tensor(source, node.inputs[1])? {
        Some(t) => t,
        None => {
            let n = &source.nodes[node.inputs[1].node];
            return Ok(ConvAnalysis::Skip(format!(
                "weight (input slot 1) is not a Const node — actual op: {}",
                n.op.name()
            )));
        }
    };
    let bias = match const_tensor(source, node.inputs[2])? {
        Some(t) => t,
        None => {
            let n = &source.nodes[node.inputs[2].node];
            return Ok(ConvAnalysis::Skip(format!(
                "bias (input slot 2) is not a Const node — actual op: {}",
                n.op.name()
            )));
        }
    };
    if weight.datum_type() != DatumType::F16 {
        return Ok(ConvAnalysis::Skip(format!(
            "weight dtype {:?} (need F16)",
            weight.datum_type()
        )));
    }
    if bias.datum_type() != DatumType::F16 {
        return Ok(ConvAnalysis::Skip(format!("bias dtype {:?} (need F16)", bias.datum_type())));
    }
    if weight.rank() != 4 {
        return Ok(ConvAnalysis::Skip(format!("weight rank {} (need 4)", weight.rank())));
    }
    let weight_shape: Vec<i64> = weight.shape().iter().map(|&s| s as i64).collect();
    let out_channels = weight_shape[0] as usize;

    // Normalise bias to per-output-channel shape `[out_channels]`. tract may
    // hand us:
    //   - a per-channel `[O]` bias (after BN folding) — use as-is
    //   - a length-1 scalar (idiomatic "no bias provided", or a broadcast
    //     placeholder) — broadcast to `[O]`
    //   - anything else — skip with a useful reason.
    let bias_owned = bias.into_owned();
    let bias = if bias_owned.len() == out_channels {
        bias_owned
    } else if bias_owned.len() == 1 {
        let scalar = unsafe { bias_owned.as_slice_unchecked::<half::f16>()[0] };
        let broadcasted: Vec<half::f16> = vec![scalar; out_channels];
        Tensor::from_shape::<half::f16>(&[out_channels], &broadcasted)?
    } else {
        return Ok(ConvAnalysis::Skip(format!(
            "bias length {} doesn't match out_channels {} and isn't a length-1 broadcast",
            bias_owned.len(),
            out_channels
        )));
    };

    let pad = match &conv.pool_spec.padding {
        PaddingSpec::Explicit(before, after) => {
            if before.len() != 2 || after.len() != 2 {
                return Ok(ConvAnalysis::Skip(format!(
                    "Explicit padding rank ({}, {}) (need (2, 2))",
                    before.len(),
                    after.len()
                )));
            }
            vec![before[0] as i32, after[0] as i32, before[1] as i32, after[1] as i32]
        }
        PaddingSpec::Valid => vec![0, 0, 0, 0],
        other => {
            return Ok(ConvAnalysis::Skip(format!(
                "padding {:?} (only Explicit/Valid supported in Phase 1)",
                other
            )));
        }
    };

    let strides = match &conv.pool_spec.strides {
        Some(s) if s.len() == 2 => vec![s[0] as i32, s[1] as i32],
        None => vec![1, 1],
        Some(s) => {
            return Ok(ConvAnalysis::Skip(format!("strides rank {} (need 2)", s.len())));
        }
    };
    let dilations = match &conv.pool_spec.dilations {
        Some(d) if d.len() == 2 => vec![d[0] as i32, d[1] as i32],
        None => vec![1, 1],
        Some(d) => {
            return Ok(ConvAnalysis::Skip(format!("dilations rank {} (need 2)", d.len())));
        }
    };

    Ok(ConvAnalysis::Translatable(ConvPlan {
        input_shape,
        output_shape,
        weight: weight.into_owned(),
        weight_shape,
        bias,
        pad,
        strides,
        dilations,
        groups: conv.group as i32,
        output_fact,
    }))
}

/// All the data needed to emit + load a single-Conv MLPackage.
pub struct ConvPlan {
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
    pub weight: Tensor,
    pub weight_shape: Vec<i64>,
    pub bias: Tensor,
    pub pad: Vec<i32>,       // [pad_top, pad_bottom, pad_left, pad_right]
    pub strides: Vec<i32>,   // [stride_h, stride_w]
    pub dilations: Vec<i32>, // [dilation_h, dilation_w]
    pub groups: i32,         // 1 = standard, in_channels = depthwise
    pub output_fact: TypedFact,
}

/// Emit the MIL ops for a single tract `Conv` translation, into the shared
/// `BlobBuilder`. Returns the list of MIL ops in topological order. The
/// caller wires `input_name` (the data input) and `output_name` (the conv
/// output) into the surrounding Program.
///
/// Used by both:
/// - `build_conv_mlpackage` — single-op wrapper for tests / Phase 1 backward compat
/// - `crate::fusion::build_subgraph_mlpackage` — multi-op subgraph builder
///
/// The const-op names are prefixed with `output_name` to guarantee uniqueness
/// across multiple Conv ops in the same subgraph.
pub fn emit_conv_mil(
    plan: &ConvPlan,
    blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let out_channels = plan.weight_shape[0];
    if plan.bias.len() != out_channels as usize {
        bail!("conv bias length {} does not match out_channels {}", plan.bias.len(), out_channels);
    }

    let weight_off = blob.add(BlobDataType::Float16, plan.weight.as_bytes());
    let bias_off = blob.add(BlobDataType::Float16, plan.bias.as_bytes());

    let weight_ty = tensor_type(DataType::Float16, &plan.weight_shape);
    let bias_ty = tensor_type(DataType::Float16, &[out_channels]);
    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let str_scalar = tensor_type_scalar(DataType::String);
    let i32t_4 = tensor_type(DataType::Int32, &[4]);
    let i32t_2 = tensor_type(DataType::Int32, &[2]);
    let i32t_scalar = tensor_type_scalar(DataType::Int32);

    // Prefix MIL constant names with the output name so different convs in the
    // same subgraph don't collide.
    let p = output_name;
    let pad_type_n = format!("{p}_pad_type");
    let pad_n = format!("{p}_pad");
    let strides_n = format!("{p}_strides");
    let dilations_n = format!("{p}_dilations");
    let groups_n = format!("{p}_groups");
    let weight_n = format!("{p}_weight");
    let bias_n = format!("{p}_bias");

    let pad_type_op =
        op_const_immediate(&pad_type_n, str_scalar, tv_strings(vec!["custom".into()]));
    let pad_op = op_const_immediate(&pad_n, i32t_4, tv_ints(plan.pad.clone()));
    let strides_op = op_const_immediate(&strides_n, i32t_2.clone(), tv_ints(plan.strides.clone()));
    let dilations_op =
        op_const_immediate(&dilations_n, i32t_2.clone(), tv_ints(plan.dilations.clone()));
    let groups_op = op_const_immediate(&groups_n, i32t_scalar, tv_ints(vec![plan.groups]));
    let weight_op = op_const_blob(&weight_n, weight_ty, mlpackage::WEIGHT_BLOB_PATH, weight_off);
    let bias_op = op_const_blob(&bias_n, bias_ty, mlpackage::WEIGHT_BLOB_PATH, bias_off);

    let mut prelude =
        vec![pad_type_op, pad_op, strides_op, dilations_op, groups_op, weight_op, bias_op];

    // Defensive input-shape normalization. Several upstream MIL ops emit
    // their output at tract's own rank — for CHW data that's rank-3
    // [C, H, W]. MIL `conv` requires rank-4 NCHW; consuming a rank-3 tensor
    // aborts MLPackage compilation with "Variadic dimension at [2, -1] of
    // tensor parameter x[0] have unexpected length 1; expected 2".
    //
    // Originally surfaced by YOLOv8n: tract's declutter rewrites a 2× nearest
    // Resize into `Reshape→AddAxis→AddAxis→MultiBroadcastTo→Reshape`; the
    // trailing Reshape in that chain emits rank-3 [C, H, W] (tract's CHW
    // truth) and feeds the next stride-2 3×3 Conv. Fix: emit a leading
    // `mb.reshape(x, shape=plan.input_shape)` to materialise a rank-4 input
    // unconditionally. When the upstream is already rank-4 NCHW with the
    // matching shape, MIL `reshape` is metadata-only — no compute cost. When
    // it's CHW rank-3 (or any other rank with the same total element count),
    // the leading N=1 gets prepended.
    let conv_in_name = format!("{p}_x_rank4");
    let conv_in_shape_n = format!("{p}_x_shape");
    let conv_in_shape_op = op_const_immediate(
        &conv_in_shape_n,
        tensor_type(DataType::Int32, &[plan.input_shape.len() as i64]),
        tv_ints(plan.input_shape.iter().map(|&v| v as i32).collect()),
    );
    let conv_in_ty = tensor_type(DataType::Float16, &plan.input_shape);
    let mut reshape_inputs: HashMap<String, mil::Argument> = HashMap::new();
    reshape_inputs.insert("x".into(), arg_name(input_name));
    reshape_inputs.insert("shape".into(), arg_name(&conv_in_shape_n));
    let conv_in_reshape_op = mil::Operation {
        r#type: "reshape".into(),
        inputs: reshape_inputs,
        outputs: vec![mil::NamedValueType { name: conv_in_name.clone(), r#type: Some(conv_in_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(conv_in_shape_op);
    prelude.push(conv_in_reshape_op);

    let mut conv_inputs: HashMap<String, mil::Argument> = HashMap::new();
    conv_inputs.insert("x".into(), arg_name(&conv_in_name));
    conv_inputs.insert("weight".into(), arg_name(&weight_n));
    conv_inputs.insert("bias".into(), arg_name(&bias_n));
    conv_inputs.insert("strides".into(), arg_name(&strides_n));
    conv_inputs.insert("pad_type".into(), arg_name(&pad_type_n));
    conv_inputs.insert("pad".into(), arg_name(&pad_n));
    conv_inputs.insert("dilations".into(), arg_name(&dilations_n));
    conv_inputs.insert("groups".into(), arg_name(&groups_n));

    let conv_op = mil::Operation {
        r#type: "conv".into(),
        inputs: conv_inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(conv_op);

    Ok(prelude)
}

/// Build a single-Conv MLPackage at `out` from a [`ConvPlan`]. Thin wrapper
/// around [`emit_conv_mil`] used by the Phase 1 single-op transform path
/// and by tests.
pub fn build_conv_mlpackage(out: &Path, plan: &ConvPlan) -> Result<()> {
    let mut blob = BlobBuilder::new();
    let mil_ops = emit_conv_mil(plan, &mut blob, COREML_INPUT_NAME, COREML_OUTPUT_NAME)?;
    let blob_bytes = blob.finish();

    let in_ty = tensor_type(DataType::Float16, &plan.input_shape);
    let program = single_function_program(
        vec![mil::NamedValueType { name: COREML_INPUT_NAME.into(), r#type: Some(in_ty) }],
        vec![COREML_OUTPUT_NAME.into()],
        mil_ops,
    );

    let model = build_conv_model(program, &plan.input_shape, &plan.output_shape);
    let model_bytes = model.encode_to_vec();
    mlpackage::write(out, &model_bytes, &blob_bytes)?;
    Ok(())
}

fn build_conv_model(program: mil::Program, in_shape: &[i64], out_shape: &[i64]) -> spec::Model {
    let in_feat = spec::FeatureDescription {
        name: COREML_INPUT_NAME.into(),
        short_description: String::new(),
        r#type: Some(spec::FeatureType {
            is_optional: false,
            r#type: Some(spec::feature_type::Type::MultiArrayType(spec::ArrayFeatureType {
                shape: in_shape.to_vec(),
                data_type: spec::array_feature_type::ArrayDataType::Float16 as i32,
                shape_flexibility: None,
                default_optional_value: None,
            })),
        }),
    };
    let out_feat = spec::FeatureDescription {
        name: COREML_OUTPUT_NAME.into(),
        short_description: String::new(),
        r#type: Some(spec::FeatureType {
            is_optional: false,
            r#type: Some(spec::feature_type::Type::MultiArrayType(spec::ArrayFeatureType {
                shape: out_shape.to_vec(),
                data_type: spec::array_feature_type::ArrayDataType::Float16 as i32,
                shape_flexibility: None,
                default_optional_value: None,
            })),
        }),
    };
    let description = spec::ModelDescription {
        input: vec![in_feat],
        output: vec![out_feat],
        functions: vec![],
        default_function_name: String::new(),
        predicted_feature_name: String::new(),
        predicted_probabilities_name: String::new(),
        training_input: vec![],
        metadata: None,
        state: vec![],
    };
    spec::Model {
        specification_version: 8,
        description: Some(description),
        is_updatable: false,
        r#type: Some(spec::model::Type::MlProgram(program)),
    }
}

/// Names for `CoremlOp` I/O slots used by [`build_conv_mlpackage`].
pub fn input_name() -> &'static str {
    COREML_INPUT_NAME
}
pub fn output_name() -> &'static str {
    COREML_OUTPUT_NAME
}

/// Return a unique temp-dir path for an MLPackage. Caller takes ownership of
/// the directory; macOS will clean it on next reboot.
pub fn next_temp_pkg_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tract-coreml-conv-{}-{n}.mlpackage", std::process::id()))
}

// -------- helpers --------

// `const_tensor` and `shape_to_concrete_i64` moved to `crate::ops::mod`
// (shared with binop and future op translators).
