//! End-to-end smoke test for `CoremlOp`.
//!
//! Builds a minimal Conv2D MLPackage from scratch (using the `mil` and
//! `mlpackage` modules), loads it via `CoremlContext`, wraps it in a
//! `CoremlOp`, and exercises the whole stack with two test cases.
//!
//! Test 1: `zero_input_yields_zero_output` is a sanity check that the load,
//! predict, and read paths are wired end-to-end (Conv with any weights on a
//! zero input must produce zero output).
//!
//! Test 2: `ones_input_ones_weight_known_pattern` is a concrete numerical
//! check. With input and weights both all-`1.0`, and 3×3 conv with padding 1,
//! the output value at each spatial position equals the count of in-range
//! receptive-field positions × 4 input channels, giving a fixed
//! 16/24/36 corner/edge/interior pattern per output channel.
//!
//! Together these prove the whole stack works *without* any `CoremlTransform`
//! or `CoremlRuntime` — those are the next milestones, layered on top.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use half::f16;
use objc2_core_ml::MLComputeUnits;
use prost::Message;

use tract_core::internal::*;

use tract_coreml::mil::blob::{BlobBuilder, BlobDataType};
use tract_coreml::mil::op::{arg_name, op_const_blob, op_const_immediate};
use tract_coreml::mil::program::single_function_program;
use tract_coreml::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints, tv_strings};
use tract_coreml::mlpackage;
use tract_coreml::proto::core_ml::specification as spec;
use tract_coreml::proto::core_ml::specification::mil_spec as mil;
use tract_coreml::{CoremlContext, CoremlOp};

const IN_SHAPE: [i64; 4] = [1, 4, 8, 8];
const WEIGHT_SHAPE: [i64; 4] = [8, 4, 3, 3];
const OUT_SHAPE: [i64; 4] = [1, 8, 8, 8];
const N_WEIGHTS: usize = 8 * 4 * 3 * 3; // 288 FP16 elements
const IN_NAME: &str = "input";
const OUT_NAME: &str = "output";

/// Build a minimal `[1,4,8,8] f16 -> Conv(3x3, pad=1) -> [1,8,8,8] f16`
/// MLPackage with the given FP16 weight bytes and write it to `out`.
fn build_conv_mlpackage(out: &Path, weight_bytes: &[u8]) -> Result<()> {
    assert_eq!(
        weight_bytes.len(),
        N_WEIGHTS * 2,
        "weight_bytes length mismatch for FP16 [8,4,3,3]"
    );

    // 1. Weight blob.
    let mut blob = BlobBuilder::new();
    let weight_offset = blob.add(BlobDataType::Float16, weight_bytes);
    let blob_bytes = blob.finish();

    // 2. MIL Program (mirrors the Phase 1 spike's structure).
    let in_ty = tensor_type(DataType::Float16, &IN_SHAPE);
    let weight_ty = tensor_type(DataType::Float16, &WEIGHT_SHAPE);
    let out_ty = tensor_type(DataType::Float16, &OUT_SHAPE);
    let str_scalar = tensor_type_scalar(DataType::String);
    let i32t_4 = tensor_type(DataType::Int32, &[4]);
    let i32t_2 = tensor_type(DataType::Int32, &[2]);
    let i32t_scalar = tensor_type_scalar(DataType::Int32);

    let pad_type_op = op_const_immediate("pad_type", str_scalar, tv_strings(vec!["custom".into()]));
    let pad_op = op_const_immediate("pad", i32t_4, tv_ints(vec![1, 1, 1, 1]));
    let strides_op = op_const_immediate("strides", i32t_2.clone(), tv_ints(vec![1, 1]));
    let dilations_op = op_const_immediate("dilations", i32t_2, tv_ints(vec![1, 1]));
    let groups_op = op_const_immediate("groups", i32t_scalar, tv_ints(vec![1]));
    let weight_op = op_const_blob("weight", weight_ty, mlpackage::WEIGHT_BLOB_PATH, weight_offset);

    let mut conv_inputs: HashMap<String, mil::Argument> = HashMap::new();
    conv_inputs.insert("x".into(), arg_name(IN_NAME));
    conv_inputs.insert("weight".into(), arg_name("weight"));
    conv_inputs.insert("strides".into(), arg_name("strides"));
    conv_inputs.insert("pad_type".into(), arg_name("pad_type"));
    conv_inputs.insert("pad".into(), arg_name("pad"));
    conv_inputs.insert("dilations".into(), arg_name("dilations"));
    conv_inputs.insert("groups".into(), arg_name("groups"));

    let conv_op = mil::Operation {
        r#type: "conv".into(),
        inputs: conv_inputs,
        outputs: vec![mil::NamedValueType { name: OUT_NAME.into(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    let program = single_function_program(
        vec![mil::NamedValueType { name: IN_NAME.into(), r#type: Some(in_ty) }],
        vec![OUT_NAME.into()],
        vec![pad_type_op, pad_op, strides_op, dilations_op, groups_op, weight_op, conv_op],
    );

    // 3. Wrap in a Model.
    let in_feat = spec::FeatureDescription {
        name: IN_NAME.into(),
        short_description: String::new(),
        r#type: Some(spec::FeatureType {
            is_optional: false,
            r#type: Some(spec::feature_type::Type::MultiArrayType(spec::ArrayFeatureType {
                shape: IN_SHAPE.to_vec(),
                data_type: spec::array_feature_type::ArrayDataType::Float16 as i32,
                shape_flexibility: None,
                default_optional_value: None,
            })),
        }),
    };
    let out_feat = spec::FeatureDescription {
        name: OUT_NAME.into(),
        short_description: String::new(),
        r#type: Some(spec::FeatureType {
            is_optional: false,
            r#type: Some(spec::feature_type::Type::MultiArrayType(spec::ArrayFeatureType {
                shape: OUT_SHAPE.to_vec(),
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
    let model = spec::Model {
        specification_version: 8, // iOS 18 / macOS 15 / Core ML 8
        description: Some(description),
        is_updatable: false,
        r#type: Some(spec::model::Type::MlProgram(program)),
    };

    let model_bytes = model.encode_to_vec();
    mlpackage::write(out, &model_bytes, &blob_bytes)?;
    Ok(())
}

fn make_op(pkg_path: &Path) -> CoremlOp {
    let ctx = Arc::new(
        CoremlContext::load_mlpackage(pkg_path, MLComputeUnits::All)
            .expect("CoremlContext::load_mlpackage"),
    );
    let in_usize: Vec<usize> = IN_SHAPE.iter().map(|&s| s as usize).collect();
    let out_usize: Vec<usize> = OUT_SHAPE.iter().map(|&s| s as usize).collect();
    CoremlOp {
        context: ctx,
        input_names: vec![IN_NAME.into()],
        output_names: vec![OUT_NAME.into()],
        output_facts: tvec![f16::fact(out_usize.clone())],
        coreml_input_shapes: vec![in_usize],
        coreml_input_dtypes: vec![f16::datum_type()],
        coreml_output_shapes: vec![out_usize],
    }
}

fn fp16_bytes_filled_with(value: f32, n: usize) -> Vec<u8> {
    let v = f16::from_f32(value);
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn zero_input_yields_zero_output() -> Result<()> {
    let pkg_path = std::env::temp_dir().join("tract-coreml-smoke-zero/conv.mlpackage");
    let weights = fp16_bytes_filled_with(1.0, N_WEIGHTS);
    build_conv_mlpackage(&pkg_path, &weights)?;

    let op = make_op(&pkg_path);
    let inp = Tensor::zero_dt(DatumType::F16, &[1, 4, 8, 8])?;
    let outputs = op.eval(tvec![inp.into()])?;
    assert_eq!(outputs.len(), 1);
    let out = &outputs[0];
    assert_eq!(out.shape(), &[1, 8, 8, 8]);
    assert_eq!(out.datum_type(), DatumType::F16);

    let bytes = out.as_bytes();
    assert!(
        bytes.iter().all(|&b| b == 0),
        "output should be all zeros for zero input; first non-zero at index {:?}",
        bytes.iter().position(|&b| b != 0)
    );
    Ok(())
}

#[test]
fn ones_input_ones_weight_known_pattern() -> Result<()> {
    let pkg_path = std::env::temp_dir().join("tract-coreml-smoke-ones/conv.mlpackage");
    let weights = fp16_bytes_filled_with(1.0, N_WEIGHTS);
    build_conv_mlpackage(&pkg_path, &weights)?;

    let op = make_op(&pkg_path);
    let inp_data: Vec<f16> = vec![f16::from_f32(1.0); IN_SHAPE.iter().product::<i64>() as usize];
    let inp = Tensor::from_shape::<f16>(&[1, 4, 8, 8], &inp_data)?;
    let outputs = op.eval(tvec![inp.into()])?;
    let out = &outputs[0];
    assert_eq!(out.shape(), &[1, 8, 8, 8]);

    // Reading f16 slice — Tensor::as_slice<f16> needs the right Datum type.
    let out_slice = unsafe { out.as_slice_unchecked::<f16>() };
    assert_eq!(out_slice.len(), 8 * 8 * 8);

    // For input + weight both all-1, output[oc, h, w] = 4 * count_of_valid_kernel_positions.
    // With pad=1, 3×3 kernel, 8×8 spatial: corners get 4 valid positions, edges 6,
    // interior 9. Multiplied by 4 in_channels: 16 / 24 / 36.
    for oc in 0..8 {
        for h in 0..8 {
            for w in 0..8 {
                let idx = (oc * 8 + h) * 8 + w;
                let got = out_slice[idx].to_f32();
                let expected = expected_value(h, w);
                let diff = (got - expected).abs();
                assert!(
                    diff < 0.5,
                    "oc={oc} h={h} w={w}: got {got}, expected {expected} (FP16 abs diff {diff})"
                );
            }
        }
    }
    Ok(())
}

fn expected_value(h: usize, w: usize) -> f32 {
    let h_edge = h == 0 || h == 7;
    let w_edge = w == 0 || w == 7;
    if h_edge && w_edge {
        16.0 // corner
    } else if h_edge || w_edge {
        24.0 // edge
    } else {
        36.0 // interior
    }
}
