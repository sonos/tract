//! Differential test: build the SAME single ONNX GRU two ways — via the
//! Scan-based decomposition (TRACT_DISABLE_OPT_GRU=1) and via the fused
//! OptGru op (default) — and bit-compare the optimized-model outputs.
//!
//! Isolates OptGru wiring correctness from full-model (df_dec) complexity.

use tract_onnx::prelude::*;

// Re-export the internal types we need by going through tract-onnx's path.
// CommonRec / GRU / WireBody live in `tract_onnx::ops::rec` (crate-internal),
// so instead of touching internals we build a real ONNX GRU NodeProto and
// run it through the standard import path. That also exercises the actual
// production code path end to end.

use tract_onnx::pb::*;

fn make_tensor_proto(name: &str, shape: &[i64], data: Vec<f32>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: tensor_proto::DataType::Float as i32,
        float_data: data,
        ..Default::default()
    }
}

/// Build a value-info. Dims with value `-1` are emitted as the symbolic
/// dim_param "S" (mirroring df_dec's symbolic sequence length).
fn value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|d| tensor_shape_proto::Dimension {
                            value: Some(if *d == -1 {
                                tensor_shape_proto::dimension::Value::DimParam("S".to_string())
                            } else {
                                tensor_shape_proto::dimension::Value::DimValue(*d)
                            }),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
        }),
        ..Default::default()
    }
}

fn build_gru_model(
    hidden: usize,
    input: usize,
    seq: usize,
    lbr: i64,
    symbolic_seq: bool,
) -> ModelProto {
    let h = hidden as i64;
    let i = input as i64;
    let s = if symbolic_seq { -1 } else { seq as i64 };

    // Deterministic weights.
    let w_data: Vec<f32> =
        (0..3 * hidden * input).map(|n| ((n as f32 * 0.083) - 1.0) * 0.2).collect();
    let r_data: Vec<f32> =
        (0..3 * hidden * hidden).map(|n| ((n as f32 * 0.071) - 1.0) * 0.15).collect();
    let b_data: Vec<f32> = (0..6 * hidden).map(|n| (n as f32 * 0.03) - 0.4).collect();

    let gru_node = NodeProto {
        op_type: "GRU".to_string(),
        input: vec![
            "X".to_string(),
            "W".to_string(),
            "R".to_string(),
            "B".to_string(),
            "".to_string(), // sequence_lens (absent)
            "initial_h".to_string(),
        ],
        output: vec!["Y".to_string(), "Y_h".to_string()],
        name: "gru".to_string(),
        attribute: vec![
            AttributeProto {
                name: "hidden_size".to_string(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: h,
                ..Default::default()
            },
            AttributeProto {
                name: "linear_before_reset".to_string(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: lbr,
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let graph = GraphProto {
        node: vec![gru_node],
        name: "g".to_string(),
        initializer: vec![
            make_tensor_proto("W", &[1, 3 * h, i], w_data),
            make_tensor_proto("R", &[1, 3 * h, h], r_data),
            make_tensor_proto("B", &[1, 6 * h], b_data),
            make_tensor_proto("initial_h", &[1, 1, h], vec![0.0; hidden]),
        ],
        input: vec![value_info("X", &[s, 1, i])],
        output: vec![value_info("Y", &[s, 1, 1, h]), value_info("Y_h", &[1, 1, h])],
        ..Default::default()
    };

    ModelProto {
        ir_version: 7,
        opset_import: vec![OperatorSetIdProto { domain: "".to_string(), version: 14 }],
        graph: Some(graph),
        ..Default::default()
    }
}

fn run_model(proto: &ModelProto, x: &Tensor, sub_s: Option<usize>) -> TVec<TValue> {
    let model = tract_onnx::onnx().model_for_proto_model(proto).unwrap();
    let mut typed = model.into_typed().unwrap();
    if let Some(s) = sub_s {
        // Substitute the symbolic "S" dim to a concrete value, mirroring the
        // TRACT_BENCH_SYMBOLS=S=100 path used for df_dec.
        let sym = typed.symbols.sym("S");
        let mut subs = std::collections::HashMap::new();
        subs.insert(sym, (s as i64).to_dim());
        typed = typed.substitute_symbols(&subs).unwrap();
    }
    let typed = typed.into_optimized().unwrap();
    let runnable = typed.into_runnable().unwrap();
    runnable.run(tvec![x.clone().into_tvalue()]).unwrap()
}

fn compare_paths(hidden: usize, input: usize, seq: usize, lbr: i64, symbolic_seq: bool) {
    let proto = build_gru_model(hidden, input, seq, lbr, symbolic_seq);
    let sub_s = if symbolic_seq { Some(seq) } else { None };

    let x_data: Vec<f32> = (0..seq * input).map(|n| ((n as f32 * 0.137) - 1.0) * 0.3).collect();
    let x = Tensor::from_shape(&[seq, 1, input], &x_data).unwrap();

    // Scan path (OptGru opt-in flag unset)
    unsafe { std::env::remove_var("TRACT_ENABLE_OPT_GRU") };
    let scan_out = run_model(&proto, &x, sub_s);

    // OptGru path (opt-in)
    unsafe { std::env::set_var("TRACT_ENABLE_OPT_GRU", "1") };
    let opt_out = run_model(&proto, &x, sub_s);
    unsafe { std::env::remove_var("TRACT_ENABLE_OPT_GRU") };

    assert_eq!(
        scan_out.len(),
        opt_out.len(),
        "output count mismatch (lbr={lbr} sym={symbolic_seq})"
    );
    for (oi, (s, o)) in scan_out.iter().zip(opt_out.iter()).enumerate() {
        let sv = s.to_plain_array_view::<f32>().unwrap();
        let ov = o.to_plain_array_view::<f32>().unwrap();
        assert_eq!(
            sv.shape(),
            ov.shape(),
            "output[{oi}] shape mismatch (lbr={lbr} sym={symbolic_seq}): scan={:?} opt={:?}",
            sv.shape(),
            ov.shape()
        );
        let max_abs: f32 = sv.iter().zip(ov.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(
            max_abs < 1e-5,
            "output[{oi}] value mismatch (lbr={lbr} sym={symbolic_seq}): max_abs={max_abs}\nscan={sv:?}\nopt={ov:?}"
        );
    }
}

#[test]
fn opt_gru_matches_scan_single_gru() {
    for lbr in [1i64, 0] {
        compare_paths(8, 5, 7, lbr, false);
    }
}

#[test]
fn opt_gru_matches_scan_symbolic_seq() {
    // df_dec uses a symbolic sequence dim S. This exercises the optimizer with
    // OptGru's symbolic output_facts (the condition that black-box df_dec
    // testing implicated).
    for lbr in [1i64, 0] {
        compare_paths(8, 5, 7, lbr, true);
    }
}

#[test]
fn opt_gru_matches_scan_dfn3_dims() {
    // Match df_dec's actual GRU dimensions: hidden=256, input=256, seq=100.
    compare_paths(256, 256, 100, 1, true);
}

/// Two stacked GRUs (GRU1.Y → Squeeze → GRU2.X), mirroring df_dec's
/// `/df_gru/gru/GRU` + `/df_gru/gru/GRU_1` structure. Both GRUs lower to
/// OptGru; compare against both lowering to Scan.
fn build_two_gru_model(hidden: usize, input: usize, seq: usize) -> ModelProto {
    let h = hidden as i64;
    let i = input as i64;
    let s = -1i64; // symbolic S

    let w1: Vec<f32> = (0..3 * hidden * input).map(|n| ((n as f32 * 0.083) - 1.0) * 0.2).collect();
    let r1: Vec<f32> =
        (0..3 * hidden * hidden).map(|n| ((n as f32 * 0.071) - 1.0) * 0.15).collect();
    let b1: Vec<f32> = (0..6 * hidden).map(|n| (n as f32 * 0.03) - 0.4).collect();
    // GRU2 takes input=hidden.
    let w2: Vec<f32> =
        (0..3 * hidden * hidden).map(|n| ((n as f32 * 0.061) - 1.0) * 0.18).collect();
    let r2: Vec<f32> =
        (0..3 * hidden * hidden).map(|n| ((n as f32 * 0.053) - 1.0) * 0.12).collect();
    let b2: Vec<f32> = (0..6 * hidden).map(|n| (n as f32 * 0.025) - 0.3).collect();

    let attrs = |lbr: i64| {
        vec![
            AttributeProto {
                name: "hidden_size".to_string(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: h,
                ..Default::default()
            },
            AttributeProto {
                name: "linear_before_reset".to_string(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: lbr,
                ..Default::default()
            },
        ]
    };

    let gru1 = NodeProto {
        op_type: "GRU".to_string(),
        input: vec!["X".into(), "W1".into(), "R1".into(), "B1".into(), "".into(), "ih1".into()],
        output: vec!["Y1".into(), "".into()],
        name: "gru1".to_string(),
        attribute: attrs(1),
        ..Default::default()
    };
    // Squeeze Y1 [seq,1,1,hid] axis 1 → [seq,1,hid] for GRU2's X.
    let squeeze = NodeProto {
        op_type: "Squeeze".to_string(),
        input: vec!["Y1".into(), "sq_axes".into()],
        output: vec!["X2".into()],
        name: "squeeze".to_string(),
        ..Default::default()
    };
    let gru2 = NodeProto {
        op_type: "GRU".to_string(),
        input: vec!["X2".into(), "W2".into(), "R2".into(), "B2".into(), "".into(), "ih2".into()],
        output: vec!["Y2".into(), "Y_h2".into()],
        name: "gru2".to_string(),
        attribute: attrs(1),
        ..Default::default()
    };

    let graph = GraphProto {
        node: vec![gru1, squeeze, gru2],
        name: "g2".to_string(),
        initializer: vec![
            make_tensor_proto("W1", &[1, 3 * h, i], w1),
            make_tensor_proto("R1", &[1, 3 * h, h], r1),
            make_tensor_proto("B1", &[1, 6 * h], b1),
            make_tensor_proto("ih1", &[1, 1, h], vec![0.0; hidden]),
            make_tensor_proto("W2", &[1, 3 * h, h], w2),
            make_tensor_proto("R2", &[1, 3 * h, h], r2),
            make_tensor_proto("B2", &[1, 6 * h], b2),
            make_tensor_proto("ih2", &[1, 1, h], vec![0.0; hidden]),
            TensorProto {
                name: "sq_axes".to_string(),
                dims: vec![1],
                data_type: tensor_proto::DataType::Int64 as i32,
                int64_data: vec![1],
                ..Default::default()
            },
        ],
        input: vec![value_info("X", &[s, 1, i])],
        output: vec![value_info("Y2", &[s, 1, 1, h]), value_info("Y_h2", &[1, 1, h])],
        ..Default::default()
    };

    ModelProto {
        ir_version: 7,
        opset_import: vec![OperatorSetIdProto { domain: "".to_string(), version: 14 }],
        graph: Some(graph),
        ..Default::default()
    }
}

#[test]
fn opt_gru_matches_scan_two_stacked() {
    let hidden = 8;
    let input = 5;
    let seq = 7;
    let proto = build_two_gru_model(hidden, input, seq);
    let x_data: Vec<f32> = (0..seq * input).map(|n| ((n as f32 * 0.137) - 1.0) * 0.3).collect();
    let x = Tensor::from_shape(&[seq, 1, input], &x_data).unwrap();

    unsafe { std::env::remove_var("TRACT_ENABLE_OPT_GRU") };
    let scan_out = run_model(&proto, &x, Some(seq));
    unsafe { std::env::set_var("TRACT_ENABLE_OPT_GRU", "1") };
    let opt_out = run_model(&proto, &x, Some(seq));
    unsafe { std::env::remove_var("TRACT_ENABLE_OPT_GRU") };

    for (oi, (s, o)) in scan_out.iter().zip(opt_out.iter()).enumerate() {
        let sv = s.to_plain_array_view::<f32>().unwrap();
        let ov = o.to_plain_array_view::<f32>().unwrap();
        assert_eq!(
            sv.shape(),
            ov.shape(),
            "out[{oi}] shape: scan={:?} opt={:?}",
            sv.shape(),
            ov.shape()
        );
        let max_abs: f32 = sv.iter().zip(ov.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(max_abs < 1e-5, "out[{oi}] mismatch: max_abs={max_abs}");
    }
}
