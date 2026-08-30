use crate::model::ParsingContext;
use crate::ops::nn::rms_norm::rms_normalization;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn simplified_layer_norm(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    // `SimplifiedLayerNormalization` is RMS normalization: it scales by the root
    // mean square and does *not* subtract the mean. Lowering it to `LayerNorm`,
    // which centres the input, produced wrong activations for every non-zero-mean
    // input (and silently dropped the optional bias input). Delegate to the RMS
    // normalization parser instead (#2646).
    rms_normalization(ctx, node)
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

    // `SimplifiedLayerNormalization` is RMS normalization, which does not subtract
    // the mean. With a non-zero-mean input the previous `LayerNorm` lowering gave a
    // completely different (centred) result. Regression test for #2646.
    #[test]
    fn simplified_layer_norm_is_rms_norm() -> TractResult<()> {
        let node = NodeProto {
            name: "sln".to_string(),
            op_type: "SimplifiedLayerNormalization".to_string(),
            input: vec!["x".to_string(), "scale".to_string()],
            output: vec!["y".to_string()],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "g".to_string(),
            input: vec![float_value_info("x", &[1, 4]), float_value_info("scale", &[4])],
            output: vec![float_value_info("y", &[1, 4])],
            ..GraphProto::default()
        };
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 18 }],
            graph: Some(graph),
            ..ModelProto::default()
        };
        let mut buf = vec![];
        model.encode(&mut buf).unwrap();
        let runnable =
            crate::onnx().model_for_read(&mut &*buf)?.into_optimized()?.into_runnable()?;

        // x has a non-zero mean; scale is all ones.
        let x = tensor2(&[[1f32, 2., 3., 4.]]);
        let scale = tensor1(&[1f32, 1., 1., 1.]);
        let out = runnable.run(tvec!(x.into_tvalue(), scale.into_tvalue()))?;

        // RMS norm: rms = sqrt(mean(x^2) + 1e-5) = sqrt(7.5 + 1e-5) ~= 2.7386
        //   y = x / rms ~= [0.3651, 0.7303, 1.0954, 1.4606]
        // (LayerNorm would centre first and yield [-1.342, -0.447, 0.447, 1.342].)
        let expected = tensor2(&[[0.365_15f32, 0.730_30, 1.095_45, 1.460_60]]);
        out[0].close_enough(&expected, Approximation::Approximate)?;
        Ok(())
    }
}
