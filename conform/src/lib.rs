#![allow(non_snake_case)]

#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate ndarray;
#[macro_use]
extern crate proptest;
extern crate protobuf;
extern crate simplelog;
extern crate tensorflow;
extern crate tfdeploy;

error_chain! {
    foreign_links {
        Io(::std::io::Error);
        NdarrayShape(::ndarray::ShapeError);
        Protobuf(::protobuf::ProtobufError);
        StrUtf8(::std::str::Utf8Error);
    }
    errors {
        TFString {}
    }
}

impl ::std::convert::From<::tensorflow::Status> for Error {
    fn from(tfs: ::tensorflow::Status) -> Error {
        format!("Tensorflow error: {:?}", tfs).into()
    }
}

pub mod tf;

pub use protobuf::Message;

use tfdeploy::analyser::TensorFact;
use tfdeploy::tfpb;
use tfdeploy::Tensor as TfdTensor;
use tfpb::tensor_shape::TensorShapeProto;
use tfpb::types::DataType;

pub fn placeholder<Shape: Into<Option<TensorShapeProto>>>(
    name: &str,
    t: DataType,
    shape: Shape,
) -> tfpb::node_def::NodeDef {
    let mut node = tfpb::node().name(name).op("Placeholder").attr("dtype", t);
    if let Some(shape) = shape.into() {
        node = node.attr("shape", shape)
    }
    node
}

pub fn tensor_shape(dims: &[usize]) -> TensorShapeProto {
    use tfpb::tensor_shape::*;
    let mut shape = TensorShapeProto::new();
    shape.set_dim(
        dims.iter()
            .map(|&d| {
                let mut dim = TensorShapeProto_Dim::new();
                dim.set_size(d as i64);
                dim
            })
            .collect(),
    );
    shape
}

pub fn placeholder_f32(name: &str) -> tfpb::node_def::NodeDef {
    placeholder(name, DataType::DT_FLOAT, None)
}

pub fn placeholder_i32(name: &str) -> tfpb::node_def::NodeDef {
    placeholder(name, DataType::DT_INT32, None)
}

pub fn compare<S: AsRef<str>>(
    graph: &[u8],
    inputs: Vec<(S, TfdTensor)>,
    output: &str,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    // Run TFD
    let model = tfdeploy::Model::for_reader(&*graph)?;
    let mut state = model.state();
    for (s, t) in &inputs {
        state
            .set_value(model.node_id_by_name(s.as_ref()).unwrap(), t.clone())
            .unwrap();
    }
    let output_id = model.node_id_by_name(output)?;
    state.compute_one(output_id)?;
    let found = &state.outputs[output_id].as_ref().unwrap();

    // Run Tensorflow
    let tf_inputs: Vec<(&str, TfdTensor)> = inputs
        .iter()
        .map(|(s, m)| (s.as_ref(), m.clone()))
        .collect();
    let expected = tf::for_slice(&graph)?.run(tf_inputs.clone(), output)?;

    prop_assert!(
        expected[0].shape() == found[0].shape() && expected[0].close_enough(&found[0]),
        "expected: {:?} found: {:?}",
        expected,
        found
    );

    // Check inference rules consistency
    let node = model.get_node(output)?;
    let inputs_vectors: Vec<TensorFact> = node.inputs
        .iter()
        .map(|(i, p)| {
            state.outputs[*i].as_ref().unwrap()[*p]
                .as_tensor()
                .clone()
                .into()
        })
        .collect();
    let output_vectors: Vec<TensorFact> = vec![
        state.outputs[output_id].as_ref().unwrap()[0]
            .as_tensor()
            .clone()
            .into(),
    ];

    info!("Checking inference on {}", output);
    let op = node.op();
    if let Err(e) = op.infer(inputs_vectors, output_vectors) {
        error!("{:?}", e);
        Err(e)?
    }

    Ok(())
}
