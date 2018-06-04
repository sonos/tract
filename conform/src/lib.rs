#![allow(non_snake_case)]

#[macro_use]
extern crate error_chain;
extern crate ndarray;
#[macro_use]
extern crate proptest;
extern crate protobuf;
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
    let owned_names: Vec<String> = inputs.iter().map(|s| s.0.as_ref().to_string()).collect();
    let inputs: Vec<(&str, TfdTensor)> = inputs
        .into_iter()
        .zip(owned_names.iter())
        .map(|((_, m), s)| (&**s, m))
        .collect();
    let expected = tf::for_slice(&graph)?.run(inputs.clone(), output)?;
    let found = tfdeploy::Model::for_reader(&*graph)?.run_with_names(inputs, output)?;
    prop_assert!(
        expected[0].shape() == found[0].shape() && expected[0].close_enough(&found[0]),
        "expected: {:?} found: {:?}",
        expected,
        found
    );
    Ok(())
}
