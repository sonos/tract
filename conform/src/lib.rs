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
#[macro_use]
extern crate tfdeploy;
extern crate tfdeploy_tf;

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
use tfdeploy_tf::tfpb;
use tfdeploy::TVec;
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
    let mut model = tfdeploy_tf::for_reader(&*graph)?;
    model.set_inputs(&inputs
            .iter()
            .map(|pair| pair.0.as_ref())
            .collect::<Vec<&str>>())?;
    model.set_outputs(&[output])?;
    let plan = tfdeploy::SimplePlan::new(&model)?;
    let mut state = tfdeploy::plan::SimpleState::new(&plan)?;
    for (ix, (_, t)) in inputs.iter().enumerate() {
        state.set_input(ix, t.clone()).unwrap();
    }
    let output = &model.node_by_name(output)?;
    info!("Checking {} behaviour against tensorflow", output.name);
    state.compute_one(output.id)?;
    let found = &state.values[output.id].as_ref().unwrap();

    // Run Tensorflow
    let tf_inputs: Vec<(&str, TfdTensor)> = inputs
        .iter()
        .map(|(s, m)| (s.as_ref(), m.clone()))
        .collect();
    let expected = tf::for_slice(&graph)?.run(tf_inputs.clone(), &output.name)?;

    prop_assert!(
        expected[0].shape() == found[0].shape() && expected[0].close_enough(&found[0], true),
        "expected: {:?} found: {:?}",
        expected,
        found
    );
    Ok(())
}

pub fn infer<S: AsRef<str>>(
    graph: &[u8],
    inputs: Vec<(S, TfdTensor)>,
    output: &str,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    // Run TFD
    let mut model = tfdeploy_tf::for_reader(&*graph)?;
    model.set_inputs(&inputs
            .iter()
            .map(|pair| pair.0.as_ref())
            .collect::<Vec<&str>>())?;
    model.set_outputs(&[output])?;
    let plan = tfdeploy::SimplePlan::new(&model)?;
    let mut state = tfdeploy::plan::SimpleState::new(&plan)?;
    for (ix, (_, t)) in inputs.iter().enumerate() {
        state.set_input(ix, t.clone()).unwrap();
    }
    let output = &model.node_by_name(output)?;
    info!("Checking {} behaviour against tensorflow", output.name);
    state.compute_one(output.id)?;
    let _found = &state.values[output.id].as_ref().unwrap();

    info!("Checking inference consistency on {}", output.name);
    let inputs_vectors: TVec<TensorFact> = output
        .inputs
        .iter()
        .map(|outlet| {
            state.values[outlet.node].as_ref().unwrap()[outlet.slot]
                .as_tensor()
                .clone()
                .into()
        })
        .collect();
    let output_vectors: TVec<TensorFact> = tvec![
        state.values[output.id].as_ref().unwrap()[0]
            .as_tensor()
            .clone()
            .into(),
    ];

    let e = output.op.infer_facts(inputs_vectors, output_vectors);
    prop_assert!(e.is_ok(), "{:?}", e);

    Ok(())
}

#[allow(dead_code)]
pub fn setup_test_logger() {
    use simplelog::{Config, LevelFilter, TermLogger};
    let _ = TermLogger::init(LevelFilter::Trace, Config::default());
}
