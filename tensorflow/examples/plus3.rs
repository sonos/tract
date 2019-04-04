extern crate ndarray;
extern crate tract_core;
extern crate tract_tensorflow;
use tract_core::prelude::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::types::DataType::DT_FLOAT;
use tract_tensorflow::ToSharedTensor;

fn main() {
    let plus3 =
        tfpb::node().op("Add").name("output").attr("T", DT_FLOAT).input("input").input("three");
    let konst = tfpb::node()
        .op("Const")
        .name("three")
        .attr("dtype", DT_FLOAT)
        .attr("value", SharedTensor::from(::ndarray::arr1(&[3.0f32])).to_tf().unwrap());
    let input = tfpb::node().op("Placeholder").name("input").attr("dtype", DT_FLOAT);
    let graph = tfpb::graph().node(input).node(konst).node(plus3);
    graph.save_to("tests/plus3.pb").unwrap();
}
