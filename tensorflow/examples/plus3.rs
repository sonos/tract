extern crate tract_tensorflow;
use std::convert::TryFrom;
use tract_hir::prelude::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::tensorflow::DataType::DtFloat;
use tract_tensorflow::tfpb::tensorflow::TensorProto;

fn main() {
    let plus3 =
        tfpb::node().op("Add").name("output").attr("T", DtFloat).input("input").input("three");
    let konst = tfpb::node()
        .op("Const")
        .name("three")
        .attr("dtype", DtFloat)
        .attr("value", TensorProto::try_from(&tensor1(&[3.0f32])).unwrap());
    let input = tfpb::node().op("Placeholder").name("input").attr("dtype", DtFloat);
    let graph = tfpb::graph().node(input).node(konst).node(plus3);
    graph.save_to("tests/plus3.pb").unwrap();
}
