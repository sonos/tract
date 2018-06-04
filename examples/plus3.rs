extern crate ndarray;
extern crate tfdeploy;
use tfdeploy::tfpb;
use tfdeploy::tfpb::types::DataType::DT_FLOAT;

fn main() {
    let plus3 = tfpb::node()
        .op("Add")
        .name("output")
        .attr("T", DT_FLOAT)
        .input("input")
        .input("three");
    let konst = tfpb::node()
        .op("Const")
        .name("three")
        .attr("dtype", DT_FLOAT)
        .attr(
            "value",
            tfdeploy::tensor::Tensor::from(::ndarray::arr1(&[3.0f32]))
                .to_pb()
                .unwrap(),
        );
    let input = tfpb::node()
        .op("Placeholder")
        .name("input")
        .attr("dtype", DT_FLOAT);
    let graph = tfpb::graph().node(input).node(konst).node(plus3);
    graph.save_to("tests/plus3.pb").unwrap();
}
