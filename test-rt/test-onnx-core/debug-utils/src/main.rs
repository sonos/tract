use protobuf::Message;
use std::fs::File;
use tract_onnx::pb::{ModelProto, ValueInfoProto};

fn main() {
    let input = std::env::args().nth(1).unwrap();
    let output = std::env::args().nth(2).unwrap();
    let mut model =
        protobuf::parse_from_reader::<ModelProto>(&mut File::open(input).unwrap()).unwrap();
    let mut graph = model.take_graph();
    let all_outputs: Vec<tract_onnx::pb::ValueInfoProto> = graph
        .get_node()
        .iter()
        .flat_map(|n| {
            n.get_output().iter().map(|s| {
                let mut vip = ValueInfoProto::new();
                vip.set_name(s.to_string());
                vip
            })
        })
        .collect();
    graph.set_output(all_outputs.into());
    model.set_graph(graph);
    let mut f = File::create(output).unwrap();
    let mut stream = protobuf::stream::CodedOutputStream::new(&mut f);
    model.write_to(&mut stream).unwrap();
    stream.flush().unwrap();
}
