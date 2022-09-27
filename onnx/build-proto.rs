fn main() {
    let _ = std::fs::create_dir_all("src/prost");
    std::env::set_var("PROTOC", protobuf_src::protoc());
    prost_build::Config::new()
        .out_dir("src/prost")
        .compile_protos(&["protos/onnx/onnx.proto3"], &["protos/"])
        .unwrap();
}
