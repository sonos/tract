macro_rules! op_onnx {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["onnx"]
        }
    };
}

