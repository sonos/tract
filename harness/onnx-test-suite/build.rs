fn main() {
    gen_onnx_test_suite::runtime("default");
    gen_onnx_test_suite::runtime("unoptimized");
}
