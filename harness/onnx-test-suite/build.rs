fn main() {
    gen_onnx_test_suite::runtime("default", |_| true);
    gen_onnx_test_suite::runtime("unoptimized", |_| true);
}
