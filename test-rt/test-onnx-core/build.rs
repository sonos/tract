fn main() {
    let suite = suite_onnx::suite();
    suite.test_runtime("default", "suite_onnx::suite()", "default()");
    suite.test_runtime("unoptimized", "suite_onnx::suite()", "unoptimized()");
}
