fn main() {
    let suite = gen_onnx_test_suite::suite();
    suite.test_runtime("default", "gen_onnx_test_suite::suite()", "default()");
    suite.test_runtime("unoptimized", "gen_onnx_test_suite::suite()", "unoptimized()");
}
