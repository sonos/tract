fn main() {
    let suite = suite_onnx::suite();
    suite.test_runtime("default", "suite_onnx::suite()", "default()", "Approximation::Approximate");
    suite.test_runtime(
        "unoptimized",
        "suite_onnx::suite()",
        "unoptimized()",
        "Approximation::Approximate",
    );
    suite.test_runtime(
        "as_blas",
        "suite_onnx::suite()",
        "as_blas()",
        "Approximation::Approximate",
    );
}
