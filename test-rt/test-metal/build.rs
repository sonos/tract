#[path = "ggml_suite.rs"]
mod ggml_suite;
#[path = "suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime("mlx", "suite::suite()", "runtime()", "Approximation::Approximate");

    suite::suite().test_runtime("mfa", "suite::suite()", "runtime()", "Approximation::Approximate");

    ggml_suite::suite().test_runtime(
        "ggml",
        "ggml_suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );

    ggml_suite::suite().test_runtime(
        "none",
        "ggml_suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );
}
