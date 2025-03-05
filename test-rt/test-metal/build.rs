#[path = "suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime("mlx", "suite::suite()", "runtime()", "Approximation::Approximate");

    suite::suite().test_runtime("mfa", "suite::suite()", "runtime()", "Approximation::Approximate");

    suite::suite().test_runtime(
        "ggml",
        "suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );
}
