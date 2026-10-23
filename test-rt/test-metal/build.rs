#[path = "ggml_suite.rs"]
mod ggml_suite;
#[path = "suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime("mlx", "suite::suite()", "runtime()", "Approximation::Approximate");

    suite::suite().test_runtime("mfa", "suite::suite()", "runtime()", "Approximation::Approximate");

    suite::suite().test_runtime("mpp", "suite::suite()", "runtime()", "Approximation::Approximate");

    ggml_suite::suite().test_runtime(
        "ggml",
        "ggml_suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );

    // matmul_f16 holds only when its products go through the ggml GEMV.
    let mut none = ggml_suite::suite().clone();
    none.ignore(
        &|t| matches!(t, [suite, section, ..] if suite == "unit" && section == "matmul_f16"),
    );
    none.test_runtime("none", "ggml_suite::suite()", "runtime()", "Approximation::Approximate");
}
