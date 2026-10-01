#[path = "suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime(
        "nnef_cycle",
        "suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );
    suite::core_suite().test_runtime(
        "nnef_core_cycle",
        "suite::core_suite()",
        "runtime()",
        "Approximation::Approximate",
    );
}
