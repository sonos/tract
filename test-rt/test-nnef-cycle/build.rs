#[path = "suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime(
        "nnef_cycle",
        "suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );
}
