#[path = "suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime(
        "tests",
        "suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );
}
