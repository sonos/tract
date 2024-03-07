#[path = "suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime(
        "as_blas",
        "suite::suite()",
        "as_blas()",
        "Approximation::Approximate",
    );
}
