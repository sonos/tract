#[path="suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime("nnef_cycle", "suite::suite()", "nnef_cycle()", "Approximation::Approximate");
}

