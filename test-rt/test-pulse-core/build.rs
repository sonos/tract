fn main() {
    suite_pulse::suite().unwrap().test_runtime(
        "optimized",
        "suite_pulse::suite().unwrap()",
        "optimized()",
        "Approximation::Approximate",
    );
}
