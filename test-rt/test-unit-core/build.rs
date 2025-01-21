fn main() {
    let suite = suite_unit::suite().unwrap();
    suite.test_runtime(
        "raw",
        "suite_unit::suite().unwrap()",
        "raw()",
        "Approximation::Approximate",
    );
    suite.test_runtime(
        "decluttered",
        "suite_unit::suite().unwrap()",
        "decluttered()",
        "Approximation::Approximate",
    );
    suite.test_runtime(
        "optimized",
        "suite_unit::suite().unwrap()",
        "optimized()",
        "Approximation::Approximate",
    );
}
