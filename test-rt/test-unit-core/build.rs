fn main() {
    let suite = suite_unit::suite().unwrap();
    suite.test_runtime(
        "default",
        "suite_unit::suite().unwrap()",
        "default()",
        "Approximation::Approximate",
    );
    suite.test_runtime(
        "unoptimized",
        "suite_unit::suite().unwrap()",
        "unoptimized()",
        "Approximation::Approximate",
    );
}
