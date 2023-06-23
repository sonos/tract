fn main() {
    let suite = suite_conv::suite().unwrap();
    suite.test_runtime("default", "suite_conv::suite().unwrap()", "default()");
    suite.test_runtime("unoptimized", "suite_conv::suite().unwrap()", "unoptimized()");
}
