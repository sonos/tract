pub fn suite() -> &'static infra::TestSuite {
    lazy_static::lazy_static! {
        static ref SUITE: infra::TestSuite  = mk_suite();
    };
    &SUITE
}

/// Every case the shared suites define. The backend declines to translate what
/// it has no kernel for, so an unsupported op runs on the CPU rather than
/// needing an entry here.
fn mk_suite() -> infra::TestSuite {
    infra::TestSuite::default()
        .with("onnx", suite_onnx::suite().clone())
        .with("unit", suite_unit::suite().unwrap().clone())
        .with("pulse", suite_pulse::suite().unwrap().clone())
}
