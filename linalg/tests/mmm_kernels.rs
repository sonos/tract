//! Harness for the mmm kernel test cases registered by every kernel declaration.
//!
//! `libtest_mimic` is used instead of the default test harness for one reason: a case whose
//! kernel this host cannot run is reported *ignored*, where a `#[test]` fn can only return
//! early and pass. `--include-ignored` still runs them, and fails as the hardware dictates.
use libtest_mimic::{Arguments, Failed, Trial};
use tract_linalg::mmm::tests::cases;

fn main() {
    let trials = cases()
        .map(|case| {
            let name = format!("{}::{}::{}", (case.kernel)(), case.family, case.case);
            Trial::test(name, || (case.run)().map_err(Failed::from))
                .with_ignored_flag(!(case.runnable)())
        })
        .collect();
    libtest_mimic::run(&Arguments::from_args(), trials).exit();
}
