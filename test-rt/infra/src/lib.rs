use std::collections::HashMap;

use dyn_clone::DynClone;
use itertools::Itertools;
use tract_core::runtime::Runtime;

type TestResult = anyhow::Result<()>;

pub trait Test: 'static + Send + Sync + DynClone {
    fn ignore(&self) -> bool;
    fn run(&self, runtime: &dyn Runtime) -> TestResult;
}

dyn_clone::clone_trait_object!(Test);

#[derive(Clone)]
pub struct TestSuite(pub HashMap<String, Box<dyn Test>>);

impl TestSuite {
    pub fn test_runtime(&self, name: &str, test_suite: &str, runtime: &str) {
        use std::io::Write;
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let out_dir = std::path::PathBuf::from(out_dir);
        let test_dir = out_dir.join("tests");
        std::fs::create_dir_all(&test_dir).unwrap();
        let test_file = test_dir.join(name).with_extension("rs");
        let mut rs = std::fs::File::create(test_file).unwrap();

        for (id, test) in self.0.iter().sorted_by_key(|(k, _)| k.clone()) {
            writeln!(rs, "#[allow(non_snake_case)]").unwrap();
            writeln!(rs, "#[test]").unwrap();
            if test.ignore() {
                writeln!(rs, "#[ignore]").unwrap();
            }
            writeln!(rs, "fn {id}() -> TractResult<()> {{",).unwrap();
            writeln!(rs, "    {test_suite}.0[{id:?}].run({runtime})").unwrap();
            writeln!(rs, "}}").unwrap();
        }
    }
}
