#![allow(clippy::len_zero)]
use std::collections::HashMap;
use std::io::Write;

use dyn_clone::DynClone;
use itertools::Itertools;
use proptest::prelude::{any_with, Arbitrary};
use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};
use tract_core::internal::Approximation;
use tract_core::runtime::Runtime;
use tract_core::tract_data::TractResult;

pub fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

pub type TestResult = anyhow::Result<()>;

pub trait Test: 'static + Send + Sync + DynClone {
    fn run(&self, id: &str, runtime: &dyn Runtime) -> TestResult {
        self.run_with_approx(id, runtime, Approximation::Close)
    }
    fn run_with_approx(&self, id: &str, runtime: &dyn Runtime, approx: Approximation)
        -> TestResult;
}

dyn_clone::clone_trait_object!(Test);

#[derive(Clone)]
pub enum TestSuite {
    Node(HashMap<String, TestSuite>),
    Leaf(Box<dyn Test>, bool),
}

impl Default for TestSuite {
    fn default() -> Self {
        setup_test_logger();
        TestSuite::Node(Default::default())
    }
}

impl<T: Test> From<T> for TestSuite {
    fn from(value: T) -> Self {
        TestSuite::Leaf(Box::new(value), true)
    }
}

impl TestSuite {
    pub fn add(&mut self, id: impl ToString, test: impl Into<TestSuite>) {
        match self {
            TestSuite::Node(it) => {
                it.insert(id.to_string(), test.into());
            }
            TestSuite::Leaf(..) => panic!("Can not add test case to a leaf"),
        }
    }

    pub fn add_arbitrary<A: Arbitrary + Test + Clone>(
        &mut self,
        id: impl ToString,
        params: A::Parameters,
    ) where
        A::Parameters: Clone + Send + Sync,
    {
        self.add(id, ProptestWrapper::<A>(params));
    }

    pub fn with(mut self, id: impl ToString, test: impl Into<TestSuite>) -> Self {
        self.add(id, test);
        self
    }

    pub fn add_test(&mut self, id: impl ToString, test: impl Test, ignore: bool) {
        match self {
            TestSuite::Node(it) => {
                it.insert(id.to_string(), TestSuite::Leaf(Box::new(test), !ignore));
            }
            TestSuite::Leaf(..) => panic!("Can not add test case to a leaf"),
        }
    }

    pub fn get(&self, id: &str) -> &dyn Test {
        match self {
            TestSuite::Node(n) => {
                if let Some((head, tail)) = id.split_once("::") {
                    n[head].get(tail)
                } else {
                    n[id].get("")
                }
            }
            TestSuite::Leaf(test, _) => &**test,
        }
    }

    pub fn get_sub(&self, id: &str) -> &TestSuite {
        match self {
            TestSuite::Node(n) => {
                if let Some((head, tail)) = id.split_once("::") {
                    n[head].get_sub(tail)
                } else {
                    n[id].get_sub("")
                }
            }
            TestSuite::Leaf(_, _) => panic!(),
        }
    }

    pub fn get_sub_mut(&mut self, id: &str) -> &mut TestSuite {
        match self {
            TestSuite::Node(n) => {
                if let Some((head, tail)) = id.split_once("::") {
                    n.get_mut(head).unwrap().get_sub_mut(tail)
                } else {
                    n.get_mut(id).unwrap()
                }
            }
            TestSuite::Leaf(_, _) => panic!(),
        }
    }

    fn ignore_rec(&mut self, prefix: &mut Vec<String>, ign: &dyn Fn(&[String]) -> bool) {
        match self {
            TestSuite::Node(n) => {
                for (id, test) in n.iter_mut().sorted_by_key(|(k, _)| k.to_owned()) {
                    prefix.push(id.to_owned());
                    test.ignore_rec(prefix, ign);
                    prefix.pop();
                }
            }
            TestSuite::Leaf(_, run) => *run = *run && !ign(&*prefix),
        }
    }

    pub fn ignore(&mut self, ign: &dyn Fn(&[String]) -> bool) {
        self.ignore_rec(&mut vec![], ign)
    }

    fn dump(
        &self,
        test_suite: &str,
        runtime: &str,
        prefix: &str,
        id: &str,
        rs: &mut impl Write,
        approx: &str,
    ) -> TractResult<()> {
        let full_id = [prefix, id].into_iter().filter(|s| s.len() > 0).join("::");
        match self {
            TestSuite::Node(h) => {
                if id.len() > 0 {
                    writeln!(rs, "mod {id} {{").unwrap();
                    writeln!(rs, "use super::*;").unwrap();
                }
                for (id, test) in h.iter().sorted_by_key(|(k, _)| k.to_owned()) {
                    test.dump(test_suite, runtime, &full_id, id, rs, approx)?;
                }
                if id.len() > 0 {
                    writeln!(rs, "}}").unwrap();
                }
            }
            TestSuite::Leaf(_, run) => {
                writeln!(rs, "#[allow(non_snake_case)]").unwrap();
                writeln!(rs, "#[test]").unwrap();
                if !run {
                    writeln!(rs, "#[ignore]").unwrap();
                }
                writeln!(rs, "fn {id}() -> TractResult<()> {{",).unwrap();
                writeln!(
                    rs,
                    "    {test_suite}.get({full_id:?}).run_with_approx({full_id:?}, {runtime}, {approx})",
                    )
                    .unwrap();
                writeln!(rs, "}}").unwrap();
            }
        }
        Ok(())
    }

    pub fn test_runtime(&self, name: &str, test_suite: &str, runtime: &str, approx: &str) {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let out_dir = std::path::PathBuf::from(out_dir);
        let test_dir = out_dir.join("tests");
        std::fs::create_dir_all(&test_dir).unwrap();
        let test_file = test_dir.join(name).with_extension("rs");
        let mut rs = std::fs::File::create(test_file).unwrap();
        self.dump(test_suite, runtime, "", "", &mut rs, approx).unwrap();
    }
}

#[derive(Clone, Debug)]
struct ProptestWrapper<A: Arbitrary + Test + Clone>(A::Parameters)
where
    A::Parameters: Clone + Send + Sync;

impl<A: Arbitrary + Test + Clone> Test for ProptestWrapper<A>
where
    A::Parameters: Clone + Send + Sync,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let mut runner = TestRunner::new(Config {
            failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
            ..Config::default()
        });
        runner.run(&any_with::<A>(self.0.clone()), |v| {
            v.run_with_approx(id, runtime, approx)
                .map_err(|e| proptest::test_runner::TestCaseError::Fail(format!("{e:?}").into()))
        })?;
        Ok(())
    }
}
