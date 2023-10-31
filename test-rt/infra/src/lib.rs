#![allow(clippy::len_zero)]
use core::fmt;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Write;

use downcast_rs::Downcast;
use dyn_clone::DynClone;
use itertools::Itertools;
use proptest::prelude::{any_with, Arbitrary};
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};
use tract_core::internal::Approximation;
use tract_core::runtime::Runtime;
use tract_core::tract_data::TractResult;

pub fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

pub type TestResult = anyhow::Result<()>;

pub trait Test: Downcast + 'static + Send + Sync + DynClone {
    fn run(&self, id: &str, runtime: &dyn Runtime) -> TestResult {
        self.run_with_approx(id, runtime, Approximation::Close)
    }
    fn run_with_approx(&self, id: &str, runtime: &dyn Runtime, approx: Approximation)
        -> TestResult;
}
downcast_rs::impl_downcast!(Test);
dyn_clone::clone_trait_object!(Test);

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum TestStatus {
    OK,
    Ignored,
    Skipped,
}

#[derive(Clone)]
pub enum TestSuite {
    Node(HashMap<String, TestSuite>),
    Leaf(Box<dyn Test>, TestStatus),
}

impl Default for TestSuite {
    fn default() -> Self {
        setup_test_logger();
        TestSuite::Node(Default::default())
    }
}

impl<T: Test> From<T> for TestSuite {
    fn from(value: T) -> Self {
        TestSuite::Leaf(Box::new(value), TestStatus::OK)
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
        A::Parameters: Clone + Send + Sync + Debug,
    {
        self.add(id, ProptestWrapper::<A>(params, |_| true));
    }

     pub fn add_arbitrary_with_filter<A: Arbitrary + Test + Clone>(
        &mut self,
        id: impl ToString,
        params: A::Parameters,
        filter: fn(&A) -> bool,
    ) where
        A::Parameters: Clone + Send + Sync + Debug,
    {
        self.add(id, ProptestWrapper::<A>(params, filter));
    }

    pub fn with(mut self, id: impl ToString, test: impl Into<TestSuite>) -> Self {
        self.add(id, test);
        self
    }

    pub fn add_test(&mut self, id: impl ToString, test: impl Test) {
        self.add_test_with_status(id, test, TestStatus::OK)
    }

    pub fn add_test_with_status(&mut self, id: impl ToString, test: impl Test, status: TestStatus) {
        match self {
            TestSuite::Node(it) => {
                it.insert(id.to_string(), TestSuite::Leaf(Box::new(test), status));
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

    fn ignore_rec(
        &mut self,
        prefix: &mut Vec<String>,
        ignore: &dyn Fn(&[String], &dyn Test) -> bool,
    ) {
        match self {
            TestSuite::Node(n) => {
                for (id, test) in n.iter_mut().sorted_by_key(|(k, _)| k.to_owned()) {
                    prefix.push(id.to_owned());
                    test.ignore_rec(prefix, ignore);
                    prefix.pop();
                }
            }
            TestSuite::Leaf(case, run) => {
                if *run == TestStatus::OK && ignore(prefix, &**case) {
                    *run = TestStatus::Ignored
                }
            }
        }
    }

    pub fn ignore(&mut self, ign: &dyn Fn(&[String]) -> bool) {
        self.ignore_rec(&mut vec![], &|name, _| ign(name))
    }

    pub fn ignore_case(&mut self, ign: &dyn Fn(&[String], &dyn Test) -> bool) {
        self.ignore_rec(&mut vec![], ign)
    }

    fn skip_rec(&mut self, prefix: &mut Vec<String>, ign: &dyn Fn(&[String]) -> bool) {
        match self {
            TestSuite::Node(n) => {
                for (id, test) in n.iter_mut().sorted_by_key(|(k, _)| k.to_owned()) {
                    prefix.push(id.to_owned());
                    test.skip_rec(prefix, ign);
                    prefix.pop();
                }
            }
            TestSuite::Leaf(_, run) => {
                if ign(&*prefix) {
                    *run = TestStatus::Skipped
                }
            }
        }
    }

    pub fn skip(&mut self, ign: &dyn Fn(&[String]) -> bool) {
        self.skip_rec(&mut vec![], ign)
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
                    writeln!(rs, "#[allow(unused_imports)] use super::*;").unwrap();
                }
                for (id, test) in h.iter().sorted_by_key(|(k, _)| k.to_owned()) {
                    test.dump(test_suite, runtime, &full_id, id, rs, approx)?;
                }
                if id.len() > 0 {
                    writeln!(rs, "}}").unwrap();
                }
            }
            TestSuite::Leaf(_, status) => {
                if *status != TestStatus::Skipped {
                    writeln!(rs, "#[allow(non_snake_case)]").unwrap();
                    writeln!(rs, "#[test]").unwrap();
                    if *status == TestStatus::Ignored {
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

/*
trait TestFilter<A>: DynClone + Send + Sync {
    fn filter(&self, a: &A) -> bool;
}
dyn_clone::clone_trait_object!(<A> TestFilter<A>);

#[derive(Clone)]
struct AcceptAllFilter;

impl<A> TestFilter<A> for AcceptAllFilter {
    fn filter(&self, _a: &A) -> bool {
        true
    }
}

#[derive(Clone)]
struct FilterWrapper<A, F>(F);

impl<A: Clone, F: Clone> TestFilter<A> for FilterWrapper<A, F> {
    fn filter(&self, a: &A) -> bool {
        (self.0)(a)
    }
}
*/

#[derive(Clone)]
struct ProptestWrapper<A: Arbitrary + Test + Clone>(A::Parameters, fn(&A) -> bool)
where
    A::Parameters: Clone + Send + Sync + Debug;

impl<A: Arbitrary + Test + Clone + Send + Sync> Debug for ProptestWrapper<A>
where
    A::Parameters: Clone + Send + Sync + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<A: Arbitrary + Test + Clone> Test for ProptestWrapper<A>
where
    A::Parameters: Clone + Send + Sync + Debug,
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
        runner.run(
            &any_with::<A>(self.0.clone()).prop_filter("Test case filter", |a| self.1(a)),
            |v| {
                v.run_with_approx(id, runtime, approx).map_err(|e| {
                    proptest::test_runner::TestCaseError::Fail(format!("{e:?}").into())
                })
            },
        )?;
        Ok(())
    }
}
