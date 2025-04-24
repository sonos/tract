use log::*;
use prost::Message;
use std::path::PathBuf;
use tract_hir::internal::*;
use tract_onnx::data_resolver::FopenDataResolver;
use tract_onnx::pb::TensorProto;
use tract_onnx::tensor::load_tensor;

use infra::{Test, TestStatus, TestSuite};

pub fn suite() -> &'static TestSuite {
    lazy_static::lazy_static! {
        static ref SUITE: TestSuite = full();
    };
    &SUITE
}

const MANIFEST_NODE: &str = include_str!("../node.txt");
const MANIFEST_SIMPLE: &str = include_str!("../simple.txt");
const MANIFEST_PYTORCH_CONVERTED: &str = include_str!("../pytorch-converted.txt");
const MANIFEST_PYTORCH_OPERATOR: &str = include_str!("../pytorch-operator.txt");

#[derive(Clone, Debug)]
struct OnnxTestCase {
    path: PathBuf,
    ignore_output_shapes: bool,
    ignore_output_type: bool,
    input: Option<String>,
}

impl Test for OnnxTestCase {
    fn run_with_approx(
        &self,
        _suite: &str,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TractResult<()> {
        setup_test_logger();
        let model_file = self.path.join("model.onnx");
        info!("Loading {model_file:?}");
        let mut onnx = tract_onnx::onnx();

        // hack: some tests (test_nonmaxsuppression_*) include the output shapes in the onnx model
        // even though there should be no way of knowing them at optimization time. This breaks
        // the solver.
        if self.ignore_output_shapes {
            onnx = onnx.with_ignore_output_shapes(true);
        }
        // in some other cases, we need to deal with a tdim vs i64 mismatch (test for Shape, and Size)
        if self.ignore_output_type {
            onnx = onnx.with_ignore_output_types(true);
        }

        trace!("Proto Model:\n{:#?}", onnx.proto_model_for_path(&model_file));
        for d in std::fs::read_dir(&self.path)? {
            let mut model = onnx.model_for_path(&model_file)?;
            let d = d?;
            if d.metadata().unwrap().is_dir()
                && d.file_name().to_str().unwrap().starts_with("test_data_set_")
            {
                let data_path = d.path();
                let mut inputs = load_half_dataset("input", &data_path);
                if let Some(input) = &self.input {
                    let mut actual_inputs = vec![];
                    let mut actual_input_values = tvec![];
                    let input_outlets = model.input_outlets()?.to_vec();
                    for (ix, outlet) in input_outlets.iter().enumerate() {
                        if &model.node(outlet.node).name == input {
                            actual_inputs.push(*outlet);
                            actual_input_values.push(inputs[ix].clone());
                        } else {
                            model.node_mut(outlet.node).op =
                                Box::new(tract_hir::ops::konst::Const::new(
                                    inputs[ix].clone().into_arc_tensor(),
                                )?);
                        }
                    }
                    model.set_input_outlets(&actual_inputs)?;
                    inputs = actual_input_values;
                }
                info!("Analyse");
                trace!("Model:\n{model:#?}");
                model.analyse(false)?;
                model = model.incorporate()?;
                let model = model.into_typed()?.into_decluttered()?;
                info!("Test model (mode: {}) {:#?}", runtime.name(), self.path);
                let runnable = runtime.prepare(model)?;
                run_model(&*runnable, inputs, &data_path, approx)?;
                info!("Test model (mode: {}) {:#?} OK.", runtime.name(), self.path);
            }
        }
        Ok(())
    }
}

fn versions() -> Vec<(&'static str, usize)> {
    let mut versions = vec![];
    if cfg!(feature = "onnx_1_4_1") {
        versions.push(("1.4.1", 9));
    }
    if cfg!(feature = "onnx_1_5_0") {
        versions.push(("1.5.0", 10));
    }
    if cfg!(feature = "onnx_1_6_0") {
        versions.push(("1.6.0", 11));
    }
    if cfg!(feature = "onnx_1_7_0") {
        versions.push(("1.7.0", 12));
    }
    if cfg!(feature = "onnx_1_8_1") {
        versions.push(("1.8.1", 13));
    }
    if cfg!(feature = "onnx_1_9_0") {
        versions.push(("1.9.0", 14));
    }
    if cfg!(feature = "onnx_1_10_2") {
        versions.push(("1.10.2", 15));
    }
    if cfg!(feature = "onnx_1_11_0") {
        versions.push(("1.11.0", 16));
    }
    if cfg!(feature = "onnx_1_12_0") {
        versions.push(("1.12.0", 17));
    }
    if cfg!(feature = "onnx_1_13_0") {
        versions.push(("1.13.0", 18));
    }
    versions
}

pub fn dir() -> PathBuf {
    let cache = ::std::env::var("CACHEDIR").unwrap_or_else(|_| "../../.cached".to_string());
    std::fs::create_dir_all(&cache).unwrap();
    PathBuf::from(cache).join("onnx")
}

pub fn ensure_onnx_git_checkout() {
    use std::sync::Once;
    static START: Once = Once::new();
    START.call_once(|| {
        use fs2::FileExt;
        std::fs::create_dir_all(dir()).unwrap();
        let lockpath = dir().join(".lock");
        let lockfile = std::fs::File::create(lockpath).unwrap();
        lockfile.lock_exclusive().unwrap();
        for (v, _) in versions() {
            let wanted = dir().join(format!("onnx-{}", v.replace('.', "_")));
            if !wanted.join("onnx/backend/test/data").exists() {
                let df = std::process::Command::new("df").arg("-h").output().unwrap();
                dbg!(df);
                let tmp = wanted.with_extension("tmp");
                let _ = std::fs::remove_dir_all(&wanted);
                let _ = std::fs::remove_dir_all(&tmp);
                let run = std::process::Command::new("git")
                    .args(["clone", "--depth=1", "--branch"])
                    .arg(format!("v{v}"))
                    .arg("https://github.com/onnx/onnx")
                    .arg(&tmp)
                    .status()
                    .unwrap();
                if !run.success() {
                    panic!("Failed to clone onnx")
                }
                std::fs::rename(tmp, wanted).unwrap();
            }
        }
    });
}

fn full() -> TestSuite {
    ensure_onnx_git_checkout();
    let mut suite = TestSuite::default();
    for (tests_set, manifest) in [
        ("node", MANIFEST_NODE),
        ("simple", MANIFEST_SIMPLE),
        ("pytorch-operator", MANIFEST_PYTORCH_OPERATOR),
        ("pytorch-converted", MANIFEST_PYTORCH_CONVERTED),
    ] {
        let working_list: Vec<(String, Vec<String>)> = manifest
            .split('\n')
            .map(|s| s.to_string())
            .filter(|s| s.trim().len() > 1 && s.trim().as_bytes()[0] != b'#')
            .map(|s| {
                let mut splits = s.split_whitespace();
                (splits.next().unwrap().to_string(), splits.map(|s| s.to_string()).collect())
            })
            .collect();

        let mut tags = TestSuite::default();
        for (onnx_tag, opset) in versions() {
            let node_tests = dir()
                .join(format!("onnx-{}", onnx_tag.replace('.', "_")))
                .join("onnx/backend/test/data")
                .join(tests_set);
            assert!(node_tests.exists());

            let identifier = "v".to_string() + &onnx_tag.replace('.', "_");

            let tests: Vec<String> = std::fs::read_dir(&node_tests)
                .unwrap()
                .map(|de| de.unwrap().file_name().to_str().unwrap().to_owned())
                .collect();
            let mut units = TestSuite::default();
            for t in &tests {
                let details = working_list.iter().find(|pair| &pair.0 == t).map(|pair| &*pair.1);
                let ignored = details.is_none()
                    || details.unwrap().iter().any(|s| {
                        s.strip_prefix("since:")
                            .map(|since| since.parse::<usize>().unwrap() > opset)
                            .unwrap_or(false)
                    });
                let ignore_output_shapes =
                    details.unwrap_or_default().iter().any(|s| s == "onnx-ignore-output-shape");
                let ignore_output_type =
                    details.unwrap_or_default().iter().any(|s| s == "onnx-ignore-output-type");
                let input = details
                    .unwrap_or_default()
                    .iter()
                    .find_map(|s| s.strip_prefix("input:"))
                    .map(|s| s.to_owned());
                units.add_test_with_status(
                    t,
                    OnnxTestCase {
                        path: node_tests.join(t),
                        ignore_output_type,
                        ignore_output_shapes,
                        input,
                    },
                    if ignored { TestStatus::Ignored } else { TestStatus::OK },
                );
            }
            tags.add(identifier, units);
        }
        suite.add(tests_set.replace('-', "_"), tags);
    }
    suite
}

#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

pub fn load_half_dataset(prefix: &str, path: &std::path::Path) -> TVec<Tensor> {
    let mut vec = tvec!();
    let len = std::fs::read_dir(path)
        .map_err(|e| format!("accessing {path:?}, {e:?}"))
        .unwrap()
        .filter(|d| d.as_ref().unwrap().file_name().to_str().unwrap().starts_with(prefix))
        .count();
    for i in 0..len {
        let filename = path.join(format!("{prefix}_{i}.pb"));
        let bytes = bytes::Bytes::from(std::fs::read(filename).unwrap());
        let tensor = TensorProto::decode(bytes).unwrap();
        let tensor = load_tensor(&FopenDataResolver, &tensor, None).unwrap();
        vec.push(tensor)
    }
    debug!("{path:?}: {vec:?}");
    vec
}

fn run_model(
    model: &dyn Runnable,
    inputs: TVec<Tensor>,
    data_path: &std::path::Path,
    approx: Approximation,
) -> TractResult<()> {
    let expected = load_half_dataset("output", data_path);
    trace!("Loaded output asserts: {expected:?}");
    let inputs = inputs.into_iter().map(|t| t.into_tvalue()).collect();
    let computed = model.run(inputs)?;
    if computed.len() != expected.len() {
        panic!(
            "For {:?}, different number of results: got:{} expected:{}",
            data_path,
            computed.len(),
            expected.len()
        );
    }
    for (ix, (a, b)) in computed.iter().zip(expected.iter()).enumerate() {
        //                println!("computed: {:?}", computed[ix].dump(true));
        //                println!("expected: {:?}", expected[ix].dump(true));
        if let Err(e) = a.close_enough(b, approx) {
            bail!(
                "For {:?}, different ({approx:?}) result for output #{}:\ngot:\n{:?}\nexpected:\n{:?} \n{}",
                data_path,
                ix,
                a.cast_to::<f32>().unwrap().to_array_view::<f32>().unwrap(),
                b.cast_to::<f32>().unwrap().to_array_view::<f32>().unwrap(),
                e //                e.display_chain()
            );
        }
    }
    Ok(())
}
