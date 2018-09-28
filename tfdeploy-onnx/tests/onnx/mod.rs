use std::{fs, path};

use tempdir::TempDir;

use tfdeploy::*;
use tfdeploy_onnx::pb::TensorProto;
use tfdeploy_onnx::*;

use rayon::prelude::*;

pub const ONNX_DIR: &'static str = ".onnx";

pub fn dir() -> path::PathBuf {
    match ::std::env::var("TRAVIS_BUILD_DIR") {
        Ok(t) => path::Path::new(&t).join("cached").join("onnx-checkout"),
        _ => ".onnx".into(),
    }
}

pub fn ensure_onnx_git_checkout() -> TfdResult<()> {
    use std::sync::Once;
    static START: Once = Once::new();
    START.call_once(|| {
        if !dir().exists() {
            let _ = fs::create_dir_all(dir().parent().unwrap());
            let tmp = TempDir::new("onnx").unwrap();
            let url = "https://github.com/onnx/onnx";
            ::git2::Repository::clone(url, &tmp).unwrap();
            fs::rename(tmp.into_path(), dir()).unwrap();
        }
    });
    Ok(())
}

pub fn load_half_dataset(prefix: &str, path: &path::Path) -> TVec<Tensor> {
    let mut vec = tvec!();
    let len = fs::read_dir(path)
        .map_err(|e| format!("accessing {:?}, {:?}", path, e))
        .unwrap()
        .filter(|d| {
            d.as_ref()
                .unwrap()
                .file_name()
                .to_str()
                .unwrap()
                .starts_with(prefix)
        }).count();
    for i in 0..len {
        let filename = path.join(format!("{}_{}.pb", prefix, i));
        let mut file = fs::File::open(filename)
                .map_err(|e| format!("accessing {:?}, {:?}", path, e))
                    .unwrap();
        let tensor: TensorProto = ::protobuf::parse_from_reader(&mut file).unwrap();
        vec.push(tensor.to_tfd().unwrap())
    }
    vec
}

pub fn load_dataset(path: &path::Path) -> (TVec<Tensor>, TVec<Tensor>) {
    (
        load_half_dataset("input", path),
        load_half_dataset("output", path),
    )
}

#[derive(Debug, Serialize, Deserialize)]
struct DataJson {
    model_name: String,
    url: String,
}

pub fn run_one(root: &path::Path, test: &str) -> TfdResult<()> {
    fn it(root: &path::Path) -> TfdResult<()> {
        let path = root.join("model.onnx");
        let model = for_path(&path)
                .map_err(|e| format!("accessing {:?}, {:?}", path, e))?;
        let inputs: Vec<&str> = model.guess_inputs().iter().map(|n| &*n.name).collect();
        let outputs: Vec<&str> = model.guess_outputs().iter().map(|n| &*n.name).collect();
        let plan = SimplePlan::new(&model, &*inputs, &*outputs)?;
        for d in fs::read_dir(root)
                .map_err(|e| format!("accessing {:?}, {:?}", root, e))? {
            let d = d?;
            if d.metadata()?.is_dir() && d
                .file_name()
                .to_str()
                .unwrap()
                .starts_with("test_data_set_")
            {
                let (inputs, expected) = load_dataset(&d.path());
                let computed = plan.run(inputs)?.remove(0);
                for (a, b) in computed.iter().zip(expected.iter()) {
                    if !a.close_enough(b, true) {
                        bail!("Different result: got:{:?} expected:{:?}", a, b)
                    }
                }
            }
        }
        Ok(())
    }
    let test_path = root.join(test);
    let path = if test_path.join("data.json").exists() {
        use fs2::FileExt;
        let f = fs::File::open(test_path.join("data.json"))?;
        let _lock = f.lock_exclusive();
        let data: DataJson = ::serde_json::from_reader(f).map_err(|e| format!("{:?}", e))?;
        if !test_path.join(&data.model_name).exists() {
            let (_, body) = ::mio_httpc::CallBuilder::get()
                .url(&data.url)
                .unwrap()
                .max_response(1_000_000_000)
                .timeout_ms(600_000)
                .exec()
                .unwrap();
            let gz = ::flate2::read::GzDecoder::new(&*body);
            let mut tar = ::tar::Archive::new(gz);
            let tmp = test_path.join("tmp");
            let _ = fs::remove_dir_all(&tmp);
            tar.unpack(&tmp).unwrap();
            fs::rename(tmp.join(&data.model_name), test_path.join(&data.model_name)).unwrap();
            let _ = fs::remove_dir_all(&tmp);
        }
        test_path.join(&data.model_name)
    } else {
        test_path
    };
    use colored::Colorize;
    match ::std::panic::catch_unwind(|| it(&path)) {
        Ok(Ok(())) => println!("  - {} {}", test, "OK".green()),
        Ok(Err(e)) => {
            println!("  - {} {} {}", test, "ERROR".yellow(), e);
            Err(e)?
        }
        Err(_) => {
            println!("  - {} {}", test, "PANIC".bright_red());
            Err("PANIC")?
        }
    }
    return Ok(());
}

pub fn run_all(tests: &str) {
    use std::io::Write;
    ensure_onnx_git_checkout().unwrap();
    let node_tests = dir().join("onnx/backend/test/data").join(tests);
    assert!(node_tests.exists());
    let filter = ::std::env::var("ONNX_TEST_FILTER").ok();
    let working_list_file = path::PathBuf::from("tests")
        .join(tests)
        .with_extension("txt");
    let working_list: Vec<String> = if let Ok(list) = fs::read_to_string(&working_list_file) {
        list.split("\n").map(|s| s.to_string()).collect()
    } else {
        vec![]
    };
    let mut tests: Vec<String> = fs::read_dir(&node_tests)
        .unwrap()
        .map(|de| de.unwrap().file_name().to_str().unwrap().to_owned())
        .filter(|n| {
            filter
                .as_ref()
                .map(|filter| n.contains(filter))
                .unwrap_or(true)
                && (::std::env::var("CI").is_err() || working_list.contains(n))
        }).collect();
    tests.sort();
    let results: Vec<bool> = tests
        .par_iter()
        .map(|test| run_one(&node_tests, &test).is_ok())
        .collect();
    if filter.is_none() && tests
        .iter()
        .zip(results.iter())
        .any(|(t, &r)| r && !working_list.contains(&t))
    {
        let mut file = fs::File::create(working_list_file).unwrap();
        for (test, &r) in tests.iter().zip(results.iter()) {
            if r {
                write!(file, "{}\n", test).unwrap();
            }
        }
    }
    let regressions = tests
        .iter()
        .zip(results.iter())
        .filter(|(t, &r)| !r && working_list.contains(&t))
        .inspect(|(t, _)| println!("regression: {}", t))
        .count();
    if regressions != 0 {
        panic!("{} regressions", regressions)
    }
}
