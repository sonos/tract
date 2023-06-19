use std::{fs, path};

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

pub fn dir() -> path::PathBuf {
    let cache = ::std::env::var("CACHEDIR").unwrap_or_else(|_| "../../.cached".to_string());
    fs::create_dir_all(&cache).unwrap();
    path::PathBuf::from(cache).join("onnx")
}

pub fn ensure_onnx_git_checkout() {
    use std::sync::Once;
    static START: Once = Once::new();
    START.call_once(|| {
        use fs2::FileExt;
        fs::create_dir_all(dir()).unwrap();
        let lockfile = dir().join(".lock");
        let _lock = fs::File::create(lockfile).unwrap().lock_exclusive();
        for (v, _) in versions() {
            let wanted = dir().join(format!("onnx-{}", v.replace('.', "_")));
            if !wanted.join("onnx/backend/test/data").exists() {
                let tmp = wanted.with_extension("tmp");
                let _ = fs::remove_dir_all(&wanted);
                let _ = fs::remove_dir_all(&tmp);
                let run = std::process::Command::new("git")
                    .arg("clone")
                    .arg("https://github.com/onnx/onnx")
                    .arg(&tmp)
                    .status()
                    .unwrap();
                if !run.success() {
                    panic!("Failed to clone onnx")
                }
                let run = std::process::Command::new("git")
                    .arg("-C")
                    .arg(&tmp)
                    .arg("checkout")
                    .arg(format!("v{v}"))
                    .status()
                    .unwrap();
                if !run.success() {
                    panic!("Failed to checkout onnx branch")
                }
                fs::rename(tmp, wanted).unwrap();
            }
        }
        println!("onnx checkout done");
    });
}

fn instantiate_onnx_test_suite(runtime_name: &str) {
    use std::io::Write;
    ensure_onnx_git_checkout();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = path::PathBuf::from(out_dir);
    let test_dir = out_dir.join("tests");
    fs::create_dir_all(&test_dir).unwrap();

    let test_file = test_dir.join(runtime_name).with_extension("rs");
    let mut rs = fs::File::create(test_file).unwrap();

    for tests_set in "node real simple pytorch-operator pytorch-converted".split_whitespace() {
        let working_list_file = path::PathBuf::from(".").join(format!("{tests_set}.txt"));
        let working_list: Vec<(String, Vec<String>)> = fs::read_to_string(&working_list_file)
            .unwrap()
            .split('\n')
            .map(|s| s.to_string())
            .filter(|s| s.trim().len() > 1 && s.trim().as_bytes()[0] != b'#')
            .map(|s| {
                let mut splits = s.split_whitespace();
                (splits.next().unwrap().to_string(), splits.map(|s| s.to_string()).collect())
            })
            .collect();

        for (onnx_tag, opset) in versions() {
            let node_tests = dir()
                .join(format!("onnx-{}", onnx_tag.replace('.', "_")))
                .join("onnx/backend/test/data")
                .join(tests_set);
            assert!(node_tests.exists());
            println!("cargo:rerun-if-changed={}", working_list_file.to_str().unwrap());

            let identifier = format!(
                "{}_{}_{}",
                tests_set.replace('-', "_"),
                onnx_tag.replace('.', "_"),
                runtime_name.to_lowercase()
            );

            let mut tests: Vec<String> = fs::read_dir(&node_tests)
                .unwrap()
                .map(|de| de.unwrap().file_name().to_str().unwrap().to_owned())
                .collect();
            tests.sort();
            writeln!(rs, "#[allow(non_snake_case)]").unwrap();
            writeln!(rs, "mod {identifier} {{").unwrap();
            writeln!(rs, "use tract_core::internal::*;").unwrap();
            writeln!(rs, "use super::onnx::run_one;").unwrap();
            for t in &tests {
                let details = working_list.iter().find(|pair| &pair.0 == t).map(|pair| &pair.1);
                let ignore = details.is_none()
                    || details.unwrap().iter().any(|s| {
                        s.strip_prefix("since:")
                            .is_some_and(|since| since.parse::<usize>().unwrap() > opset)
                    });
                writeln!(rs, "#[test]").unwrap();
                if ignore {
                    writeln!(rs, "#[ignore]").unwrap();
                }
                writeln!(rs, "fn {t}() -> TractResult<()> {{").unwrap();
                writeln!(
                    rs,
                    "run_one({node_tests:?}, {t:?}, super::{runtime_name}(), &{:?})",
                    details.map(|v| &**v).unwrap_or(&[])
                )
                .unwrap();
                writeln!(rs, "}}").unwrap();
            }
            writeln!(rs, "}}").unwrap();
        }
    }
}

fn main() {
    instantiate_onnx_test_suite("default");
    instantiate_onnx_test_suite("unoptimized");
}
