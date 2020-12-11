use std::{fs, path};

fn versions() -> Vec<&'static str> {
    let mut versions = vec![];
    if cfg!(feature = "onnx_1_4_1") {
        versions.push("1.4.1");
    }
    if cfg!(feature = "onnx_1_5_0") {
        versions.push("1.5.0");
    }
    if cfg!(feature = "onnx_1_6_0") {
        versions.push("1.6.0");
    }
    if cfg!(feature = "onnx_1_7_0") {
        versions.push("1.7.0");
    }
    versions
}

pub fn dir() -> path::PathBuf {
    let cache = ::std::env::var("CACHEDIR").ok().unwrap_or("../../.cached".to_string());
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
        let _lock = fs::File::create(&lockfile).unwrap().lock_exclusive();
        for v in versions() {
            let wanted = dir().join(format!("onnx-{}", v.replace(".", "_")));
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
                    .arg(format!("v{}", v))
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

#[derive(PartialEq, Copy, Clone, Debug)]
enum Mode {
    Plain,
    Optim,
    NNEF,
}

pub fn make_test_file(root: &mut fs::File, tests_set: &str, onnx_tag: &str) {
    use std::io::Write;
    use Mode::*;
    ensure_onnx_git_checkout();
    let node_tests = dir()
        .join(format!("onnx-{}", onnx_tag.replace(".", "_")))
        .join("onnx/backend/test/data")
        .join(tests_set);
    assert!(node_tests.exists());
    let working_list_file =
        path::PathBuf::from(".").join(format!("{}-{}.txt", tests_set, onnx_tag));
    println!("cargo:rerun-if-changed={}", working_list_file.to_str().unwrap());
    let working_list: Vec<(String, Vec<String>)> = fs::read_to_string(&working_list_file)
        .unwrap()
        .split("\n")
        .map(|s| s.to_string())
        .filter(|s| s.trim().len() > 1 && s.trim().as_bytes()[0] != b'#')
        .map(|s| {
            let mut splits = s.split_whitespace();
            (splits.next().unwrap().to_string(), splits.map(|s| s.to_string()).collect())
        })
        .collect();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = path::PathBuf::from(out_dir);
    let test_dir = out_dir.join("tests");
    let tests_set_ver = format!("{}_{}", tests_set.replace("-", "_"), onnx_tag.replace(".", "_"));

    writeln!(root, "include!(concat!(env!(\"OUT_DIR\"), \"/tests/{}.rs\"));", tests_set_ver)
        .unwrap();

    let test_file = test_dir.join(&tests_set_ver).with_extension("rs");
    let mut rs = fs::File::create(test_file).unwrap();
    let mut tests: Vec<String> = fs::read_dir(&node_tests)
        .unwrap()
        .map(|de| de.unwrap().file_name().to_str().unwrap().to_owned())
        .collect();
    tests.sort();
    writeln!(rs, "mod {} {{", tests_set_ver).unwrap();
    for &mode in &[Plain, Optim, NNEF] {
        writeln!(rs, "mod {} {{", format!("{:?}", mode).to_lowercase()).unwrap();
        writeln!(rs, "use tract_core::internal::*;").unwrap();
        writeln!(rs, "use crate::onnx::{{run_one, Mode}};").unwrap();
        for t in &tests {
            writeln!(rs, "#[test]").unwrap();
            let pair = working_list.iter().find(|pair| &*pair.0 == &*t);
            let ignore = pair.is_none()
                || match mode {
                    Mode::Plain => false,
                    Mode::Optim => pair.as_ref().unwrap().1.contains(&"not-typable".to_string()),
                    Mode::NNEF => {
                        pair.as_ref().unwrap().1.contains(&"not-typable".to_string())
                            || pair.as_ref().unwrap().1.contains(&"not-nnef".to_string())
                    }
                };
            if ignore {
                writeln!(rs, "#[ignore]").unwrap();
            }
            let more = pair.map(|p| &*p.1).unwrap_or(&[]);
            writeln!(rs, "fn {}() -> TractResult<()> {{", t).unwrap();
            writeln!(rs, "run_one({:?}, {:?}, Mode::{:?}, &{:?})", node_tests, t, mode, more)
                .unwrap();
            writeln!(rs, "}}").unwrap();
        }
        writeln!(rs, "}}").unwrap();
    }
    writeln!(rs, "}}").unwrap();
}

fn main() {
    ensure_onnx_git_checkout();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = path::PathBuf::from(out_dir);
    let test_dir = out_dir.join("tests");
    fs::create_dir_all(&test_dir).unwrap();
    let mut root = fs::File::create(test_dir.join("root.rs")).unwrap();
    for set in "node real simple pytorch-operator pytorch-converted".split_whitespace() {
        for ver in versions() {
            make_test_file(&mut root, set, ver);
        }
    }
}
