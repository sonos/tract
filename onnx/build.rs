extern crate git2;
extern crate tempdir;

use std::{fs, path};

use tempdir::TempDir;

pub fn dir() -> path::PathBuf {
    match ::std::env::var("TRAVIS_BUILD_DIR") {
        Ok(t) => path::Path::new(&t).join("cached").join("onnx-checkout"),
        _ => ".onnx".into(),
    }
}

pub fn ensure_onnx_git_checkout() {
    use std::sync::Once;
    static START: Once = Once::new();
    START.call_once(|| {
        if !dir().exists() {
            println!("DEBUG ensure_onnx_git_checkout 1");
            let _ = fs::create_dir_all(dir().parent().unwrap());
            println!("DEBUG ensure_onnx_git_checkout 2");
            let tmp = TempDir::new("onnx").unwrap();
            println!("DEBUG ensure_onnx_git_checkout 3");
            let url = "https://github.com/onnx/onnx";
            println!("DEBUG ensure_onnx_git_checkout 4");
            ::git2::Repository::clone(url, &tmp).unwrap();
            println!("DEBUG ensure_onnx_git_checkout 5");
            fs::rename(tmp.into_path(), dir()).unwrap();
        }
    });
}

pub fn make_test_file(tests_set: &str) {
    use std::io::Write;
    ensure_onnx_git_checkout();
    let node_tests = dir().join("onnx/backend/test/data").join(tests_set);
    assert!(node_tests.exists());
    let working_list_file = path::PathBuf::from("tests")
        .join(tests_set)
        .with_extension("txt");
    let working_list: Vec<String> = if let Ok(list) = fs::read_to_string(&working_list_file) {
        list.split("\n")
            .map(|s| s.to_string())
            .filter(|s| s.trim().len() > 1 && s.trim().as_bytes()[0] != b'#')
            .collect()
    } else {
        vec![]
    };
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = path::PathBuf::from(out_dir);
    let test_dir = out_dir.join("tests");
    fs::create_dir_all(&test_dir).unwrap();
    let test_file = test_dir.join(tests_set).with_extension("rs");
    let mut rs = fs::File::create(test_file).unwrap();
    let mut tests: Vec<String> = fs::read_dir(&node_tests)
        .unwrap()
        .map(|de| de.unwrap().file_name().to_str().unwrap().to_owned())
        .collect();
    tests.sort();
    writeln!(rs, "mod {} {{", tests_set.replace("-", "_")).unwrap();
    for (s, optim) in &[("plain", false), ("optim", true)] {
        writeln!(rs, "mod {} {{", s).unwrap();
        for t in &tests {
            writeln!(rs, "#[test]").unwrap();
            if !working_list.contains(&t) {
                writeln!(rs, "#[ignore]").unwrap();
            }
            writeln!(rs, "fn {}() {{", t).unwrap();
            writeln!(
                rs,
                "::onnx::run_one({:?}, {:?}, {:?})",
                node_tests, t, optim
            ).unwrap();
            writeln!(rs, "}}").unwrap();
        }
        writeln!(rs, "}}").unwrap();
    }
    writeln!(rs, "}}").unwrap();
}

fn main() {
    ensure_onnx_git_checkout();
    make_test_file("node");
    make_test_file("real");
    make_test_file("simple");
    make_test_file("pytorch-operator");
    make_test_file("pytorch-converted");
}
