use std::env::var;
use std::ffi;
use std::fs;
use std::path;

fn main() {
    let target = var("TARGET").unwrap();
    let family = var("CARGO_CFG_TARGET_FAMILY").unwrap_or("unknown".to_string());
    let arch = var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = var("CARGO_CFG_TARGET_OS").unwrap();
    let out_dir = path::PathBuf::from(var("OUT_DIR").unwrap());
    if arch == "x86_64" {
        let files = preprocess_files("x86_64/fma");
        if family == "windows" {
            let mut lib_exe =
                cc::windows_registry::find(&*target, "lib.exe").expect("Could not find lib.exe");
            lib_exe.arg(format!("/out:{}", out_dir.join("x86_64_fma.lib").to_str().unwrap()));
            for f in files {
                let mut obj = f.clone();
                obj.set_extension("o");
                let mut ml_exe = cc::windows_registry::find(&*target, "ml64.exe")
                    .expect("Could not find ml64.exe");
                assert!(ml_exe.arg("/Fo").arg(&obj).arg("/c").arg(f).status().unwrap().success());
                lib_exe.arg(obj);
            }
            assert!(lib_exe.status().unwrap().success());
            println!("cargo:rustc-link-search=native={}", out_dir.to_str().unwrap());
            println!("cargo:rustc-link-lib=static=x86_64_fma");
        } else if os == "macos" {
            let lib = out_dir.join("libx86_64_fma.a");
            if lib.exists() {
                std::fs::remove_file(lib).unwrap();
            }
            let mut lib = std::process::Command::new("xcrun");
            lib.args(&["ar", "-rv"]).arg(out_dir.join("libx86_64_fma.a"));
            for f in files {
                let mut obj = f.clone();
                obj.set_extension("o");
                assert!(std::process::Command::new("cc")
                    .args(&["-c", "-o"])
                    .arg(&obj)
                    .arg(&f)
                    .status()
                    .unwrap()
                    .success());
                lib.arg(obj);
            }
            assert!(lib.status().unwrap().success());
            println!("cargo:rustc-link-search=native={}", out_dir.to_str().unwrap());
            println!("cargo:rustc-link-lib=static=x86_64_fma");
        } else {
            cc::Build::new().files(files).flag("-mfma").static_flag(true).compile("x86_64_fma");
        }
    }
    if arch == "arm" || arch == "armv7" {
        let files = preprocess_files("arm32/armvfpv2");
        cc::Build::new()
            .files(files)
            .flag("-marm")
            .flag("-mfpu=vfp")
            .static_flag(true)
            .compile("armvfpv2");
        let files = preprocess_files("arm32/armv7neon");
        cc::Build::new()
            .files(files)
            .flag("-marm")
            .flag("-mfpu=neon")
            .static_flag(true)
            .compile("armv7neon");
    }
    if arch == "aarch64" {
        let files = preprocess_files("arm64/arm64simd");
        cc::Build::new().files(files).static_flag(true).compile("arm64");
    }
}

fn preprocess_files(input: impl AsRef<path::Path>) -> Vec<path::PathBuf> {
    let out_dir = path::PathBuf::from(var("OUT_DIR").unwrap());
    let mut v = vec![];
    for f in input.as_ref().read_dir().unwrap() {
        let f = f.unwrap();
        if f.path().extension() == Some(ffi::OsStr::new("tmpl")) {
            let mut file = out_dir.join(f.path().file_name().unwrap());
            file.set_extension("S");
            preprocess_file(f.path(), &file);
            v.push(file);
        }
    }
    v
}

fn preprocess_file(input: impl AsRef<path::Path>, output: impl AsRef<path::Path>) {
    let family = var("CARGO_CFG_TARGET_FAMILY").unwrap();
    let os = var("CARGO_CFG_TARGET_OS").unwrap();
    let mut globals = liquid::value::Object::new();
    globals.insert("family".into(), liquid::value::Value::scalar(family.clone()));
    globals.insert("os".into(), liquid::value::Value::scalar(os.clone()));
    let mut input = fs::read_to_string(input).unwrap();
    if family == "windows" {
        input =
            input.lines().map(|line| line.replace("//", ";")).collect::<Vec<String>>().join("\n");
    }
    globals.insert(
        "L".into(),
        liquid::value::Value::scalar(if os == "macos" {
            "L"
        } else if family == "windows" {
            ""
        } else {
            "."
        }),
    );
    liquid::ParserBuilder::with_liquid()
        .build()
        .unwrap()
        .parse(&*input)
        .unwrap()
        .render_to(&mut fs::File::create(&output).unwrap(), &globals)
        .unwrap();
}
