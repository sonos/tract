use std::{env, ffi, fs, path};

fn var(k: &str) -> String {
    env::var(k).unwrap()
}

fn use_masm() -> bool {
    env::var("CARGO_CFG_TARGET_ENV") == Ok("msvc".to_string()) && var("HOST").contains("-windows-")
}

fn main() {
    let target = var("TARGET");
    let arch = var("CARGO_CFG_TARGET_ARCH");
    let os = var("CARGO_CFG_TARGET_OS");
    let out_dir = path::PathBuf::from(var("OUT_DIR"));

    match arch.as_ref() {
        "x86_64" => {
            let files = preprocess_files("x86_64/fma", &[]);

            match os.as_ref() {
                "windows" => {
                    if use_masm() {
                        let mut lib_exe = cc::windows_registry::find(&*target, "lib.exe")
                            .expect("Could not find lib.exe");
                        lib_exe.arg(format!(
                            "/out:{}",
                            out_dir.join("x86_64_fma.lib").to_str().unwrap()
                        ));
                        for f in files {
                            let mut obj = f.clone();
                            for (i, l) in std::fs::read_to_string(&f).unwrap().lines().enumerate() {
                                println!("{:8} {}", i, l);
                            }
                            obj.set_extension("o");
                            let mut ml_exe = cc::windows_registry::find(&*target, "ml64.exe")
                                .expect("Could not find ml64.exe");
                            assert!(ml_exe
                                .arg("/Fo")
                                .arg(&obj)
                                .arg("/c")
                                .arg(f)
                                .status()
                                .unwrap()
                                .success());
                            lib_exe.arg(obj);
                        }
                        assert!(lib_exe.status().unwrap().success());
                        println!("cargo:rustc-link-search=native={}", out_dir.to_str().unwrap());
                        println!("cargo:rustc-link-lib=static=x86_64_fma");
                    } else {
                        cc::Build::new()
                            .files(files)
                            .flag("-mfma")
                            .static_flag(true)
                            .compile("x86_64_fma");

                        // clang at least (dunno about gcc) outputs .asm files in the
                        // root directory that we need to clean up so we don't pollute
                        // the build output/working directory
                        let _ = fs::remove_file("fma_mmm_f32_16x6.asm");
                        let _ = fs::remove_file("fma_mmm_i8_8x8.asm");
                        let _ = fs::remove_file("fma_sigmoid_f32.asm");
                    }
                }
                "macos" => {
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
                            .args(&["-arch", "x86_64"])
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
                }
                _ => {
                    cc::Build::new()
                        .files(files)
                        .flag("-mfma")
                        .static_flag(true)
                        .compile("x86_64_fma");
                }
            }
        }
        "arm" | "armv7" => {
            let files = preprocess_files("arm32/armvfpv2", &[]);
            cc::Build::new()
                .files(files)
                .flag("-marm")
                .flag("-mfpu=vfp")
                .static_flag(true)
                .compile("armvfpv2");
            let files = preprocess_files("arm32/armv7neon", &[]);
            cc::Build::new()
                .files(files)
                .flag("-marm")
                .flag("-mfpu=neon")
                .static_flag(true)
                .compile("armv7neon");
        }
        "aarch64" => {
            let files = preprocess_files("arm64/arm64simd", &[("core", vec!["a5x", "gen"])]);
            cc::Build::new().files(files).static_flag(true).compile("arm64");
        }
        _ => {}
    }
}

type Variant = (&'static str, Vec<&'static str>);

fn preprocess_files(input: impl AsRef<path::Path>, variants: &[Variant]) -> Vec<path::PathBuf> {
    let out_dir = path::PathBuf::from(var("OUT_DIR"));
    let mut files = vec![];
    for f in input.as_ref().read_dir().unwrap() {
        let f = f.unwrap();
        if f.path().extension() == Some(ffi::OsStr::new("tmpl")) {
            let tmpl_file = f.path().file_name().unwrap().to_str().unwrap().to_owned();
            let concerned_variants: Vec<&Variant> =
                variants.iter().filter(|v| tmpl_file.contains(v.0)).collect();
            let expanded_variants = concerned_variants.iter().map(|pair| pair.1.len()).product();
            for v in 0..expanded_variants {
                let mut tmpl_file = tmpl_file.clone();
                let mut id = v;
                let mut globals = vec![];
                for variable in variants {
                    let key = variable.0;
                    let value = variable.1[id % variable.1.len()];
                    globals.push((key, value));
                    tmpl_file = tmpl_file.replace(key, value);
                    id /= variable.1.len();
                }
                let mut file = out_dir.join(tmpl_file);
                file.set_extension("S");
                preprocess_file(f.path(), &file, &globals);
                files.push(file);
            }
        }
    }
    files
}

fn preprocess_file(
    input: impl AsRef<path::Path>,
    output: impl AsRef<path::Path>,
    variants: &[(&'static str, &'static str)],
) {
    let family = var("CARGO_CFG_TARGET_FAMILY");
    let os = var("CARGO_CFG_TARGET_OS");
    // We also check to see if we're on a windows host, if we aren't, we won't be
    // able to use the Microsoft assemblers,
    let msvc = use_masm();
    let mut input = fs::read_to_string(input).unwrap();
    if msvc {
        input =
            input.lines().map(|line| line.replace("//", ";")).collect::<Vec<String>>().join("\n");
    }
    let l = if os == "macos" {
        "L"
    } else if family == "windows" {
        ""
    } else {
        "."
    }
    .to_owned();
    let g = if os == "macos" || os == "ios" { "_" } else { "" };
    let mut globals = liquid::object!({
        "msvc": msvc,
        "family": family,
        "os": os,
        "L": l,
        "G": g,
    });
    for (k, v) in variants {
        globals.insert(k.to_string().into(), liquid::model::Value::scalar(*v));
    }
    liquid::ParserBuilder::with_stdlib()
        .build()
        .unwrap()
        .parse(&*input)
        .unwrap()
        .render_to(&mut fs::File::create(&output).unwrap(), &globals)
        .unwrap();
}
