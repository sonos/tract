#![allow(clippy::box_default)]

use liquid_core::Runtime;
use liquid_core::{Display_filter, Filter, FilterReflection, ParseFilter};
use liquid_core::{Value, ValueView};

use std::{env, ffi, fs, path};

#[path = "arm64/apple_amx/instructions.rs"]
mod apple_amx_instructions;

fn var(k: &str) -> String {
    env::var(k).unwrap()
}

fn use_masm() -> bool {
    env::var("CARGO_CFG_TARGET_ENV") == Ok("msvc".to_string()) && var("HOST").contains("-windows-")
}

fn include_amx() -> bool {
    let arch = var("CARGO_CFG_TARGET_ARCH");
    let os = var("CARGO_CFG_TARGET_OS");
    os == "macos"
        || (env::var("CARGO_FEATURE_APPLE_AMX_IOS").is_ok() && os == "ios" && arch == "aarch64")
}

fn jump_table() -> Vec<String> {
    println!("cargo:rerun-if-changed=src/frame/mmm/fuse.rs");
    std::fs::read_to_string("src/frame/mmm/fuse.rs")
        .unwrap()
        .lines()
        .filter(|l| l.contains("// jump_to:"))
        .map(|l| l.split("jump_to:").nth(1).unwrap().to_owned())
        .collect()
}

#[derive(Clone, Debug)]
struct ConfigForHalf {
    extra_flags: Vec<String>,
    needs_pragma: bool,
}

impl ConfigForHalf {
    fn new(extra_flags: Vec<String>, needs_pragma: bool) -> ConfigForHalf {
        ConfigForHalf { extra_flags, needs_pragma }
    }

    fn all() -> Vec<ConfigForHalf> {
        let mut configs = vec![];
        for extra_flags in
            [vec![], vec!["-march=armv8.2-a".to_string()], vec!["-mcpu=cortex-a55".to_string()]]
        {
            for needs_pragma in [false, true] {
                configs.push(ConfigForHalf::new(extra_flags.clone(), needs_pragma))
            }
        }
        configs
    }

    fn cc(&self) -> cc::Build {
        let mut cc = cc::Build::new();
        for flag in &self.extra_flags {
            cc.flag(flag);
        }
        cc
    }

    fn works(&self) -> bool {
        let filename = if self.needs_pragma {
            "arm64/arm64fp16/dummy_fmla_pragma.S"
        } else {
            "arm64/arm64fp16/dummy_fmla_no_pragma.S"
        };
        self.cc().static_flag(true).file(filename).try_compile("dummy").is_ok()
    }

    pub fn probe() -> Option<ConfigForHalf> {
        Self::all().iter().find(|c| c.works()).cloned()
    }
}

fn main() {
    let target = var("TARGET");
    let arch = var("CARGO_CFG_TARGET_ARCH");
    let os = var("CARGO_CFG_TARGET_OS");
    let out_dir = path::PathBuf::from(var("OUT_DIR"));

    let suffix = env!("CARGO_PKG_VERSION").replace(['-', '.'], "_");
    make_extern_kernel_decl_macro(&out_dir, &suffix);

    match arch.as_ref() {
        "x86_64" => {
            let mut files = preprocess_files("x86_64/fma", &[], &suffix, false);
            files.extend(preprocess_files("x86_64/avx512", &[], &suffix, false));

            if os == "windows" {
                if use_masm() {
                    let mut lib_exe = cc::windows_registry::find(&target, "lib.exe")
                        .expect("Could not find lib.exe");
                    lib_exe
                        .arg(format!("/out:{}", out_dir.join("x86_64_fma.lib").to_str().unwrap()));
                    for f in files {
                        let mut obj = f.clone();
                        obj.set_extension("o");
                        let mut ml_exe = cc::windows_registry::find(&target, "ml64.exe")
                            .expect("Could not find ml64.exe");
                        if !ml_exe
                            .arg("/Fo")
                            .arg(&obj)
                            .arg("/c")
                            .arg(&f)
                            .status()
                            .unwrap()
                            .success()
                        {
                            for (i, l) in std::fs::read_to_string(&f).unwrap().lines().enumerate() {
                                println!("{i:8} {l}");
                            }
                            panic!();
                        }
                        lib_exe.arg(obj);
                    }
                    assert!(lib_exe.status().unwrap().success());
                    println!("cargo:rustc-link-search=native={}", out_dir.to_str().unwrap());
                    println!("cargo:rustc-link-lib=static=x86_64_fma");
                } else {
                    cc::Build::new()
                        .files(files)
                        .flag("-mfma")
                        .flag("-mf16c")
                        .static_flag(true)
                        .compile("x86_64_fma");

                    // clang at least (dunno about gcc) outputs .asm files in the
                    // root directory that we need to clean up so we don't pollute
                    // the build output/working directory
                    let _ = fs::remove_file("fma_mmm_f32_16x6.asm");
                    let _ = fs::remove_file("fma_mmm_i32_8x8.asm");
                    let _ = fs::remove_file("fma_sigmoid_f32.asm");
                    let _ = fs::remove_file("fma_tanh_f32.asm");
                }
            } else {
                cc::Build::new().files(files).flag("-mfma").static_flag(true).compile("x86_64_fma");
            }
        }
        "arm" | "armv7" => {
            let files = preprocess_files("arm32/armvfpv2", &[], &suffix, false);
            cc::Build::new()
                .files(files)
                .flag("-marm")
                .flag("-mfpu=vfp")
                .static_flag(true)
                .compile("armvfpv2");
            let files = preprocess_files(
                "arm32/armv7neon",
                &[("core", vec!["cortexa7", "cortexa9", "generic"])],
                &suffix,
                false,
            );
            cc::Build::new()
                .files(files)
                .flag("-marm")
                .flag("-mfpu=neon")
                .static_flag(true)
                .compile("armv7neon");
        }
        "aarch64" => {
            let files = preprocess_files(
                "arm64/arm64simd",
                &[("core", vec!["a53", "a55", "gen"])],
                &suffix,
                false,
            );
            cc::Build::new().files(files).static_flag(true).compile("arm64simd");
            if include_amx() {
                let files = preprocess_files("arm64/apple_amx", &[], &suffix, false);
                cc::Build::new().files(files).static_flag(true).compile("appleamx");
            }
            if std::env::var("CARGO_FEATURE_NO_FP16").is_err() {
                let config =
                    ConfigForHalf::probe().expect("No configuration found for fp16 support");
                let files = preprocess_files(
                    "arm64/arm64fp16",
                    &[("core", vec!["a55", "gen"])],
                    &suffix,
                    config.needs_pragma,
                );
                config.cc().files(files).static_flag(true).compile("arm64fp16")
            }
        }
        _ => {}
    }
}

type Variant = (&'static str, Vec<&'static str>);

fn preprocess_files(
    input: impl AsRef<path::Path>,
    variants: &[Variant],
    suffix: &str,
    needs_pragma: bool,
) -> Vec<path::PathBuf> {
    let out_dir = path::PathBuf::from(var("OUT_DIR"));
    let mut files = vec![];
    let dir_entries = {
        let mut dir_entries: Vec<fs::DirEntry> =
            input.as_ref().read_dir().unwrap().map(|f| f.unwrap()).collect();
        dir_entries.sort_by_key(|a| a.path());
        dir_entries
    };
    for f in dir_entries {
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
                preprocess_file(f.path(), &file, &globals, suffix, needs_pragma);
                files.push(file);
            }
        }
    }
    files
}

fn strip_comments(s: String, msvc: bool) -> String {
    if msvc {
        s.lines().map(|line| line.replace("//", ";")).collect::<Vec<String>>().join("\n")
    } else {
        s
    }
}

fn preprocess_file(
    template: impl AsRef<path::Path>,
    output: impl AsRef<path::Path>,
    variants: &[(&'static str, &'static str)],
    suffix: &str,
    needs_pragma: bool,
) {
    println!("cargo:rerun-if-changed={}", template.as_ref().to_string_lossy());
    let family = var("CARGO_CFG_TARGET_FAMILY");
    let os = var("CARGO_CFG_TARGET_OS");

    // We also check to see if we're on a windows host, if we aren't, we won't be
    // able to use the Microsoft assemblers,
    let msvc = use_masm();
    println!("cargo:rerun-if-changed={}", template.as_ref().to_string_lossy());
    let mut input = fs::read_to_string(&template).unwrap();
    input = strip_comments(input, msvc);
    let l = if os == "macos" {
        "L"
    } else if family == "windows" {
        ""
    } else {
        ".L"
    }
    .to_owned();
    let long = if msvc { "dd" } else { ".long" };
    let g = if os == "macos" || os == "ios" { "_" } else { "" };
    // note: use .align with bytes instead of p2align since they both use direct bytes.
    let align = if msvc { "align" } else { ".align" };
    let mut globals = liquid::object!({
        "msvc": msvc,
        "needs_pragma": needs_pragma,
        "family": family,
        "os": os,
        "L": l,
        "G": g,
        "suffix": suffix,
        "long": long,
        "jump_table": jump_table(),
        "align": align,
        "offset": if msvc { "offset" } else { "rip + "},
    });
    for (k, v) in variants {
        globals.insert(k.to_string().into(), liquid::model::Value::scalar(*v));
    }
    let partials = load_partials(template.as_ref().parent().unwrap(), msvc);
    let mut parser = liquid::ParserBuilder::with_stdlib()
        .partials(liquid::partials::LazyCompiler::new(partials))
        .filter(F16);
    if include_amx() {
        parser = apple_amx_instructions::register(parser);
        globals.extend(apple_amx_instructions::globals());
    }
    if let Err(e) = parser
        .build()
        .and_then(|p| p.parse(&input))
        .and_then(|r| r.render_to(&mut fs::File::create(&output).unwrap(), &globals))
    {
        eprintln!("Processing {}", template.as_ref().to_string_lossy());
        eprintln!("{e}");
        panic!()
    }
}

fn load_partials(p: &path::Path, msvc: bool) -> liquid::partials::InMemorySource {
    let mut mem = liquid::partials::InMemorySource::new();
    for f in walkdir::WalkDir::new(p) {
        let f = f.unwrap();
        if f.path().is_dir() {
            continue;
        }

        let ext = f.path().extension().map(|s| s.to_string_lossy()).unwrap_or("".into());
        let text = std::fs::read_to_string(f.path()).unwrap_or_else(|_| panic!("file {f:?}"));
        let text = match ext.as_ref() {
            "tmpli" => Some(text.replace("{{", "{").replace("}}", "}")),
            "tmpliq" => Some(text),
            _ => None,
        };
        if let Some(text) = text {
            let text = strip_comments(text, msvc);
            let key =
                f.path().strip_prefix(p).unwrap().to_str().unwrap().to_owned().replace('\\', "/");
            println!("cargo:rerun-if-changed={}", f.path().to_string_lossy().replace('\\', "/"));

            mem.add(key, text);
        }
    }
    mem
}

fn make_extern_kernel_decl_macro(out_dir: &path::Path, suffix: &str) {
    let macro_decl = r#"
    macro_rules! extern_kernel {
        (fn $name: ident($($par_name:ident : $par_type: ty ),*) -> $rv: ty) => {
            paste! {
                extern "C" { pub fn [<$name _ _suffix>]($(par_name: $par_type),*) -> $rv; }
                pub use [<$name _ _suffix>] as $name;
            }
        }
    }"#
    .replace("_suffix", suffix);
    std::fs::write(out_dir.join("extern_kernel_macro.rs"), macro_decl).unwrap();
}

#[derive(Clone, ParseFilter, FilterReflection)]
#[filter(
    name = "float16",
    description = "Write a float16 constant with the .float16 directive in gcc, or as short in clang",
    parsed(F16Filter)
)]
pub struct F16;

#[derive(Debug, Default, Display_filter)]
#[name = "float16"]
struct F16Filter;

impl Filter for F16Filter {
    fn evaluate(
        &self,
        input: &dyn ValueView,
        _runtime: &dyn Runtime,
    ) -> liquid_core::Result<Value> {
        let input: f32 = input.as_scalar().unwrap().to_float().unwrap() as f32;
        let value = half::f16::from_f32(input);
        let bits = value.to_bits();
        Ok(format!(".short {bits}").to_value())
    }
}
