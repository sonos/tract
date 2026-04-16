use std::{env, fs, path};

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
        self.cc().file(filename).try_compile("dummy").is_ok()
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
                cc::Build::new().files(files).flag("-mfma").compile("x86_64_fma");
            }
        }
        "arm" | "armv7" => {
            let files = preprocess_files("arm32/armvfpv2", &[], &suffix, false);
            cc::Build::new().files(files).flag("-marm").flag("-mfpu=vfp").compile("armvfpv2");
            let files = preprocess_files(
                "arm32/armv7neon",
                &[("core", vec!["cortexa7", "cortexa9", "generic"])],
                &suffix,
                false,
            );
            cc::Build::new().files(files).flag("-marm").flag("-mfpu=neon").compile("armv7neon");
        }
        "aarch64" => {
            let files = preprocess_files(
                "arm64/arm64simd",
                &[("core", vec!["a53", "a55", "gen"])],
                &suffix,
                false,
            );
            cc::Build::new().files(files).compile("arm64simd");
            if include_amx() {
                let files = preprocess_files("arm64/apple_amx", &[], &suffix, false);
                cc::Build::new().files(files).compile("appleamx");
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
                config.cc().files(files).compile("arm64fp16")
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
        let fname = f.path().file_name().unwrap().to_str().unwrap().to_owned();
        if fname.ends_with(".S.j2") {
            let tmpl_file = fname;
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
                let out_name = tmpl_file.strip_suffix(".S.j2").unwrap();
                let file = out_dir.join(format!("{out_name}.S"));
                preprocess_file(f.path(), &file, &globals, suffix, needs_pragma);
                files.push(file);
            }
        }
    }
    files
}

fn strip_comments(s: &str) -> String {
    s.lines().map(|line| line.replace("//", ";")).collect::<Vec<String>>().join("\n")
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

    let msvc = use_masm();
    println!("cargo:rerun-if-changed={}", template.as_ref().to_string_lossy());
    let input = fs::read_to_string(&template).unwrap();
    let l = if os == "macos" {
        "L"
    } else if family == "windows" {
        ""
    } else {
        ".L"
    }
    .to_owned();
    let long = if msvc { "dd" } else { ".long" };
    let g = if os == "macos" || os == "ios" || os == "watchos" || os == "tvos" { "_" } else { "" };
    let align = if msvc { "align" } else { ".align" };
    let offset = if msvc { "offset" } else { "rip + " };

    let mut env = build_jinja_env(template.as_ref().parent().unwrap());

    let main_name = template.as_ref().file_name().unwrap().to_str().unwrap();
    env.add_template_owned(main_name.to_string(), input).unwrap_or_else(|e| {
        eprintln!("Parsing {}: {e}", template.as_ref().to_string_lossy());
        panic!();
    });

    let tmpl = env.get_template(main_name).unwrap();

    let mut ctx = std::collections::BTreeMap::<String, minijinja::Value>::new();
    ctx.insert("msvc".into(), msvc.into());
    ctx.insert("needs_pragma".into(), needs_pragma.into());
    ctx.insert("family".into(), family.into());
    ctx.insert("os".into(), os.into());
    ctx.insert("L".into(), l.into());
    ctx.insert("G".into(), g.into());
    ctx.insert("suffix".into(), suffix.into());
    ctx.insert("long".into(), long.into());
    ctx.insert("jump_table".into(), minijinja::Value::from_serialize(&jump_table()));
    ctx.insert("align".into(), align.into());
    ctx.insert("offset".into(), offset.into());

    for (k, v) in variants {
        ctx.insert(k.to_string(), (*v).into());
    }

    if include_amx() {
        let (amx_set, amx_clr) = amx_globals();
        ctx.insert("AMX_SET".into(), amx_set.into());
        ctx.insert("AMX_CLR".into(), amx_clr.into());
    }

    match tmpl.render(&ctx) {
        Ok(rendered) => {
            let rendered = if msvc { strip_comments(&rendered) } else { rendered };
            fs::write(&output, rendered).unwrap();
        }
        Err(e) => {
            eprintln!("Rendering {}: {e:#}", template.as_ref().to_string_lossy());
            panic!();
        }
    }
}

fn build_jinja_env(template_dir: &path::Path) -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();

    // Custom filters
    env.add_filter("float16", float16_filter);
    env.add_filter("setting", setting_filter);
    env.add_filter("lsl", lsl_filter);
    env.add_filter("u", unsigned_filter);

    // Custom function: amx("op", gpr) -> assembly .word directive
    env.add_function("amx", amx_function);

    // Load all partials (.j2 = Jinja2 macros/includes, .S.raw = raw assembly with brace escaping)
    for f in walkdir::WalkDir::new(template_dir) {
        let f = f.unwrap();
        if f.path().is_dir() {
            continue;
        }

        let fname = f.path().file_name().unwrap().to_str().unwrap().to_owned();
        let text = std::fs::read_to_string(f.path()).unwrap_or_else(|_| panic!("file {f:?}"));
        let text = if fname.ends_with(".S.raw") {
            Some(text.replace("{{", "{").replace("}}", "}"))
        } else if fname.ends_with(".j2") && !fname.ends_with(".S.j2") {
            Some(text)
        } else {
            None
        };
        if let Some(text) = text {
            let key = f
                .path()
                .strip_prefix(template_dir)
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
                .replace('\\', "/");
            println!("cargo:rerun-if-changed={}", f.path().to_string_lossy().replace('\\', "/"));
            env.add_template_owned(key, text).unwrap_or_else(|e| {
                eprintln!("Parsing partial {}: {e}", f.path().to_string_lossy());
                panic!();
            });
        }
    }

    env
}

fn make_extern_kernel_decl_macro(out_dir: &path::Path, suffix: &str) {
    let macro_decl = r#"
    macro_rules! extern_kernel {
        (fn $name: ident($($par_name:ident : $par_type: ty ),*) -> $rv: ty) => {
            paste! {
                unsafe extern "C" { pub fn [<$name _ _suffix>]($(par_name: $par_type),*) -> $rv; }
                pub use [<$name _ _suffix>] as $name;
            }
        }
    }"#
    .replace("_suffix", suffix);
    std::fs::write(out_dir.join("extern_kernel_macro.rs"), macro_decl).unwrap();
}

// --- Custom filters and functions ---

fn float16_filter(value: f64) -> String {
    let bits = half::f16::from_f32(value as f32).to_bits();
    format!(".short {bits}")
}

fn setting_filter(value: i64, bit: i64) -> String {
    let result = value | (1i64 << bit);
    result.to_string()
}

fn lsl_filter(value: i64, shift: i64) -> String {
    let result = value << shift;
    result.to_string()
}

fn unsigned_filter(value: i64) -> String {
    let result = value as u64;
    result.to_string()
}

fn amx_function(op: String, gpr: u32) -> String {
    let ops = [
        "ldx", "ldy", "stx", "sty", "ldz", "stz", "ldzi", "stzi", "extrx", "extry", "fma64",
        "fms64", "fma32", "fms32", "mac16", "fma16", "fms16", "setclr", "vecint", "vecfp",
        "matint", "matfp", "genlut",
    ];
    let op_id = ops.iter().position(|x| *x == op.as_str()).unwrap();
    format!(".word 0x{:x} \t\t\t\t// AMX {} x{}\n", 0x201000 + (op_id << 5) + gpr as usize, op, gpr)
}

fn amx_nop_op_imm5(op: usize, imm5: usize) -> String {
    format!("nop\nnop\nnop\n.word 0x{:x}\n", (0x201000 + (op << 5) + imm5))
}

fn amx_globals() -> (String, String) {
    (amx_nop_op_imm5(17, 0), amx_nop_op_imm5(17, 1))
}
