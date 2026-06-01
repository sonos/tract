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

fn include_sme() -> bool {
    let arch = var("CARGO_CFG_TARGET_ARCH");
    let os = var("CARGO_CFG_TARGET_OS");
    arch == "aarch64" && (os == "macos" || os == "linux")
}

// Probe whether the target assembler can actually assemble SME instructions.
// Old binutils (e.g. the Debian stretch aarch64 cross-toolchain used in CI)
// predate SME and reject the mnemonics even with `.arch armv9-a+sme2`, which
// breaks the build. When the probe fails we skip the SME kernels entirely;
// the matching `tract_sme` cfg keeps the Rust side from referencing the
// (now absent) kernel symbols, and dispatch falls back to the portable path.
fn assembler_supports_sme() -> bool {
    cc::Build::new()
        .file("arm64/sme/dummy_sme.S")
        .cargo_metadata(false)
        .cargo_warnings(false)
        .warnings(false)
        .try_compile("tract_sme_probe")
        .is_ok()
}

// Probe whether the target assembler can encode FEAT_DotProd `sdot` (the
// indexed int8 form used by arm64simd_mmm_i32_8x8_dot). Old binutils — notably
// the Debian stretch aarch64 cross-toolchain in CI — predate FEAT_DotProd and
// reject `.cpu ...+dotprod` / `sdot` outright. When the probe fails we skip the
// SDOT kernel and the `tract_arm64_dotprod` cfg; the runtime falls back to the
// SMLAL 8x8 i32 kernel.
fn assembler_supports_dotprod() -> bool {
    cc::Build::new()
        .file("arm64/arm64simd/dummy_dotprod.S")
        .cargo_metadata(false)
        .cargo_warnings(false)
        .warnings(false)
        .try_compile("tract_dotprod_probe")
        .is_ok()
}

// Probe whether the target assembler can encode `vpdpbusd ymm` (AVX-512 VNNI
// with AVX-512 VL, i.e. the 256-bit form). binutils gained this in ~2.30
// (2018); the Debian stretch toolchain ships 2.28 and rejects the mnemonic.
// When the probe fails we skip the VNNI kernel and the `tract_avx512vnni` cfg;
// the runtime falls back to the AVX2 i32 path.
fn assembler_supports_avx512vnni() -> bool {
    cc::Build::new()
        .file("x86_64/avx512vnni/dummy_vnni.S")
        .cargo_metadata(false)
        .cargo_warnings(false)
        .warnings(false)
        .try_compile("tract_avx512vnni_probe")
        .is_ok()
}

// Probe whether the target assembler can actually assemble Intel AMX int8
// instructions (`ldtilecfg`, `tilezero`, `tdpbusd`, `tilerelease`). Older
// binutils (e.g. Debian stretch's gas 2.28) predate AMX and reject these
// mnemonics outright, which would break the x86_64 build for users on those
// toolchains. When the probe fails we skip the AMX kernel entirely; the
// matching `tract_amx_int8` cfg keeps the Rust side from referencing the
// (absent) kernel symbol, and `qmmm_i32` dispatch falls back to VNNI (or
// AVX2 when VNNI is itself unavailable).
fn assembler_supports_amx_int8() -> bool {
    cc::Build::new()
        .file("x86_64/avx512amx/dummy.S")
        .cargo_metadata(false)
        .cargo_warnings(false)
        .warnings(false)
        .try_compile("tract_amx_int8_probe")
        .is_ok()
}

// Probe whether the target assembler can assemble AMX bf16 instructions
// (`tdpbf16ps`). Both int8 and bf16 AMX mnemonics require binutils >= 2.34,
// so in practice this probe succeeds whenever `assembler_supports_amx_int8`
// does. Provided separately so the two cfgs are independently controlled
// and users on exotic toolchains can opt-out of just the bf16 kernel.
fn assembler_supports_amx_bf16() -> bool {
    cc::Build::new()
        .file("x86_64/avx512amx/dummy_bf16.S")
        .cargo_metadata(false)
        .cargo_warnings(false)
        .warnings(false)
        .try_compile("tract_amx_bf16_probe")
        .is_ok()
}

fn include_sve() -> bool {
    // SVE/SVE2 lives on ARMv9 server/mobile cores (Neoverse V1+/N2+, Cortex-X2+,
    // Graviton 3/4) — Linux aarch64. No Apple silicon has SVE.
    var("CARGO_CFG_TARGET_ARCH") == "aarch64" && var("CARGO_CFG_TARGET_OS") == "linux"
}

// Probe whether the C compiler supports SVE intrinsics (arm_sve.h + `+sve`).
// Old toolchains (e.g. the Debian stretch cross-gcc) lack them; when the probe
// fails we skip the SVE kernels and the `tract_sve` cfg, so the Rust side never
// references the (absent) symbols and dispatch falls back to NEON.
fn compiler_supports_sve() -> bool {
    let out_dir = path::PathBuf::from(var("OUT_DIR"));
    let probe = out_dir.join("sve_probe.c");
    fs::write(&probe, "#include <arm_sve.h>\nint p(void){ return (int)svcntw(); }\n").unwrap();
    cc::Build::new()
        .file(&probe)
        .flag("-march=armv8.2-a+sve")
        .cargo_metadata(false)
        .cargo_warnings(false)
        .warnings(false)
        .try_compile("tract_sve_probe")
        .is_ok()
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

    // `tract_sme` is set below only when both include_sme() and the assembler
    // SME probe succeed; declare it so rustc's unexpected-cfg lint stays quiet.
    println!("cargo:rustc-check-cfg=cfg(tract_sme)");
    // Set below only when include_sve() and the SVE compiler probe both pass.
    println!("cargo:rustc-check-cfg=cfg(tract_sve)");
    // Set below only when the aarch64 assembler probe for `sdot` passes.
    println!("cargo:rustc-check-cfg=cfg(tract_arm64_dotprod)");
    // Set below only when the x86_64 assembler probe for vpdpbusd ymm passes.
    println!("cargo:rustc-check-cfg=cfg(tract_avx512vnni)");
    // Set below only when the x86_64 assembler accepts AMX int8 mnemonics
    // (avoids breaking the build on toolchains predating AMX).
    println!("cargo:rustc-check-cfg=cfg(tract_amx_int8)");
    // Set below only when the assembler accepts AMX bf16 mnemonics (tdpbf16ps).
    println!("cargo:rustc-check-cfg=cfg(tract_amx_bf16)");

    match arch.as_ref() {
        "x86_64" => {
            let mut files = preprocess_files("x86_64/fma", &[], &suffix, false);
            // The VNNI kernel is compiled separately (conditional on a probe) to
            // avoid breaking old assemblers. Remove it from the main file list.
            files.retain(|f| {
                !f.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("avx512vnni"))
            });
            files.extend(preprocess_files("x86_64/avx512", &[], &suffix, false));

            // Pull the AMX kernel templates out of the generic fma bulk-compile
            // so they can be gated behind assembler probes below. All AMX
            // mnemonics require gas >= 2.34; old toolchains (Debian stretch's
            // binutils 2.28) would otherwise fail the whole build.
            //
            // Split by accumulator type:
            //   avx512amx_*_i32_* → tdpbssd   → gated on tract_amx_int8
            //   avx512amx_*_f32_* → tdpbf16ps → gated on tract_amx_bf16
            let amx_int8_files: Vec<path::PathBuf> = files
                .iter()
                .filter(|f| {
                    f.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("avx512amx_") && n.contains("_i32_"))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();
            let amx_bf16_files: Vec<path::PathBuf> = files
                .iter()
                .filter(|f| {
                    f.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("avx512amx_") && n.contains("_f32_"))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();
            files.retain(|f| !amx_int8_files.contains(f) && !amx_bf16_files.contains(f));

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
            // VNNI kernel compiled separately so old assemblers (binutils < 2.30,
            // e.g. Debian stretch) that can't encode `vpdpbusd ymm` don't break
            // the whole x86_64 build. The `tract_avx512vnni` cfg gates the
            // matching Rust extern declarations and dispatch registration.
            //
            // The template stays in x86_64/fma/ (alongside dispatcher.j2 and the
            // other partials it includes) so the jinja env can resolve its includes.
            if assembler_supports_avx512vnni() {
                let tmpl = path::Path::new("x86_64/fma/avx512vnni_mmm_i32_8x8.S.j2");
                let out = out_dir.join(format!("avx512vnni_mmm_i32_8x8_{suffix}.S"));
                preprocess_file(tmpl, &out, &[], &suffix, false);
                cc::Build::new().file(&out).flag("-mfma").compile("x86_64_avx512vnni");
                println!("cargo:rustc-cfg=tract_avx512vnni");
            }

            // AMX int8 kernel: compile only when the assembler accepts the
            // mnemonics, and the kernel template was actually pulled aside
            // above. Unix only for now (the .S uses the GAS intel-syntax
            // path). The `tract_amx_int8` cfg gates the Rust-side symbol
            // reference: when the probe fails on old toolchains (e.g. Debian
            // stretch's binutils 2.28), the kernel is omitted and `qmmm_i32`
            // dispatch falls back to VNNI or AVX2 with no build error.
            if os != "windows"
                && !amx_int8_files.is_empty()
                && assembler_supports_amx_int8()
            {
                cc::Build::new()
                    .files(&amx_int8_files)
                    .compile("x86_64_avx512amx");
                println!("cargo:rustc-cfg=tract_amx_int8");
            }

            // AMX bf16 kernel for f32 matmul (tdpbf16ps). Same toolchain
            // requirement and Unix-only constraint as the int8 path. When the
            // probe fails, the `tract_amx_bf16` cfg stays unset and
            // `plug_avx512amx_bf16` is compiled out — `mmm_f32` then falls
            // back to AVX-512 / FMA without any build error.
            if os != "windows"
                && !amx_bf16_files.is_empty()
                && assembler_supports_amx_bf16()
            {
                cc::Build::new()
                    .files(&amx_bf16_files)
                    .compile("x86_64_avx512amx_bf16");
                println!("cargo:rustc-cfg=tract_amx_bf16");
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
            let mut files = preprocess_files(
                "arm64/arm64simd",
                &[("core", vec!["a53", "a55", "gen"])],
                &suffix,
                false,
            );
            // The SDOT kernel is compiled separately (conditional on a probe) so
            // old assemblers (binutils < 2.30, e.g. Debian stretch) that can't
            // encode `sdot` don't break the whole arm64simd build. Remove it
            // from the main file list.
            files.retain(|f| {
                !f.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("_dot"))
            });
            cc::Build::new().files(files).compile("arm64simd");
            // The template stays in arm64/arm64simd/ (alongside the jinja partials
            // it includes) so the env can resolve its includes. The
            // `tract_arm64_dotprod` cfg gates the matching Rust extern + dispatch.
            if assembler_supports_dotprod() {
                let tmpl = path::Path::new("arm64/arm64simd/arm64simd_mmm_i32_8x8_dot.S.j2");
                let out = out_dir.join(format!("arm64simd_mmm_i32_8x8_dot_{suffix}.S"));
                preprocess_file(tmpl, &out, &[], &suffix, false);
                cc::Build::new().file(&out).compile("arm64simd_dot");
                println!("cargo:rustc-cfg=tract_arm64_dotprod");
            }
            if include_amx() {
                let files = preprocess_files("arm64/apple_amx", &[], &suffix, false);
                cc::Build::new().files(files).compile("appleamx");
            }
            if include_sme() && assembler_supports_sme() {
                let files = preprocess_files("arm64/sme", &[], &suffix, false);
                cc::Build::new().files(files).compile("sme");
                println!("cargo:rustc-cfg=tract_sme");
            }
            if include_sve() && compiler_supports_sve() {
                // VLA SVE kernels (C intrinsics, fixed symbols — not suffix-templated).
                cc::Build::new()
                    .file("arm64/sve/sve_mmm_f32.c")
                    .file("arm64/sve/sve_mmv_f32_64x1.c")
                    .file("arm64/sve/sve_mmm_i32.c")
                    .file("arm64/sve/sve_mmm_i32_64x1.c")
                    .file("arm64/sve/sve_rms_norm.c")
                    .flag("-march=armv8.2-a+sve")
                    .compile("tract_sve_kernels");
                // f16 kernels need native FP16 arithmetic (+fp16); compiled
                // separately so the +sve-only kernels above never gain fp16
                // codegen. Runtime-gated on has_fp16() as well as SVE2.
                cc::Build::new()
                    .file("arm64/sve/sve_mmm_f16.c")
                    .file("arm64/sve/sve_mmv_f16_64x1.c")
                    .flag("-march=armv8.2-a+sve+fp16")
                    .compile("tract_sve_f16_kernels");
                println!("cargo:rustc-cfg=tract_sve");
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

/// Replace `//` assembly comments with `;` for MSVC assembler.
/// Must be called on rendered output (not on Jinja2 source, which uses `//` for integer division).
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
    ctx.insert("jump_table".into(), minijinja::Value::from_serialize(jump_table()));
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
