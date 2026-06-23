//! Architecture-detection knobs.
//!
//! These are declared unconditionally so they appear in the stack-wide knob
//! listing on every platform. Each is consumed only by the relevant
//! architecture's detection code (`arm64`, `arm32`, …) and is inert elsewhere.

tract_data::declare_knob!(
    TRACT_DOTPROD_DISABLE,
    bool,
    false,
    "aarch64: disable the FEAT_DotProd (SDOT/UDOT) kernels, forcing the SMLAL fallback."
);

tract_data::declare_knob!(
    TRACT_SVE_DISABLE,
    bool,
    false,
    "aarch64: disable SVE/SVE2 kernels, forcing the NEON path."
);

tract_data::declare_knob!(
    TRACT_SME_DISABLE,
    bool,
    false,
    "aarch64: disable SME kernels, forcing the SVE/NEON path."
);

tract_data::declare_knob!(
    TRACT_CPU_AARCH64_KIND,
    Option<String>,
    None,
    "aarch64: force the detected CPU family (a53/a55/a72/a73/a75/neoverse/applem) instead of probing."
);

tract_data::declare_knob!(
    TRACT_CPU_AARCH64_OVERRIDE_CPU_PART,
    Option<String>,
    None,
    "aarch64: force the raw CPU part id (e.g. 0xd03) before the kind-lookup table runs."
);

tract_data::declare_knob!(
    TRACT_CPU_ARM32_NEON,
    Option<bool>,
    None,
    "armv7: force NEON detection on/off instead of probing /proc/cpuinfo."
);

tract_data::declare_knob!(
    TRACT_AVX512_FMA_UNITS,
    Option<String>,
    None,
    "x86_64: force the 512-bit FMA-port count (1 or 2) instead of runtime-probing; gates the zmm VNNI 16x16 kernel."
);

tract_data::declare_knob!(
    TRACT_AMX_BF16,
    bool,
    false,
    "x86_64: opt in to the lossy AMX bf16 kernel for f32 matmul (operands truncated to bf16, ~1/2^8 relative error per multiply). Off by default."
);
