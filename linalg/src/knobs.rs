//! Architecture-detection knobs.
//!
//! These are declared unconditionally so they appear in the stack-wide knob
//! listing on every platform. Each is consumed only by the relevant
//! architecture's detection code (`arm64`, `arm32`, …) and is inert elsewhere.

tract_data::declare_knob!(
    DOTPROD_DISABLE,
    bool,
    false,
    "TRACT_DOTPROD_DISABLE",
    "aarch64: disable the FEAT_DotProd (SDOT/UDOT) kernels, forcing the SMLAL fallback."
);

tract_data::declare_knob!(
    SVE_DISABLE,
    bool,
    false,
    "TRACT_SVE_DISABLE",
    "aarch64: disable SVE/SVE2 kernels, forcing the NEON path."
);

tract_data::declare_knob!(
    SME_DISABLE,
    bool,
    false,
    "TRACT_SME_DISABLE",
    "aarch64: disable SME kernels, forcing the SVE/NEON path."
);

tract_data::declare_knob!(
    CPU_AARCH64_KIND,
    Option<String>,
    None,
    "TRACT_CPU_AARCH64_KIND",
    "aarch64: force the detected CPU family (a53/a55/a72/a73/a75/neoverse/applem) instead of probing."
);

tract_data::declare_knob!(
    CPU_AARCH64_OVERRIDE_CPU_PART,
    Option<String>,
    None,
    "TRACT_CPU_AARCH64_OVERRIDE_CPU_PART",
    "aarch64: force the raw CPU part id (e.g. 0xd03) before the kind-lookup table runs."
);

tract_data::declare_knob!(
    CPU_ARM32_NEON,
    Option<bool>,
    None,
    "TRACT_CPU_ARM32_NEON",
    "armv7: force NEON detection on/off instead of probing /proc/cpuinfo."
);
