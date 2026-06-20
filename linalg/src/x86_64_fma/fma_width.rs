// How many 512-bit FMA / VNNI issue ports does the running core have?
//
// The zmm-wide `avx512vnni_mmm_i32_16x16` kernel is only worth selecting over
// the ymm-wide `avx512vnni_mmm_i32_8x8` on cores that can retire a 512-bit
// VPDPBUSD as fast as a 256-bit one -- i.e. cores with *two* 512-bit FMA units
// (Cascade/Cooper Lake, Ice Lake-SP, Sapphire/Emerald/Granite Rapids). On a
// single-512-FMA core (every AVX-512 *client* part -- Cannon/Ice-U/Tiger/Rocket
// Lake -- and AMD Zen4/5, whose 512-bit ops are double-pumped) one 512-bit
// VPDPBUSD/cycle delivers the same MAC/s as two 256-bit VPDPBUSD/cycle, so the
// 16x16 tile gives *zero* throughput upside while adding A-packing, the
// 16-column +128 bias correction and a bigger epilogue -- a net regression on
// real matmuls (measured on an i9-11900KB: int8 LLM/TDNN prefill -4..-11%).
//
// We answer the question by **measuring** rather than by a CPUID model table.
// The table would be a proxy (model -> assumed port count) with two blind spots:
// it cannot tell a 1-FMA Bronze/Silver server SKU from its 2-FMA Gold/Platinum
// sibling (same DisplayModel), and every future dual-FMA part needs a code edit.
// A micro-probe instead measures the thing we actually optimise -- the achieved
// MAC/s of a 512-bit VPDPBUSD loop vs a 256-bit one on *this* core, right now,
// including the AVX-512 frequency-license downclock and any hypervisor 512-bit
// throttling. On a host that throttles 512-bit, 8x8 genuinely wins and the probe
// correctly says so, where a nameplate table would wrongly pick 16x16.
//
// Method (see `probe_fma_units`): time a tight, dependency-free VPDPBUSD loop at
// both widths and compare MAC/s. The only subtlety is the AVX-512 frequency
// license: after heavy 512-bit work the core stays downclocked for ~2 ms, so we
// measure **ymm first** (core still at its high frequency), then warm up and
// measure zmm at its own settled frequency. `mac_ratio = 2 * t_ymm / t_zmm`
// lands near 2.0 on a dual-FMA core and at/below 1.0 on a single-FMA one; we
// split at 1.5. Best-of-N (minimum elapsed = peak throughput) rejects scheduling
// noise, since interference only ever slows a sample. The whole probe is ~1-2 ms
// and runs at most once per process (memoised).
//
// `TRACT_AVX512_FMA_UNITS` (`1` or `2`) skips the probe entirely -- an escape
// hatch for locked-down hosts, reproducible benchmarking, or forcing a verdict.

use std::sync::OnceLock;
use std::time::Instant;

/// Number of 512-bit FMA/VNNI issue ports on this core (1 or 2). Memoised.
pub fn avx512_fma_units() -> u8 {
    static UNITS: OnceLock<u8> = OnceLock::new();
    *UNITS.get_or_init(detect_fma_units)
}

/// True iff the core has two 512-bit FMA ports, i.e. the zmm 16x16 VNNI kernel
/// can out-run the ymm 8x8 kernel. See module docs for the full rationale.
pub fn has_dual_avx512_fma() -> bool {
    avx512_fma_units() >= 2
}

fn detect_fma_units() -> u8 {
    if let Ok(v) = std::env::var("TRACT_AVX512_FMA_UNITS") {
        match v.trim() {
            "1" => return 1,
            "2" => return 2,
            other => log::warn!(
                "TRACT_AVX512_FMA_UNITS={other:?} not understood (expected 1 or 2); probing"
            ),
        }
    }
    // VPDPBUSD (and the zmm 16x16 kernel) require AVX-512-VNNI; without it the
    // wide kernel is never plugged, so the safe single-FMA answer is correct.
    if !std::is_x86_feature_detected!("avx512vnni") {
        return 1;
    }
    // Safety: AVX-512-VNNI (hence the AVX-512F/BW/VL the probe touches) is
    // present per the check above.
    let units = unsafe { probe_fma_units() };
    log::info!(
        "avx512_fma_units: probed {units} (512-bit FMA port{})",
        if units == 1 { "" } else { "s" }
    );
    units
}

/// One inner block = 16 independent VPDPBUSD on zmm0..zmm15 (sources zmm16/17),
/// repeated `iters` times. 16 independent accumulators fully hide VPDPBUSD's
/// ~5-cycle latency across both issue ports, so the loop is throughput-bound:
/// elapsed time reflects how many 512-bit VPDPBUSD/cycle this core can retire.
#[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
unsafe fn vnni_zmm_burst(iters: u64) {
    unsafe {
        std::arch::asm!(
            "vpxorq zmm0,  zmm0,  zmm0",  "vpxorq zmm1,  zmm1,  zmm1",
            "vpxorq zmm2,  zmm2,  zmm2",  "vpxorq zmm3,  zmm3,  zmm3",
            "vpxorq zmm4,  zmm4,  zmm4",  "vpxorq zmm5,  zmm5,  zmm5",
            "vpxorq zmm6,  zmm6,  zmm6",  "vpxorq zmm7,  zmm7,  zmm7",
            "vpxorq zmm8,  zmm8,  zmm8",  "vpxorq zmm9,  zmm9,  zmm9",
            "vpxorq zmm10, zmm10, zmm10", "vpxorq zmm11, zmm11, zmm11",
            "vpxorq zmm12, zmm12, zmm12", "vpxorq zmm13, zmm13, zmm13",
            "vpxorq zmm14, zmm14, zmm14", "vpxorq zmm15, zmm15, zmm15",
            "vpxorq zmm16, zmm16, zmm16", "vpxorq zmm17, zmm17, zmm17",
            "2:",
            "vpdpbusd zmm0,  zmm16, zmm17", "vpdpbusd zmm1,  zmm16, zmm17",
            "vpdpbusd zmm2,  zmm16, zmm17", "vpdpbusd zmm3,  zmm16, zmm17",
            "vpdpbusd zmm4,  zmm16, zmm17", "vpdpbusd zmm5,  zmm16, zmm17",
            "vpdpbusd zmm6,  zmm16, zmm17", "vpdpbusd zmm7,  zmm16, zmm17",
            "vpdpbusd zmm8,  zmm16, zmm17", "vpdpbusd zmm9,  zmm16, zmm17",
            "vpdpbusd zmm10, zmm16, zmm17", "vpdpbusd zmm11, zmm16, zmm17",
            "vpdpbusd zmm12, zmm16, zmm17", "vpdpbusd zmm13, zmm16, zmm17",
            "vpdpbusd zmm14, zmm16, zmm17", "vpdpbusd zmm15, zmm16, zmm17",
            "dec {n}",
            "jnz 2b",
            n = inout(reg) iters => _,
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
            out("zmm16") _, out("zmm17") _,
            options(nostack, nomem),
        );
    }
}

/// 256-bit twin of `vnni_zmm_burst`: same structure, ymm registers (EVEX so
/// ymm16/17 are addressable). Each VPDPBUSD does half the MACs of the zmm one.
#[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
unsafe fn vnni_ymm_burst(iters: u64) {
    unsafe {
        std::arch::asm!(
            "vpxord ymm0,  ymm0,  ymm0",  "vpxord ymm1,  ymm1,  ymm1",
            "vpxord ymm2,  ymm2,  ymm2",  "vpxord ymm3,  ymm3,  ymm3",
            "vpxord ymm4,  ymm4,  ymm4",  "vpxord ymm5,  ymm5,  ymm5",
            "vpxord ymm6,  ymm6,  ymm6",  "vpxord ymm7,  ymm7,  ymm7",
            "vpxord ymm8,  ymm8,  ymm8",  "vpxord ymm9,  ymm9,  ymm9",
            "vpxord ymm10, ymm10, ymm10", "vpxord ymm11, ymm11, ymm11",
            "vpxord ymm12, ymm12, ymm12", "vpxord ymm13, ymm13, ymm13",
            "vpxord ymm14, ymm14, ymm14", "vpxord ymm15, ymm15, ymm15",
            "vpxord ymm16, ymm16, ymm16", "vpxord ymm17, ymm17, ymm17",
            "2:",
            "vpdpbusd ymm0,  ymm16, ymm17", "vpdpbusd ymm1,  ymm16, ymm17",
            "vpdpbusd ymm2,  ymm16, ymm17", "vpdpbusd ymm3,  ymm16, ymm17",
            "vpdpbusd ymm4,  ymm16, ymm17", "vpdpbusd ymm5,  ymm16, ymm17",
            "vpdpbusd ymm6,  ymm16, ymm17", "vpdpbusd ymm7,  ymm16, ymm17",
            "vpdpbusd ymm8,  ymm16, ymm17", "vpdpbusd ymm9,  ymm16, ymm17",
            "vpdpbusd ymm10, ymm16, ymm17", "vpdpbusd ymm11, ymm16, ymm17",
            "vpdpbusd ymm12, ymm16, ymm17", "vpdpbusd ymm13, ymm16, ymm17",
            "vpdpbusd ymm14, ymm16, ymm17", "vpdpbusd ymm15, ymm16, ymm17",
            "dec {n}",
            "jnz 2b",
            n = inout(reg) iters => _,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
            out("ymm16") _, out("ymm17") _,
            options(nostack, nomem),
        );
    }
}

#[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
unsafe fn probe_fma_units() -> u8 {
    const TRIALS: u32 = 5;
    const WINDOW_NS: u128 = 150_000; // ~150 us per timed window
    const MAX_ITERS: u64 = 1 << 22;

    // Calibrate iteration count on the (cheaper) ymm path so each timed window
    // is long enough to dwarf Instant resolution but stays sub-millisecond.
    let mut iters: u64 = 1 << 10;
    loop {
        let t = Instant::now();
        unsafe { vnni_ymm_burst(iters) };
        if t.elapsed().as_nanos() >= WINDOW_NS || iters >= MAX_ITERS {
            break;
        }
        iters <<= 1;
    }

    let best = |zmm: bool| -> u128 {
        let mut best = u128::MAX;
        for _ in 0..TRIALS {
            let t = Instant::now();
            if zmm {
                unsafe { vnni_zmm_burst(iters) };
            } else {
                unsafe { vnni_ymm_burst(iters) };
            }
            best = best.min(t.elapsed().as_nanos().max(1));
        }
        best
    };

    // Measure ymm first, while the core is still at its high (light-license)
    // frequency; then warm up and measure zmm at its settled AVX-512 frequency.
    let t_ymm = best(false);
    unsafe { vnni_zmm_burst(iters) }; // freq-settle warmup, untimed
    let t_zmm = best(true);

    // Equal iteration counts, so zmm does 2x the MACs of ymm per op:
    //   mac_ratio = (16*64*iters / t_zmm) / (16*32*iters / t_ymm) = 2 * t_ymm / t_zmm
    // ~2.0 on a dual-512-FMA core, <=~1.0 on a single-FMA one. Split at 1.5.
    let mac_ratio = 2.0 * (t_ymm as f64) / (t_zmm as f64);
    log::debug!(
        "avx512_fma_units probe: iters={iters} t_ymm={t_ymm}ns t_zmm={t_zmm}ns mac_ratio={mac_ratio:.2}"
    );
    if mac_ratio >= 1.5 { 2 } else { 1 }
}
