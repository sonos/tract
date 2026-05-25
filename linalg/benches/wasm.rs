//! WASM kernel microbenches. Run on wasm32 only.
//!
//!   RUSTFLAGS='-C target-feature=+simd128' \
//!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
//!     cargo bench --release --target wasm32-wasip1 -p tract-linalg --bench wasm
//!
//! Re-run with `+simd128,+relaxed-simd` to compare baseline mul+add against
//! the FMA emit driven by the `madd_f32x4!` macro in `linalg/src/wasm.rs`.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("this bench only runs on wasm32 targets — skipping on host");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    let target = if cfg!(target_feature = "relaxed-simd") {
        "+simd128,+relaxed-simd (FMA)"
    } else {
        "+simd128 only (mul+add)"
    };

    eprintln!("=== WASM 8x8 GEMM microbench ({target}) ===");
    bench_8x8::run();

    eprintln!();
    eprintln!("=== Isolated 32x1 GEMV microbench ({target}) ===");
    bench_32x1::run();

    eprintln!();
    eprintln!("=== Isolated 16x1 GEMV microbench ({target}) ===");
    bench_16x1::run();

    eprintln!();
    eprintln!("=== int8 (i8->i32) 4x4 GEMM: wasm SIMD vs generic scalar ({target}) ===");
    bench_i8_4x4::run();

    #[cfg(target_feature = "relaxed-simd")]
    {
        eprintln!();
        eprintln!("=== int8 relaxed-dot prototype: relaxed_dot vs widening (4x4 tile) ===");
        bench_relaxed_dot::run();
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    eprintln!("\n(int8 relaxed-dot prototype skipped — rebuild with +relaxed-simd)");
}

#[cfg(target_arch = "wasm32")]
mod bench_8x8 {
    //! Microbench: time `wasm_f32_8x8` (the GEMM kernel for N>=2) at shapes
    //! relevant to DFN3, transformer FFN, and CNN→GEMM workloads.

    use std::time::Instant;
    use tract_data::internal::*;
    use tract_linalg::mmm::{AsInputValue, FusedSpec};

    fn run_one(
        kernel: &dyn tract_linalg::mmm::MatMatMul,
        m: usize,
        k: usize,
        n: usize,
        iters: usize,
    ) -> f64 {
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, n]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, n]).unwrap();

        for _ in 0..50 {
            unsafe {
                kernel
                    .run(
                        m,
                        n,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        n,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn pick(name: &str) -> Box<dyn tract_linalg::mmm::MatMatMul> {
        let mut ops = tract_linalg::generic();
        tract_linalg::wasm::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_shape(label: &str, m: usize, k: usize, n: usize, iters: usize) {
        let k88 = pick("wasm_f32_8x8");
        let ns = run_one(&*k88, m, k, n, iters);
        let m_tiles = m.div_ceil(8);
        let n_tiles = n.div_ceil(8);
        let total_tiles = m_tiles * n_tiles;
        let per_tile_ns = ns / total_tiles as f64;
        eprintln!(
            "{label} (m={m} k={k} n={n}, iters={iters}): {ns:.0} ns/call \
             ({total_tiles} 8x8 tiles, {per_tile_ns:.1} ns/tile)"
        );
    }

    pub fn run() {
        // DFN3 N>1 GEMM case (the primary 8x8 hit on DFN3).
        bench_shape("DFN3-style m=64 k=64 n=8", 64, 64, 8, 50_000);
        // Larger N — typical batched/transformer GEMM.
        bench_shape("m=64 k=64 n=64", 64, 64, 64, 10_000);
        bench_shape("m=128 k=128 n=8", 128, 128, 8, 20_000);
        bench_shape("m=128 k=128 n=64", 128, 128, 64, 5_000);
        bench_shape("m=256 k=256 n=8", 256, 256, 8, 5_000);
        bench_shape("m=256 k=256 n=64", 256, 256, 64, 1_000);
        // Whisper-tiny FFN-ish (large K, small N).
        bench_shape("m=384 k=1536 n=8", 384, 1536, 8, 1_000);
    }
}

#[cfg(target_arch = "wasm32")]
mod bench_32x1 {
    //! Isolated, statistics-aware microbench for `wasm_f32_32x1` to investigate
    //! the apparent regression at M=100/256 in `microbench_dispatch_gemv`. That
    //! bench loops all 4 GEMV kernels back-to-back at every shape, biasing the
    //! later-running kernel (32x1) with cache contention and thermal buildup.
    //! This module benches 32x1 alone, with min-of-N reporting across
    //! repetitions to expose variance honestly.

    use std::time::Instant;
    use tract_data::internal::*;
    use tract_linalg::mmm::{AsInputValue, FusedSpec};

    fn run_one(kernel: &dyn tract_linalg::mmm::MatMatMul, m: usize, k: usize, iters: usize) -> f64 {
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();

        // Generous warmup — 200 calls primes the JIT and hot caches.
        for _ in 0..200 {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn pick(name: &str) -> Box<dyn tract_linalg::mmm::MatMatMul> {
        let mut ops = tract_linalg::generic();
        tract_linalg::wasm::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_min_of_n(label: &str, m: usize, k: usize, iters: usize, repetitions: usize) {
        let kernel = pick("wasm_f32_32x1");
        let mut samples: Vec<f64> = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            samples.push(run_one(&*kernel, m, k, iters));
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = samples[0];
        let median = samples[samples.len() / 2];
        let max = samples[samples.len() - 1];
        let pct_spread = (max - min) / min * 100.0;
        eprintln!(
            "{label} (m={m} k={k}, {iters} iters × {repetitions} reps): \
             min={min:.0} median={median:.0} max={max:.0} ns/call (spread {pct_spread:.0}%)"
        );
    }

    pub fn run() {
        // Suspect shapes from microbench_dispatch_gemv (apparent regression):
        bench_min_of_n("M=100 k=256", 100, 256, 10_000, 10);
        bench_min_of_n("M=256 k=256", 256, 256, 5_000, 10);
        bench_min_of_n("M=256 k=512", 256, 512, 2_000, 10);
        // Reference shapes (showed clean speedup before):
        bench_min_of_n("M=24 k=256", 24, 256, 30_000, 10);
        bench_min_of_n("M=64 k=96", 64, 96, 20_000, 10);
    }
}

#[cfg(target_arch = "wasm32")]
mod bench_16x1 {
    //! Isolated 16x1 GEMV microbench — same methodology as bench_32x1.
    //! 16x1 has 4 SIMD accumulators per K-step, which under +relaxed-simd
    //! exposes the destructive-fmla accumulator recurrence (4-cycle latency
    //! throttling throughput to 1 FMA/cycle even though Apple Silicon pipes
    //! can do 4). Used to validate that the fix in linalg/src/wasm.rs (which
    //! routes 16x1 through `madd_f32x4_nofma!` to use separate mul+add)
    //! recovers the regression PR #2199 missed.
    use std::time::Instant;
    use tract_data::internal::*;
    use tract_linalg::mmm::{AsInputValue, FusedSpec};

    fn run_one(kernel: &dyn tract_linalg::mmm::MatMatMul, m: usize, k: usize, iters: usize) -> f64 {
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();

        for _ in 0..200 {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn pick(name: &str) -> Box<dyn tract_linalg::mmm::MatMatMul> {
        let mut ops = tract_linalg::generic();
        tract_linalg::wasm::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_min_of_n(label: &str, m: usize, k: usize, iters: usize, repetitions: usize) {
        let kernel = pick("wasm_f32_16x1");
        let mut samples: Vec<f64> = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            samples.push(run_one(&*kernel, m, k, iters));
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = samples[0];
        let median = samples[samples.len() / 2];
        let max = samples[samples.len() - 1];
        let pct_spread = (max - min) / min * 100.0;
        eprintln!(
            "{label} (m={m} k={k}, {iters} iters × {repetitions} reps): \
             min={min:.0} median={median:.0} max={max:.0} ns/call (spread {pct_spread:.0}%)"
        );
    }

    pub fn run() {
        // 16x1's natural band per plug()'s mmv_f32 closure: M ∈ 9..=16
        bench_min_of_n("M=9 k=256", 9, 256, 30_000, 10);
        bench_min_of_n("M=12 k=256", 12, 256, 30_000, 10);
        bench_min_of_n("M=16 k=96", 16, 96, 30_000, 10);
        bench_min_of_n("M=16 k=256", 16, 256, 20_000, 10);
        bench_min_of_n("M=16 k=512", 16, 512, 10_000, 10);
        bench_min_of_n("M=16 k=1024", 16, 1024, 5_000, 10);
    }
}

#[cfg(target_arch = "wasm32")]
mod bench_i8_4x4 {
    //! int8 (i8->i32) GEMM microbench: the new SIMD `wasm_i32_4x4` vs the scalar
    //! `generic_i32_4x4` fallback. Both kernels expose the *identical* i8i8
    //! PackedI8K4 packing (packing index 1), the same 4x4 tile and i32
    //! accumulator — so the ratio is a clean read on what the SIMD
    //! widening-extmul AddMatMul buys over the generic scalar loop. min-of-N
    //! reporting per kernel to keep the variance honest.

    use std::time::Instant;
    use tract_data::internal::*;
    use tract_linalg::mmm::{AsInputValue, FusedSpec, MatMatMul};

    // i8i8 packing slot is index 1 on both generic_i32_4x4 and wasm_i32_4x4.
    const I8I8: usize = 1;

    fn run_one(kernel: &dyn MatMatMul, m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let packing = &kernel.packings()[I8I8];
        let a = Tensor::zero::<i8>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<i8>(&[k, n]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<i32>(&[m, n]).unwrap();

        // Warmup: prime the JIT and hot caches.
        for _ in 0..50 {
            unsafe {
                kernel
                    .run(
                        m,
                        n,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: I8I8,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        n,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: I8I8,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn pick(name: &str) -> Box<dyn MatMatMul> {
        let mut ops = tract_linalg::generic();
        tract_linalg::wasm::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn min_of_n(
        kernel: &dyn MatMatMul,
        m: usize,
        k: usize,
        n: usize,
        iters: usize,
        reps: usize,
    ) -> f64 {
        let mut samples: Vec<f64> = (0..reps).map(|_| run_one(kernel, m, k, n, iters)).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[0]
    }

    fn bench(label: &str, m: usize, k: usize, n: usize, iters: usize, reps: usize) {
        let wasm = pick("wasm_i32_4x4");
        let generic = pick("generic_i32_4x4");
        let w = min_of_n(&*wasm, m, k, n, iters, reps);
        let g = min_of_n(&*generic, m, k, n, iters, reps);
        let tiles = m.div_ceil(4) * n.div_ceil(4);
        eprintln!(
            "{label} (m={m} k={k} n={n}, {iters} iters × {reps} reps): \
             wasm={w:.0} generic={g:.0} ns/call  speedup={:.2}x  \
             ({tiles} 4x4 tiles, wasm {:.1} ns/tile)",
            g / w,
            w / tiles as f64
        );
    }

    pub fn run() {
        // Square GEMMs across sizes (compute-bound, the SIMD path's home turf).
        bench("square m=64 k=64 n=64", 64, 64, 64, 5_000, 8);
        bench("square m=128 k=128 n=128", 128, 128, 128, 1_000, 8);
        bench("square m=256 k=256 n=256", 256, 256, 256, 200, 8);
        // Transformer-ish: large K, moderate M/N (MiniLM/FFN projections).
        bench("m=128 k=384 n=384", 128, 384, 384, 500, 8);
        bench("m=64 k=1536 n=64", 64, 1536, 64, 1_000, 8);
        // CNN→GEMM (InceptionV1-style im2col), small N.
        bench("m=256 k=256 n=16", 256, 256, 16, 2_000, 8);
    }
}

// Prototype: int8 4x4 tile via `i32x4_relaxed_dot_i8x16_i7x16_add` (SDOT-analog,
// 4 i8 MACs/lane, no widening) vs the deterministic widening path. Only compiles
// under +relaxed-simd. Isolates a single cache-resident 4x4 tile so the ratio is
// a pure instruction-density read. Includes a bit-exactness check on wasmtime.
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
mod bench_relaxed_dot {
    use std::arch::wasm32::*;
    use std::hint::black_box;
    use std::time::Instant;

    // Logical A is [4][k] row-major (a[m*k + ik]); logical B is [k][4] row-major
    // (b[ik*4 + n]). Reference 4x4 = sum_ik A[m][ik] * B[ik][n].
    fn reference_tile(a: &[i8], b: &[i8], k: usize) -> [i32; 16] {
        let mut c = [0i32; 16];
        for ik in 0..k {
            for m in 0..4 {
                for n in 0..4 {
                    c[m * 4 + n] += a[m * k + ik] as i32 * b[ik * 4 + n] as i32;
                }
            }
        }
        c
    }

    // K-major A: out[ik*4 + m] = A[m][ik] (m inner) — what the widening kernel reads.
    fn pack_a_kmajor(a: &[i8], k: usize) -> Vec<i8> {
        let mut o = vec![0i8; k * 4];
        for ik in 0..k {
            for m in 0..4 {
                o[ik * 4 + m] = a[m * k + ik];
            }
        }
        o
    }
    // K-major B is exactly the logical [ik*4 + n] layout already.

    // M-major A, K contiguous, K padded to mult of 4: out[m*kp + ik] = A[m][ik].
    fn pack_a_mmajor(a: &[i8], k: usize) -> (Vec<i8>, usize) {
        let kp = k.div_ceil(4) * 4;
        let mut o = vec![0i8; 4 * kp];
        for m in 0..4 {
            for ik in 0..k {
                o[m * kp + ik] = a[m * k + ik];
            }
        }
        (o, kp)
    }
    // K=4-inner B: out[kb*16 + n*4 + kr] = B[4kb+kr][n] — the relaxed-dot layout.
    fn pack_b_k4(b: &[i8], k: usize) -> Vec<i8> {
        let kp = k.div_ceil(4) * 4;
        let mut o = vec![0i8; kp * 4];
        for kb in 0..kp / 4 {
            for kr in 0..4 {
                let kk = 4 * kb + kr;
                if kk >= k {
                    continue;
                }
                for n in 0..4 {
                    o[kb * 16 + n * 4 + kr] = b[kk * 4 + n];
                }
            }
        }
        o
    }

    // Current deterministic approach: widen B to i32x4 per k, splat A, mul+add.
    unsafe fn widening_tile(a_km: *const i8, b_km: *const i8, k: usize) -> [i32; 16] {
        unsafe {
            let mut acc = [i32x4_splat(0); 4];
            for ik in 0..k {
                let bw = v128_load32_zero(b_km.add(4 * ik) as *const u32);
                let bw = i16x8_extend_low_i8x16(bw);
                let bw = i32x4_extend_low_i16x8(bw);
                let ar = a_km.add(4 * ik);
                acc[0] = i32x4_add(acc[0], i32x4_mul(i32x4_splat(*ar.add(0) as i32), bw));
                acc[1] = i32x4_add(acc[1], i32x4_mul(i32x4_splat(*ar.add(1) as i32), bw));
                acc[2] = i32x4_add(acc[2], i32x4_mul(i32x4_splat(*ar.add(2) as i32), bw));
                acc[3] = i32x4_add(acc[3], i32x4_mul(i32x4_splat(*ar.add(3) as i32), bw));
            }
            let mut c = [0i32; 16];
            for m in 0..4 {
                v128_store(c[m * 4..].as_mut_ptr() as *mut v128, acc[m]);
            }
            c
        }
    }

    // Relaxed-dot: per 4-K block, one v128 B-load shared across 4 rows; each row
    // broadcasts its 4 K-bytes and issues one relaxed_dot. 64 MACs in 4 dots.
    unsafe fn relaxed_tile(apk: *const i8, bpk: *const i8, kp: usize) -> [i32; 16] {
        unsafe {
            let mut acc = [i32x4_splat(0); 4];
            for kb in 0..kp / 4 {
                let b_all = v128_load(bpk.add(kb * 16) as *const v128);
                for m in 0..4 {
                    let a4 = (apk.add(m * kp + kb * 4) as *const i32).read_unaligned();
                    let a_m = i32x4_splat(a4);
                    acc[m] = i32x4_relaxed_dot_i8x16_i7x16_add(a_m, b_all, acc[m]);
                }
            }
            let mut c = [0i32; 16];
            for m in 0..4 {
                v128_store(c[m * 4..].as_mut_ptr() as *mut v128, acc[m]);
            }
            c
        }
    }

    fn gen_data(k: usize, seed: i32, bits7: bool) -> Vec<i8> {
        (0..k * 4)
            .map(|i| {
                let v = ((i as i32).wrapping_mul(97).wrapping_add(seed).wrapping_mul(31)) & 0xff;
                let v = (v - 128) as i8; // full i8 range
                if bits7 { (v as i32).clamp(-63, 63) as i8 } else { v }
            })
            .collect()
    }

    fn check(label: &str, k: usize, b_bits7: bool) {
        let a = gen_data(k, 1, false);
        let b = gen_data(k, 7, b_bits7);
        let reference = reference_tile(&a, &b, k);

        let a_km = pack_a_kmajor(&a, k);
        let w = unsafe { widening_tile(a_km.as_ptr(), b.as_ptr(), k) };
        assert_eq!(w, reference, "widening_tile mismatch ({label})");

        let (a_mm, kp) = pack_a_mmajor(&a, k);
        let b_k4 = pack_b_k4(&b, k);
        let r = unsafe { relaxed_tile(a_mm.as_ptr(), b_k4.as_ptr(), kp) };
        let exact = r == reference;
        eprintln!(
            "  correctness {label} (k={k}, B={}): widening=exact  relaxed={}",
            if b_bits7 { "7-bit" } else { "full-i8" },
            if exact { "EXACT" } else { "DIFFERS (non-deterministic intermediate)" }
        );
        if b_bits7 {
            assert!(exact, "relaxed_dot must be exact when B is 7-bit ({label})");
        }
    }

    fn time_relaxed(apk: &[i8], bpk: &[i8], kp: usize, iters: usize) -> f64 {
        let mut sink = 0i32;
        for _ in 0..50 {
            sink ^= unsafe { relaxed_tile(apk.as_ptr(), bpk.as_ptr(), kp) }[0];
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let c = unsafe {
                relaxed_tile(black_box(apk).as_ptr(), black_box(bpk).as_ptr(), black_box(kp))
            };
            sink ^= c[5];
        }
        black_box(sink);
        t0.elapsed().as_secs_f64() / iters as f64 * 1e9
    }

    fn time_widening(a_km: &[i8], b_km: &[i8], k: usize, iters: usize) -> f64 {
        let mut sink = 0i32;
        for _ in 0..50 {
            sink ^= unsafe { widening_tile(a_km.as_ptr(), b_km.as_ptr(), k) }[0];
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let c = unsafe {
                widening_tile(black_box(a_km).as_ptr(), black_box(b_km).as_ptr(), black_box(k))
            };
            sink ^= c[5];
        }
        black_box(sink);
        t0.elapsed().as_secs_f64() / iters as f64 * 1e9
    }

    fn min_of_n(f: &mut dyn FnMut() -> f64, reps: usize) -> f64 {
        let mut s: Vec<f64> = (0..reps).map(|_| f()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[0]
    }

    fn bench(k: usize, iters: usize, reps: usize) {
        let a = gen_data(k, 1, false);
        let b = gen_data(k, 7, false);
        let a_km = pack_a_kmajor(&a, k);
        let (a_mm, kp) = pack_a_mmajor(&a, k);
        let b_k4 = pack_b_k4(&b, k);

        let w = min_of_n(&mut || time_widening(&a_km, &b, k, iters), reps);
        let r = min_of_n(&mut || time_relaxed(&a_mm, &b_k4, kp, iters), reps);
        eprintln!(
            "  4x4 tile k={k} ({iters} iters × {reps} reps): \
             widening={w:.1} relaxed={r:.1} ns/call  speedup={:.2}x",
            w / r
        );
    }

    pub fn run() {
        // Bit-exactness on wasmtime: full-i8 (engine-dependent intermediate) and
        // 7-bit B (guaranteed no i16 overflow → deterministic on any engine).
        check("k=64", 64, false);
        check("k=64", 64, true);
        check("k=260-padded", 260, false);
        check("k=260-padded", 260, true);
        eprintln!();
        // Throughput: single cache-resident 4x4 tile across K depths.
        bench(64, 200_000, 8);
        bench(256, 50_000, 8);
        bench(1024, 10_000, 8);
        bench(1536, 8_000, 8);
    }
}
