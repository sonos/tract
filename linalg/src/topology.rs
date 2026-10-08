//! How many workers a matmul may put on one execution unit.
//!
//! The pool fills the first free pipe of the unit the kernel uses, then the next
//! pipe, and stops when that unit is full. A second worker does not stack onto a
//! pipe that already has one, and a matrix kernel does not borrow the SIMD cores.
//!
//! Apple silicon reads `hw.perflevel0`. SIMD pipes are the performance cores;
//! NEON and SME kernels take one worker each. AMX is one matrix unit shared per
//! L2 cluster, `physicalcpu / cpusperl2`, so an AMX kernel that fills its pipe
//! stops at that count. A missing `cpusperl2` is one matrix pipe, not a
//! guessed Max or Ultra. Efficiency cores live in `perflevel1` and stay out of
//! both counts. An M1 and an M4, a Max, and an Ultra are the same division; the
//! coefficient table stays keyed by chip generation.
//!
//! x86 reads the physical core count. Intel AMX is one TMUL per physical core,
//! shared by SMT siblings, so the matrix count is that core count when AMX-INT8
//! or AMX-BF16 is usable, and zero otherwise. A zero matrix count uses the SIMD
//! cores rather than refusing the launch. No probe leaves the pool uncapped.
//! Windows falls back to `available_parallelism`, which counts logical processors.
//!
//! The shape-aware Apple worker count lives next to that cost model. This module
//! is the geometry it reads. Callers are split by target, so unused items are expected.

#![cfg_attr(not(all(target_os = "macos", feature = "multithread-mm")), allow(dead_code))]

use std::io::Write;

use crate::isa::{Isa, IsaReq};

/// Geometry of `hw.perflevel0`, the performance cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerfCluster {
    /// `hw.perflevel0.physicalcpu`.
    pub physical_cpus: usize,
    /// `hw.perflevel0.cpusperl2`, when the kernel exposes it.
    pub cpus_per_l2: Option<usize>,
}

impl PerfCluster {
    /// Matrix pipes in this performance level: one per L2 cluster.
    ///
    /// `physical_cpus / cpus_per_l2`, and at least one. A missing or zero sharing
    /// degree is one pipe — the base-chip fallback — rather than an invented
    /// cluster count.
    pub fn matrix_units(self) -> usize {
        match self.cpus_per_l2 {
            Some(per) if per > 0 => (self.physical_cpus / per).max(1),
            _ => 1,
        }
    }
}

/// Pipes a matmul can fill, independent of who built the chip.
///
/// `simd` is one pipe per core that runs a vector kernel: Apple performance
/// cores, or x86 physical cores. `matrix` is one pipe per matrix unit. On Apple
/// that unit is the AMX pipe, shared by an L2 cluster, so several cores count
/// as one pipe; SME kernels take the `simd` roof instead. On Intel AMX each
/// physical core has its own TMUL, so `matrix` equals `simd`.
/// Zero means the machine has no matrix unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputePipes {
    pub simd: usize,
    pub matrix: usize,
}

impl ComputePipes {
    /// Apple performance cluster, as pipes. The matrix count is one AMX pipe per
    /// L2 cluster.
    pub fn from_perf_cluster(cluster: PerfCluster) -> Self {
        ComputePipes { simd: cluster.physical_cpus.max(1), matrix: cluster.matrix_units() }
    }

    /// Workers of this kind a pool of `pool` may run.
    ///
    /// One worker per pipe, in order, and no more. A matrix kernel with no
    /// matrix unit takes the SIMD cores: the launch must not see a roof of zero.
    pub fn worker_roof(self, pool: usize, matrix: bool) -> usize {
        let pool = pool.max(1);
        let units = if matrix { self.matrix } else { self.simd };
        if units == 0 {
            return pool.min(self.simd.max(1));
        }
        pool.min(units).max(1)
    }
}

/// Threads a per-core kernel and a per-cluster matrix kernel are scored with.
///
/// The dispatch cap is the same pair with the pool taken out: a kernel may not
/// run on more threads than the divisor it was chosen with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(all(target_os = "macos", any(target_arch = "aarch64", feature = "foreign-inventory"))),
    allow(dead_code)
)]
pub struct DispatchThreads {
    /// One worker per performance core: NEON and SME kernels.
    pub neon: usize,
    /// One worker per L2-cluster matrix pipe: an AMX kernel that fills its pipe.
    pub matrix: usize,
}

/// Thread counts for a pool of `pool` threads on `cluster`.
///
/// Both are clamped to the performance cluster. Threads past that are efficiency
/// cores and do not speed either kernel up.
#[cfg_attr(
    not(all(target_os = "macos", any(target_arch = "aarch64", feature = "foreign-inventory"))),
    allow(dead_code)
)]
pub fn dispatch_threads(pool: usize, cluster: PerfCluster) -> DispatchThreads {
    let pipes = ComputePipes::from_perf_cluster(cluster);
    DispatchThreads { neon: pipes.worker_roof(pool, false), matrix: pipes.worker_roof(pool, true) }
}

/// True for Apple's matrix ISAs: SME, SME2, and AMX. This is the "is an Apple
/// matrix kernel" check; of the three only AMX takes the per-cluster pipe roof
/// ([`uses_amx_cluster_pipe`]). x86 AMX is a different unit and is not included.
pub fn uses_apple_matrix_pipe(isa: IsaReq) -> bool {
    isa.needs.iter().any(|i| matches!(i, Isa::Aarch64Sme | Isa::Aarch64Sme2 | Isa::Aarch64AppleAmx))
}

/// Apple AMX is one matrix unit shared per L2 cluster; SME issues per
/// performance core, so it takes the SIMD roof instead.
pub fn uses_amx_cluster_pipe(isa: IsaReq) -> bool {
    isa.needs.iter().any(|i| matches!(i, Isa::Aarch64AppleAmx))
}

/// A kernel that issues to a matrix unit.
///
/// On Apple that is SME, SME2, or AMX; only AMX shares one unit per L2 cluster.
/// Intel AMX is one TMUL per physical core. [`uses_apple_matrix_pipe`] stays
/// the Apple-only check.
#[cfg_attr(all(target_os = "macos", feature = "multithread-mm"), allow(dead_code))]
pub fn uses_matrix_unit(isa: IsaReq) -> bool {
    uses_apple_matrix_pipe(isa)
        || isa.needs.iter().any(|i| matches!(i, Isa::X86_64AmxInt8 | Isa::X86_64AmxBf16))
}

/// How many of this kernel's chunks the Apple cluster geometry allows.
///
/// AMX is capped at one worker per L2-cluster pipe; every other kernel, NEON
/// and SME included, at the performance-core count. `usize::MAX` when no
/// cluster was passed in. The launch path uses [`hardware_workers`] on a
/// machine without `perflevel0`, and the Apple cost model on macOS, so this
/// roof is unused in those builds.
#[cfg_attr(all(target_os = "macos", feature = "multithread-mm"), allow(dead_code))]
pub fn concurrency_cap(cluster: Option<PerfCluster>, isa: IsaReq) -> usize {
    let Some(cluster) = cluster else {
        return usize::MAX;
    };
    if uses_amx_cluster_pipe(isa) { cluster.matrix_units() } else { cluster.physical_cpus.max(1) }
}

/// Append one line per matmul chunk when `TRACT_SIMD_TRACE` is a file path.
///
/// The line is `thread path kernel`. `thread` is the rayon worker index, or
/// `caller` when the chunk runs on the thread that entered the operator. One
/// chunk is one path: a pipe-filling matrix kernel stays on `caller`, and a
/// short-K one can show up on workers.
pub fn trace_simd(kernel: &str, isa: IsaReq) {
    let Some(path) = simd_trace_log() else { return };
    let thread = simd_trace_thread();
    let kind = simd_kind(isa);
    let mut file = path.lock().unwrap();
    let _ = writeln!(file, "{thread}\t{kind}\t{kernel}");
}

fn simd_kind(isa: IsaReq) -> &'static str {
    let sme = isa.needs.iter().any(|i| matches!(i, Isa::Aarch64Sme | Isa::Aarch64Sme2));
    let amx = isa.needs.iter().any(|i| matches!(i, Isa::Aarch64AppleAmx));
    if sme {
        "sme"
    } else if amx {
        "amx"
    } else {
        "neon"
    }
}

fn simd_trace_thread() -> String {
    #[cfg(feature = "multithread-mm")]
    if let Some(index) = rayon::current_thread_index() {
        return index.to_string();
    }
    "caller".to_string()
}

fn simd_trace_log() -> Option<&'static std::sync::Mutex<std::fs::File>> {
    use std::sync::{Mutex, OnceLock};
    static LOG: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let path = std::env::var_os("TRACT_SIMD_TRACE")?;
        std::fs::File::create(&path).ok().map(Mutex::new)
    })
    .as_ref()
}

/// This machine's pipes, or `None` when the probe finds nothing.
///
/// Apple `perflevel0` wins when it is present, including under translation.
/// Otherwise an x86_64 build reads physical cores and, when the process can
/// use Intel AMX, one matrix pipe per core. Anywhere else there is no roof.
#[cfg_attr(all(target_os = "macos", feature = "multithread-mm"), allow(dead_code))]
pub fn compute_pipes() -> Option<ComputePipes> {
    if let Some(cluster) = perf_cluster() {
        return Some(ComputePipes::from_perf_cluster(cluster));
    }
    #[cfg(target_arch = "x86_64")]
    {
        return x86_pipes();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// Workers for a kernel the cost model is not scoring.
///
/// A known machine clamps `pool` to [`ComputePipes::worker_roof`]. No probe
/// returns [`usize::MAX`], the uncapped pool.
#[cfg_attr(all(target_os = "macos", feature = "multithread-mm"), allow(dead_code))]
pub fn hardware_workers(pool: usize, matrix: bool) -> usize {
    match compute_pipes() {
        Some(pipes) => pipes.worker_roof(pool, matrix),
        None => usize::MAX,
    }
}

/// Physical cores on x86, and one matrix pipe per core when Intel AMX is usable.
///
/// SMT siblings share a TMUL, so the count is physical cores, not logical
/// processors. Linux divides the online list by cpu0's sibling list. macOS
/// reads `hw.physicalcpu`. Any other OS uses `available_parallelism`, which
/// includes SMT. AMX permission is the same gate the kernels use, so a CPUID
/// bit without an OS grant does not invent pipes.
#[cfg(target_arch = "x86_64")]
fn x86_pipes() -> Option<ComputePipes> {
    let simd = physical_cores()?;
    let matrix = if crate::x86_64::amx::has_amx_int8() || crate::x86_64::amx_bf16::has_amx_bf16() {
        simd
    } else {
        0
    };
    Some(ComputePipes { simd, matrix })
}

#[cfg(target_arch = "x86_64")]
fn physical_cores() -> Option<usize> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        crate::cache::sysctl_usize("hw.physicalcpu")
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let online = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
        let logical = crate::cache::count_cpu_list(&online);
        if logical == 0 {
            return None;
        }
        let smt =
            std::fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
                .ok()
                .map(|s| crate::cache::count_cpu_list(&s))
                .filter(|&n| n > 0)
                .unwrap_or(1);
        Some((logical / smt).max(1))
    }
    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android"
    )))]
    {
        std::thread::available_parallelism().ok().map(|n| n.get())
    }
}

/// This machine's performance cluster, or `None` when `perflevel0` is absent.
///
/// Memoised for the process. The matmul launch reads this on every product, and
/// a fresh sysctl is slower than a short GEMV.
pub fn perf_cluster() -> Option<PerfCluster> {
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        return None;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use std::sync::OnceLock;
        static CLUSTER: OnceLock<Option<PerfCluster>> = OnceLock::new();
        *CLUSTER.get_or_init(|| {
            let physical_cpus = crate::cache::sysctl_usize("hw.perflevel0.physicalcpu")?;
            let cpus_per_l2 = crate::cache::sysctl_usize("hw.perflevel0.cpusperl2");
            Some(PerfCluster { physical_cpus, cpus_per_l2 })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sme() -> IsaReq {
        IsaReq::ANY.needing(&[Isa::Aarch64Sme])
    }

    fn sme2() -> IsaReq {
        IsaReq::ANY.needing(&[Isa::Aarch64Sme2])
    }

    fn apple_amx() -> IsaReq {
        IsaReq::ANY.needing(&[Isa::Aarch64AppleAmx])
    }

    #[test]
    fn budgets_follow_cluster_geometry() {
        // Base M4: 4 performance cores sharing one L2, so one pipe. A pool the
        // size of hw.ncpu (4+6) still scores NEON at 4 and the matrix pipe at 1.
        let base_m4 = PerfCluster { physical_cpus: 4, cpus_per_l2: Some(4) };
        assert_eq!(base_m4.matrix_units(), 1);
        assert_eq!(dispatch_threads(10, base_m4), DispatchThreads { neon: 4, matrix: 1 });
        assert_eq!(dispatch_threads(4, base_m4), DispatchThreads { neon: 4, matrix: 1 });
        assert_eq!(dispatch_threads(2, base_m4), DispatchThreads { neon: 2, matrix: 1 });
        assert_eq!(dispatch_threads(1, base_m4), DispatchThreads { neon: 1, matrix: 1 });

        // 12 performance cores, 4 per L2: three pipes. The pool past the
        // performance cluster stops at 12 and 3.
        let wide = PerfCluster { physical_cpus: 12, cpus_per_l2: Some(4) };
        assert_eq!(wide.matrix_units(), 3);
        assert_eq!(dispatch_threads(16, wide), DispatchThreads { neon: 12, matrix: 3 });
        assert_eq!(dispatch_threads(8, wide), DispatchThreads { neon: 8, matrix: 3 });
        assert_eq!(dispatch_threads(2, wide), DispatchThreads { neon: 2, matrix: 2 });

        // Two clusters of 4, the M1 Pro / M1 Max shape.
        let two_clusters = PerfCluster { physical_cpus: 8, cpus_per_l2: Some(4) };
        assert_eq!(dispatch_threads(10, two_clusters), DispatchThreads { neon: 8, matrix: 2 });

        // Sharing degree not exposed: one pipe, and NEON still uses the P-core count.
        let unknown = PerfCluster { physical_cpus: 8, cpus_per_l2: None };
        assert_eq!(unknown.matrix_units(), 1);
        assert_eq!(dispatch_threads(10, unknown), DispatchThreads { neon: 8, matrix: 1 });
        let zero = PerfCluster { physical_cpus: 8, cpus_per_l2: Some(0) };
        assert_eq!(zero.matrix_units(), 1);
    }

    #[test]
    fn amx_keeps_the_cluster_cap_and_a_missing_topology_caps_nothing() {
        let m4 = Some(PerfCluster { physical_cpus: 4, cpus_per_l2: Some(4) });
        assert_eq!(concurrency_cap(m4, sme()), 4);
        assert_eq!(concurrency_cap(m4, sme2()), 4);
        assert_eq!(concurrency_cap(m4, apple_amx()), 1);
        assert_eq!(concurrency_cap(m4, IsaReq::ANY), 4);
        // fp16 / dotprod NEON scales with the performance cores, not the pipe.
        assert_eq!(concurrency_cap(m4, IsaReq::ANY.needing(&[Isa::Aarch64Fp16])), 4);

        let wide = Some(PerfCluster { physical_cpus: 12, cpus_per_l2: Some(4) });
        assert_eq!(concurrency_cap(wide, sme()), 12);
        assert_eq!(concurrency_cap(wide, apple_amx()), 3);
        assert_eq!(concurrency_cap(wide, IsaReq::ANY), 12);

        // x86 AMX is not this pipe: on a machine with no perflevel it is uncapped,
        // and it is not grouped with SME when a cluster is injected.
        let x86_amx = IsaReq::ANY.needing(&[Isa::X86_64AmxInt8]);
        let x86_bf16 = IsaReq::ANY.needing(&[Isa::X86_64AmxBf16]);
        assert!(!uses_apple_matrix_pipe(x86_amx));
        assert!(!uses_apple_matrix_pipe(x86_bf16));
        assert!(uses_matrix_unit(x86_amx));
        assert!(uses_matrix_unit(x86_bf16));
        assert!(uses_matrix_unit(sme()));
        assert!(!uses_matrix_unit(IsaReq::ANY));
        assert!(uses_apple_matrix_pipe(sme()));
        assert!(uses_apple_matrix_pipe(sme2()));
        assert!(uses_apple_matrix_pipe(apple_amx()));
        assert!(uses_amx_cluster_pipe(apple_amx()));
        assert!(!uses_amx_cluster_pipe(sme()));
        assert!(!uses_amx_cluster_pipe(sme2()));
        assert!(!uses_amx_cluster_pipe(x86_amx));
        assert_eq!(concurrency_cap(None, sme()), usize::MAX);
        assert_eq!(concurrency_cap(None, x86_amx), usize::MAX);
    }

    /// One worker per pipe, then the next pipe. The numbers are injected: an
    /// Apple cluster, an Intel part with one TMUL per core, and a part with no
    /// matrix unit at all.
    #[test]
    fn pipes_fill_one_worker_then_the_next_unit() {
        let base = ComputePipes { simd: 4, matrix: 1 };
        assert_eq!(base.worker_roof(8, true), 1);
        assert_eq!(base.worker_roof(8, false), 4);
        assert_eq!(base.worker_roof(2, false), 2);
        assert_eq!(base.worker_roof(1, true), 1);

        // M1 Pro / Max: eight performance cores, two matrix pipes.
        let pro = ComputePipes { simd: 8, matrix: 2 };
        assert_eq!(pro.worker_roof(16, true), 2);
        assert_eq!(pro.worker_roof(16, false), 8);
        assert_eq!(pro.worker_roof(1, true), 1);
        // M1 Ultra: sixteen performance cores, four matrix pipes.
        let ultra = ComputePipes::from_perf_cluster(PerfCluster {
            physical_cpus: 16,
            cpus_per_l2: Some(4),
        });
        assert_eq!(ultra, ComputePipes { simd: 16, matrix: 4 });
        assert_eq!(ultra.worker_roof(32, true), 4);
        assert_eq!(ultra.worker_roof(3, true), 3);

        // Intel AMX: one TMUL per physical core, shared by its SMT siblings.
        let intel = ComputePipes { simd: 56, matrix: 56 };
        assert_eq!(intel.worker_roof(128, true), 56);
        assert_eq!(intel.worker_roof(8, true), 8);
        assert_eq!(intel.worker_roof(8, false), 8);

        // No matrix unit. The matrix roof is the SIMD cores, never zero.
        let simd_only = ComputePipes { simd: 8, matrix: 0 };
        assert_eq!(simd_only.worker_roof(32, true), 8);
        assert_eq!(simd_only.worker_roof(4, true), 4);
        assert_eq!(simd_only.worker_roof(32, false), 8);
    }

    /// The live probe and the launch roof agree with the cluster formula.
    #[cfg(target_os = "macos")]
    #[test]
    fn probed_pipes_match_the_performance_cluster() {
        let Some(cluster) = perf_cluster() else { return };
        let pipes = compute_pipes().unwrap();
        assert_eq!(pipes, ComputePipes::from_perf_cluster(cluster));
        assert_eq!(hardware_workers(8, true), pipes.worker_roof(8, true));
        assert_eq!(hardware_workers(8, false), pipes.worker_roof(8, false));
    }

    #[test]
    fn scored_threads_are_the_pool_clamped_to_the_dispatch_cap() {
        let cluster = PerfCluster { physical_cpus: 12, cpus_per_l2: Some(4) };
        let scored = dispatch_threads(16, cluster);
        assert_eq!(scored.matrix, 16.min(concurrency_cap(Some(cluster), apple_amx())));
        assert_eq!(scored.neon, 16.min(concurrency_cap(Some(cluster), IsaReq::ANY)));
        assert_eq!(scored.neon, 16.min(concurrency_cap(Some(cluster), sme())));
        let one = dispatch_threads(1, cluster);
        assert_eq!(one, DispatchThreads { neon: 1, matrix: 1 });
    }

    /// The probe is whatever this host exposes. Injected counts cover the formula;
    /// here we only check that a machine which has `perflevel0` is read as that
    /// cluster and not as `hw.physicalcpu` once efficiency cores exist.
    #[cfg(target_os = "macos")]
    #[test]
    fn probed_cluster_is_perflevel0_not_the_whole_soc() {
        let Some(cluster) = perf_cluster() else { return };
        assert!(cluster.physical_cpus >= 1);
        assert_eq!(
            cluster.matrix_units(),
            match cluster.cpus_per_l2 {
                Some(per) if per > 0 => (cluster.physical_cpus / per).max(1),
                _ => 1,
            }
        );
        let all = crate::cache::sysctl_usize("hw.physicalcpu");
        let efficiency = crate::cache::sysctl_usize("hw.perflevel1.physicalcpu");
        if let (Some(all), Some(efficiency)) = (all, efficiency) {
            assert_eq!(cluster.physical_cpus + efficiency, all);
            assert!(cluster.physical_cpus < all);
        }
    }
}
