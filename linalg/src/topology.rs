//! The machine's cores as the OS reports them: how many, in which performance
//! levels, and how many share an L2.
//!
//! Facts only. How a kernel's speed scales with threads on this layout — how
//! many workers a shared matrix unit absorbs, whether efficiency cores help —
//! is for the cost models to measure, not for this module to assume.
//!
//! Apple targets read `hw.nperflevels` and each `hw.perflevelN`, performance
//! cores first. Linux reports one level of physical cores (online CPUs divided
//! by cpu0's SMT siblings). Other targets report one level of
//! `available_parallelism`, which counts logical processors.

use std::sync::OnceLock;

use crate::isa::Isa;
use crate::isa::IsaReq;

/// One performance level: Apple's `hw.perflevelN`, or every physical core
/// where the OS has no notion of levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreLevel {
    pub physical_cpus: usize,
    /// Cores sharing one L2, when the OS reports it.
    pub cpus_per_l2: Option<usize>,
}

impl CoreLevel {
    /// L2 clusters in this level: `physical_cpus / cpus_per_l2`, at least one.
    /// An unreported or zero sharing degree counts as a single cluster.
    pub fn clusters(self) -> usize {
        match self.cpus_per_l2 {
            Some(per) if per > 0 => (self.physical_cpus / per).max(1),
            _ => 1,
        }
    }
}

/// The machine's core levels, fastest first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub levels: Vec<CoreLevel>,
}

impl Topology {
    /// Physical cores across all levels.
    pub fn physical_cpus(&self) -> usize {
        self.levels.iter().map(|l| l.physical_cpus).sum()
    }

    /// The fastest level.
    pub fn performance(&self) -> Option<CoreLevel> {
        self.levels.first().copied()
    }
}

/// This machine's topology, probed once per process. `None` when the OS
/// reports nothing usable.
pub fn topology() -> Option<&'static Topology> {
    static TOPOLOGY: OnceLock<Option<Topology>> = OnceLock::new();
    TOPOLOGY.get_or_init(probe).as_ref()
}

/// Whether a kernel with these requirements issues to a matrix unit: Apple
/// AMX, SME or SME2, or Intel AMX.
pub fn uses_matrix_unit(isa: IsaReq) -> bool {
    isa.needs.iter().any(|i| {
        matches!(
            i,
            Isa::Aarch64AppleAmx
                | Isa::Aarch64Sme
                | Isa::Aarch64Sme2
                | Isa::X86_64AmxInt8
                | Isa::X86_64AmxBf16
        )
    })
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn probe() -> Option<Topology> {
    use crate::cache::sysctl_usize;
    let count = sysctl_usize("hw.nperflevels").unwrap_or(1);
    let levels: Vec<CoreLevel> = (0..count)
        .filter_map(|l| {
            Some(CoreLevel {
                physical_cpus: sysctl_usize(&format!("hw.perflevel{l}.physicalcpu"))?,
                cpus_per_l2: sysctl_usize(&format!("hw.perflevel{l}.cpusperl2")),
            })
        })
        .collect();
    if !levels.is_empty() {
        return Some(Topology { levels });
    }
    let physical_cpus = sysctl_usize("hw.physicalcpu")?;
    Some(Topology { levels: vec![CoreLevel { physical_cpus, cpus_per_l2: None }] })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn probe() -> Option<Topology> {
    use crate::cache::count_cpu_list;
    let online = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    let logical = count_cpu_list(&online);
    if logical == 0 {
        return None;
    }
    let smt = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
        .ok()
        .map(|s| count_cpu_list(&s))
        .filter(|&n| n > 0)
        .unwrap_or(1);
    Some(Topology {
        levels: vec![CoreLevel { physical_cpus: (logical / smt).max(1), cpus_per_l2: None }],
    })
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android"
)))]
fn probe() -> Option<Topology> {
    let physical_cpus = std::thread::available_parallelism().ok()?.get();
    Some(Topology { levels: vec![CoreLevel { physical_cpus, cpus_per_l2: None }] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_divide_cores_by_l2_sharing() {
        let m4_performance = CoreLevel { physical_cpus: 4, cpus_per_l2: Some(4) };
        assert_eq!(m4_performance.clusters(), 1);
        let m1_pro_performance = CoreLevel { physical_cpus: 8, cpus_per_l2: Some(4) };
        assert_eq!(m1_pro_performance.clusters(), 2);
        let unreported = CoreLevel { physical_cpus: 8, cpus_per_l2: None };
        assert_eq!(unreported.clusters(), 1);
        let zero = CoreLevel { physical_cpus: 8, cpus_per_l2: Some(0) };
        assert_eq!(zero.clusters(), 1);
    }

    #[test]
    fn levels_add_up_and_the_first_is_performance() {
        let m4 = Topology {
            levels: vec![
                CoreLevel { physical_cpus: 4, cpus_per_l2: Some(4) },
                CoreLevel { physical_cpus: 6, cpus_per_l2: Some(6) },
            ],
        };
        assert_eq!(m4.physical_cpus(), 10);
        assert_eq!(m4.performance(), Some(m4.levels[0]));
    }

    #[test]
    fn matrix_units_are_apple_amx_sme_and_intel_amx() {
        for req in [
            IsaReq::ANY.needing(&[Isa::Aarch64AppleAmx]),
            IsaReq::ANY.needing(&[Isa::Aarch64Sme]),
            IsaReq::ANY.needing(&[Isa::Aarch64Sme2]),
            IsaReq::ANY.needing(&[Isa::X86_64AmxInt8]),
            IsaReq::ANY.needing(&[Isa::X86_64AmxBf16]),
        ] {
            assert!(uses_matrix_unit(req));
        }
        assert!(!uses_matrix_unit(IsaReq::ANY));
        assert!(!uses_matrix_unit(IsaReq::ANY.needing(&[Isa::Aarch64Fp16])));
    }

    /// Whatever this host reports must be self-consistent; on Apple silicon the
    /// levels must account for every physical core.
    #[test]
    fn probe_is_consistent_with_the_os() {
        let Some(t) = topology() else { return };
        assert!(!t.levels.is_empty());
        assert!(t.levels.iter().all(|l| l.physical_cpus > 0));
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(all) = crate::cache::sysctl_usize("hw.physicalcpu") {
            assert_eq!(t.physical_cpus(), all);
        }
    }
}
