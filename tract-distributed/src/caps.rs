//! Node capability + host-stats sampling (via `sysinfo`) for the control plane.

use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::protocol::{NodeCaps, NodeStats};

/// Holds a `System` refreshed on demand so CPU% deltas are meaningful.
pub struct HostSampler {
    sys: System,
    pid: Pid,
}

/// Physical memory this process really occupies. On macOS the classic resident-set
/// size omits compressed and GPU-driver pages — a worker holding a multi-GB shard
/// reports single-digit MB — so read `phys_footprint`, the figure Activity Monitor
/// shows. Elsewhere RSS is representative.
fn process_memory(sys: &System, pid: Pid) -> u64 {
    #[cfg(target_vendor = "apple")]
    {
        use libproc::libproc::pid_rusage::{RUsageInfoV2, pidrusage};
        if let Ok(u) = pidrusage::<RUsageInfoV2>(pid.as_u32() as i32) {
            return u.ri_phys_footprint;
        }
    }
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

impl Default for HostSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl HostSampler {
    pub fn new() -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        HostSampler { sys, pid: Pid::from_u32(std::process::id()) }
    }

    /// A node id that is stable across restarts, so a restarted worker reclaims
    /// its dashboard card instead of appearing as a new node. If `name` is given
    /// it is used verbatim; give co-located workers distinct names. Otherwise a
    /// per-host id is persisted under `~/.dis-tract/node_id` and reused (EXO's
    /// approach — correct for one worker per host).
    pub fn node_id(name: Option<&str>) -> String {
        if let Some(n) = name {
            return n.to_string();
        }
        let fallback = || {
            format!(
                "{}-{}",
                System::host_name().unwrap_or_else(|| "host".into()),
                std::process::id()
            )
        };
        let Ok(home) = std::env::var("HOME") else { return fallback() };
        let dir = std::path::Path::new(&home).join(".dis-tract");
        let path = dir.join("node_id");
        if let Ok(id) = std::fs::read_to_string(&path)
            && !id.trim().is_empty()
        {
            return id.trim().to_string();
        }
        let fresh = fallback();
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(&path, &fresh);
        fresh
    }

    /// Capabilities advertised on join. `budget_frac` caps how much of available
    /// memory the planner may fill for this node's shard.
    pub fn caps(&mut self, node_id: String, backend: String, budget_frac: f64) -> NodeCaps {
        self.sys.refresh_memory();
        let avail = self.sys.available_memory();
        NodeCaps {
            node_id,
            hostname: System::host_name().unwrap_or_default(),
            backend,
            total_mem: self.sys.total_memory(),
            avail_mem: avail,
            mem_budget: (avail as f64 * budget_frac) as u64,
            cpus: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        }
    }

    /// A telemetry snapshot; the caller fills in stage/tokens/latency.
    #[allow(clippy::too_many_arguments)]
    pub fn stats(
        &mut self,
        node_id: String,
        stage: usize,
        backend: String,
        tokens: u64,
        last_step_ms: f64,
        tok_s: f64,
        mem_budget: u64,
        weights_bytes: u64,
    ) -> NodeStats {
        self.sys.refresh_memory();
        self.sys.refresh_cpu_usage();
        self.sys.refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
        NodeStats {
            node_id,
            hostname: System::host_name().unwrap_or_default(),
            stage,
            backend,
            tokens,
            last_step_ms,
            tok_s,
            host_cpu: self.sys.global_cpu_usage(),
            host_mem_used: self.sys.used_memory(),
            host_mem_total: self.sys.total_memory(),
            mem_footprint: process_memory(&self.sys, self.pid),
            mem_budget,
            weights_bytes,
        }
    }
}
