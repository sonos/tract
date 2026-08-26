//! What mmm selection answers, for every machine at once, as one diffable dump.
//!
//! Not a report to read: a fixed sweep printed in a fixed order, so two builds can be dumped and
//! the outputs diffed. That is what a dispatch refactor needs to show it changed nothing, and the
//! sweep is hard-coded rather than configurable precisely so the two sides always agree on it.
//!
//! Both halves of selection are printed: the sets it narrows ([`declared`], [`runnable_for`]) and
//! the choice it makes over them. A kernel is named, never run, so this is safe on any host and
//! covers architectures this one cannot execute — with tract-linalg's `foreign-inventory`, on by
//! default for x86_64 and Apple hosts, every tree tract has. The header names them.
//!
//! What it cannot be is a checked-in expected output. A few tiers read the machine rather than
//! the instruction set — the Apple chip generation, the Cortex model behind the a53/a55 boosts,
//! the AMX virtualisation heuristic, the `TRACT_AMX_BF16` knob — so the same revision dumps
//! differently on different hosts. Diff two builds on one host, and run it on the board when the
//! board is what the question is about.
//!
//! One row is not the whole truth either: for the architecture this host is, a built kernel stays
//! in the set only if the host's own instruction set can run it, so a native row describes no
//! machine smaller than this host and `TRACT_CPU_ISA` is what shrinks it. That skew is the same
//! on both sides of a diff, which is what the dump is for.

use tract_core::internal::*;
use tract_data::prelude::{Datum, f16};
use tract_linalg::isa::{Arch, IsaSet};
use tract_linalg::mmm::{MmmDispatch, Query, Suitable, retain_best};
use tract_linalg::mmm_routines::{declared, runnable_for};

/// The dims every query is built from, each standing for a band the shape rules treat
/// differently: unknown, a matrix-vector, a tile no kernel fills, the widths the wasm and AMX
/// gates key on, and enough to be blocked.
const DIMS: [Option<usize>; 7] = [None, Some(1), Some(2), Some(8), Some(16), Some(64), Some(512)];

/// One way to compute the query, named so two dumps compare: the kernel, which of its packings,
/// and whether it is reached through a panel extractor.
fn tag(suitable: &Suitable) -> String {
    let (mmm, packing, extractor) = suitable;
    format!("{}#{}{}", mmm.name(), packing, if extractor.is_some() { "+x" } else { "" })
}

fn dim(d: Option<usize>) -> String {
    d.map(|d| d.to_string()).unwrap_or_else(|| "*".to_string())
}

/// Every distinct machine the ladders describe, in a fixed order. Consecutive rungs that add no
/// feature tract has a kernel behind collapse, so an architecture contributes one entry per
/// cohort that actually differs rather than one per level.
fn machines() -> Vec<IsaSet> {
    let mut machines: Vec<IsaSet> = vec![];
    for isa in IsaSet::every_ladder() {
        if machines.last() != Some(&isa) {
            machines.push(isa);
        }
    }
    machines
}

pub fn dump() -> TractResult<()> {
    let arches: Vec<String> = Arch::ALL
        .iter()
        .filter(|a| declared().any(|r| (r.make)().arch() == Some(**a)))
        .map(|a| a.to_string())
        .collect();
    // Which trees this build compiled, which is also the honest report of whether
    // `foreign-inventory` is on: it is a feature of tract-linalg, not of this crate, so nothing
    // here can read it directly.
    println!("# trees\t{}", arches.join(" "));

    let mut rows: Vec<String> = declared()
        .map(|r| {
            let k = (r.make)();
            let arch = k.arch().map(|a| a.to_string()).unwrap_or_else(|| "generic".to_string());
            format!(
                "DECL\t{}\t{arch}\tbuilt={}\temulated={}\tisa={:?}\tboost={}",
                k.name(),
                k.built(),
                k.emulated(),
                k.isa(),
                k.boost()
            )
        })
        .collect();
    rows.sort();
    rows.iter().for_each(|r| println!("{r}"));

    for arch in Arch::ALL {
        let mut names: Vec<String> =
            runnable_for(arch).iter().map(|k| k.name().to_string()).collect();
        names.sort();
        println!("RUNNABLE\t{arch}\t{}\t{}", names.len(), names.join(","));
    }

    for isa in machines() {
        let dispatch = MmmDispatch::for_isa(isa);
        println!(
            "LADDER\t{isa:?}\t{}",
            dispatch.tiers().iter().map(|t| t.name).collect::<Vec<_>>().join(" > ")
        );
        for acc in [f32::datum_type(), f16::datum_type(), i32::datum_type()] {
            for m in DIMS {
                for k in DIMS {
                    for n in DIMS {
                        let query = Query::plain(acc, m, k, n);
                        let mut suitable = dispatch.suitable(&query);
                        // The ungated answer, then the one selection actually honours: they part
                        // when a tier names a kernel that is not written for this architecture.
                        let tier = dispatch
                            .preferred_kernel(acc, m, k, n)
                            .map(|mmm| mmm.name().to_string());
                        let honoured = dispatch.preferred(&query, &suitable).map(|s| tag(&s));
                        let pick = dispatch.pick(&query).map(|s| tag(&s));
                        retain_best(&mut suitable);
                        let mut retained: Vec<String> = suitable.iter().map(tag).collect();
                        retained.sort();
                        println!(
                            "PICK\t{isa:?}\t{acc:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                            dim(m),
                            dim(k),
                            dim(n),
                            tier.unwrap_or_else(|| "-".to_string()),
                            honoured.unwrap_or_else(|| "-".to_string()),
                            pick.unwrap_or_else(|| "-".to_string()),
                            retained.join(",")
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
