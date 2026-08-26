//! The single-winner kernel matrix: which implementation every machine would run for every
//! function, as one grid.
//!
//! A column is a deployment target, a row is a function and datum type. The colour is the
//! message: a cell served by a kernel written for that architecture is green, one falling back on
//! portable Rust is red, and a pair no tree implements at all is a dot. A red cell on a column
//! whose hardware could do better is an optimisation waiting to be written, which is what the
//! trailing gap list collects.
//!
//! The cell text names what the winning kernel needs from the instruction set, so it also shows
//! the subtler gap: an f16 row answering `aarch64` on a column that has `fp16` is served by the
//! baseline NEON tree, which means an f32 round trip where a native f16 kernel could run.
//!
//! Columns come from the instruction-set ladders rather than a hand-kept list — a rung whose
//! answers are identical to the rung below it earns no column and is named in the footer instead
//! — plus one for the host, which is exact rather than the nearest rung: a machine can have
//! `avx512f` without the `avx-vnni` its ladder step bundles, and then no rung describes it.
//!
//! Only [`crate::selection`] speaks for matmul. These are the kernels chosen once per function,
//! with no shape to weigh.

use nu_ansi_term::Color::*;
use tract_core::internal::*;
use tract_data::prelude::DatumType;
use tract_linalg::isa::IsaSet;
use tract_linalg::routines::{Func, Routine, RoutineFactory, best_for, declared};

/// Width of one machine column, wider when cells carry kernel names. Anything longer is elided
/// here and, for a machine, spelled in full by the key above the grid.
static W: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(16);

fn width() -> usize {
    W.load(std::sync::atomic::Ordering::Relaxed)
}

/// The shape of kernel a descriptor is, which is what its factory arm already says. Rows are
/// grouped by it so a reader compares like with like.
fn family(factory: &RoutineFactory) -> (u8, &'static str) {
    match factory {
        RoutineFactory::F32(_) | RoutineFactory::F16(_) => (0, "element-wise"),
        RoutineFactory::F32Param(_) | RoutineFactory::F16Param(_) => {
            (1, "element-wise, scalar parameter")
        }
        RoutineFactory::F32Reduce(_) | RoutineFactory::F16Reduce(_) => (2, "reduction"),
        RoutineFactory::F32MapReduce(_) => (3, "map-reduction"),
        RoutineFactory::RmsNormF32 { .. } => (4, "fused row-wise"),
        RoutineFactory::LutU8 { .. } => (5, "look-up table"),
        RoutineFactory::BinF32 { .. } | RoutineFactory::BinF16 { .. } => (6, "binary"),
    }
}

/// The features a set offers beyond its bare architecture, which is what distinguishes one
/// machine from another.
fn features(isa: &IsaSet) -> Vec<&'static str> {
    isa.iter().filter(|i| !i.is_arch()).map(|i| i.name()).collect()
}

fn arch_name(isa: &IsaSet) -> String {
    isa.arch().map(|a| a.to_string()).unwrap_or_else(|| "no tree".to_string())
}

/// The whole machine, for the key. Rendered as `tract selection` renders it, so one machine reads
/// the same in both.
fn machine_name(isa: &IsaSet) -> String {
    format!("{isa:?}")
}

/// What the kernel serving a cell needs to run: the features it declares beyond its
/// architecture, or the architecture alone when it declares none. A portable kernel needs
/// nothing and says so.
fn needs_label(routine: &Routine) -> String {
    let Some(arch) = routine.arch else { return "generic".to_string() };
    let needs: Vec<&str> =
        routine.isa.needs.iter().filter(|i| !i.is_arch()).map(|i| i.name()).collect();
    if needs.is_empty() { arch.to_string() } else { needs.join("+") }
}

/// The kernel's own name, minus what the row already says: the function and the datum type. What
/// survives is the tree it lives in and how it is built — `_lut`, `_fused`, the vector width — so
/// two machines running the same tree can still be told apart.
fn name_label(routine: &Routine, func: Func, dt: DatumType) -> String {
    let spent: Vec<String> = func
        .name()
        .split('_')
        .map(str::to_string)
        .chain(std::iter::once(format!("{dt:?}").to_lowercase()))
        .collect();
    let kept: Vec<&str> =
        routine.name().split('_').filter(|t| !spent.iter().any(|s| s == t)).collect();
    kept.join("_")
}

/// Right-align inside one column, always leaving a space so two full cells never run together.
/// Truncation counts characters rather than bytes: the ellipsis is not ASCII even when the names
/// are.
fn elide(text: &str) -> String {
    let w = width();
    let room = w - 1;
    let text: String = if text.chars().count() <= room {
        text.to_string()
    } else {
        text.chars().take(room - 1).chain(std::iter::once('…')).collect()
    };
    format!("{text:>w$}")
}

/// Every distinct pair a tree implements, grouped by kernel shape then named in a stable order.
fn rows() -> Vec<(Func, DatumType, u8, &'static str)> {
    let mut rows: Vec<(Func, DatumType, u8, &'static str)> = declared()
        .map(|r| {
            let (order, group) = family(&r.factory);
            (r.func, r.dt(), order, group)
        })
        .collect();
    rows.sort_by_key(|(func, dt, order, _)| (*order, func.name(), format!("{dt:?}")));
    rows.dedup_by_key(|(func, dt, _, _)| (*func, *dt));
    rows
}

/// The machines worth a column, and the ones folded into the rung below them. A rung earns its
/// width by changing an answer; the host is always shown, whether or not a rung matches it.
fn columns(rows: &[(Func, DatumType, u8, &'static str)]) -> (Vec<(String, IsaSet)>, Vec<String>) {
    let answers = |isa: &IsaSet| -> Vec<Option<&'static str>> {
        rows.iter().map(|(f, dt, ..)| best_for(*f, *dt, isa).map(|r| r.name())).collect()
    };
    let mut cols: Vec<(String, IsaSet)> = vec![];
    let mut folded: Vec<String> = vec![];
    let mut last: Option<(IsaSet, Vec<Option<&'static str>>)> = None;
    for isa in crate::selection::machines() {
        let mine = answers(&isa);
        if let Some((prev, ref theirs)) = last
            && prev.arch() == isa.arch()
            && *theirs == mine
        {
            folded.push(machine_name(&isa));
            continue;
        }
        let previous: Vec<&str> = match last {
            Some((prev, _)) if prev.arch() == isa.arch() => features(&prev),
            _ => vec![],
        };
        let added: Vec<&str> =
            features(&isa).into_iter().filter(|f| !previous.contains(f)).collect();
        let label =
            if added.is_empty() { arch_name(&isa) } else { format!("+{}", added.join("+")) };
        cols.push((label, isa));
        last = Some((isa, mine));
    }
    cols.push(("this host".to_string(), tract_linalg::isa::native()));
    (cols, folded)
}

pub fn dump(names: bool) -> TractResult<()> {
    if names {
        W.store(23, std::sync::atomic::Ordering::Relaxed);
    }
    let rows = rows();
    let (cols, folded) = columns(&rows);
    let host = cols.len() - 1;

    println!();
    println!("{}", White.bold().paint("# Routine kernels by machine"));
    println!(
        "  {} written for the architecture   {} portable Rust   {} nothing implements it",
        Green.paint("■"),
        LightRed.paint("■"),
        DarkGray.paint("·"),
    );
    println!();
    for (label, isa) in &cols {
        println!("  {}   {}", elide(label), DarkGray.paint(machine_name(isa)));
    }
    println!();

    print!("{:<26}", "");
    for (i, (label, _)) in cols.iter().enumerate() {
        let head = elide(label);
        if i == host { print!("{}", White.bold().paint(head)) } else { print!("{head}") }
    }
    println!();

    let mut group = "";
    let mut gaps: Vec<String> = vec![];
    for (func, dt, _, this_group) in &rows {
        if *this_group != group {
            group = this_group;
            println!("{}", DarkGray.paint(format!("── {group} ──")));
        }
        print!("{:<26}", format!("{}/{dt:?}", func.name()));
        for (i, (_, isa)) in cols.iter().enumerate() {
            let answer = best_for(*func, *dt, isa);
            let (text, color) = match answer {
                None => ("·".to_string(), DarkGray),
                Some(r) => {
                    let text = if names { name_label(r, *func, *dt) } else { needs_label(r) };
                    (text, if r.arch.is_some() { Green } else { LightRed })
                }
            };
            // Only the host earns a gap list. Every column has portable cells, but a baseline
            // machine is portable almost throughout and nobody writes VFP or SSE2 kernels to fix
            // that, so listing them all would bury the one list a reader can act on.
            if i == host
                && answer.is_some_and(|r| r.arch.is_none())
                && cols
                    .iter()
                    .any(|(_, other)| best_for(*func, *dt, other).is_some_and(|r| r.arch.is_some()))
            {
                gaps.push(format!("{}/{dt:?}", func.name()));
            }
            let text = elide(&text);
            if i == host {
                print!("{}", color.bold().paint(text))
            } else {
                print!("{}", color.paint(text))
            }
        }
        println!();
    }

    if !folded.is_empty() {
        println!();
        println!(
            "{}",
            DarkGray.paint(format!("changes nothing, so has no column: {}", folded.join(", ")))
        );
    }
    report_gaps(gaps);
    Ok(())
}

/// What this machine runs portable Rust for while another machine has a real kernel: the ports
/// worth writing here. Only the host, because the grid already shows every column in colour and
/// a baseline machine is portable nearly throughout.
fn report_gaps(gaps: Vec<String>) {
    if gaps.is_empty() {
        return;
    }
    println!();
    println!("{}", White.bold().paint("# Gaps on this host"));
    println!("  {}", DarkGray.paint("portable Rust here, hand-written on some other machine"));
    for gap in &gaps {
        println!("  {gap}");
    }
}
