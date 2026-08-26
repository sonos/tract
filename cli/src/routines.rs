//! The single-winner kernel matrix: which implementation every machine would run for every
//! function, as one grid of coloured cells, with the whole list for one machine underneath.
//!
//! A column is a deployment target, a row is a function and datum type, and the colour is the
//! whole message: green for a kernel written for that machine's own rung of the instruction-set
//! ladder, white for correct code that was not written for it -- portable Rust, or an architecture
//! kernel from a rung below -- red for portable f16 code, which converts to f32 and back around
//! every single operation, blue for a cell closed on purpose, and a dot for a pair nothing
//! implements. Red is the cost no machine has to pay.
//!
//! Columns come from the instruction-set ladders rather than a hand-kept list, one per rung under
//! its architecture, plus one for the host, which is exact rather than the nearest rung: a machine
//! can have `avx512f` without the `avx-vnni` its ladder step bundles, and then no rung describes
//! it. A rung whose answers repeat the rung below still earns a column, because a step with
//! nothing written for it yet is what a reader is looking for.
//!
//! Which kernel a cell stands for is the list under the grid, for the host or for whatever
//! `--isa` names, along with what closed a blue one. Machines are named above the grid in the form
//! `--isa` reads back, so a column can be pasted in.
//!
//! Only [`crate::selection`] speaks for matmul. These are the kernels chosen once per function,
//! with no shape to weigh.

use nu_ansi_term::Color;
use nu_ansi_term::Color::*;
use tract_core::internal::*;
use tract_data::prelude::DatumType;
use tract_linalg::isa::IsaSet;
use tract_linalg::routines::{
    Func, RoutineFactory, Standing, best_for, declared, settled_why, standing,
};

/// Width of one machine column. A cell is one character; the header labels want the rest.
const COLUMN: usize = 9;

/// What a cell is drawn with: enough ink for its colour to register at a glance, and the same
/// disc left open where nothing is implemented at all. One character each, so a font that draws
/// them wide shifts every cell alike.
const FILLED: &str = "⬤";
const HOLLOW: &str = "◯";

/// Width of the function-and-datum-type label opening a row.
const ROW: usize = 20;

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

/// The architecture a column belongs to, short enough for a column: what the tree is named after,
/// without the baseline feature the wasm tree carries in its own name.
fn arch_name(isa: &IsaSet) -> String {
    isa.arch()
        .map(|a| a.to_string().split('+').next().unwrap_or_default().to_string())
        .unwrap_or_else(|| "none".to_string())
}

/// The whole machine, in the form `--isa` reads back. Rendered as `tract selection` renders it,
/// so one machine reads the same in both.
fn machine_name(isa: &IsaSet) -> String {
    format!("{isa:?}")
}

/// The colour a standing is drawn in. It carries the verdict, so a cell itself only has to say
/// whether anything is there at all.
fn ink(standing: Standing) -> Color {
    match standing {
        Standing::Missing => DarkGray,
        Standing::Dedicated => Green,
        Standing::Unspecialized => White,
        Standing::Emulated => LightRed,
        Standing::Settled => Blue,
    }
}

/// Centre inside `w` characters, always leaving a space so two full labels never run together.
/// Truncation counts characters rather than bytes: the ellipsis is not ASCII even when the labels
/// are.
fn centre_in(text: &str, w: usize) -> String {
    let room = w - 1;
    let text: String = if text.chars().count() <= room {
        text.to_string()
    } else {
        text.chars().take(room - 1).chain(std::iter::once('…')).collect()
    };
    format!("{text:^w$}")
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

/// One column per rung of every architecture's ladder, by the rung's nickname, plus the host,
/// which is exact rather than the nearest rung.
fn columns() -> Vec<(String, IsaSet)> {
    let mut cols: Vec<(String, IsaSet)> = crate::selection::machines()
        .into_iter()
        .map(|isa| (isa.nickname().to_string(), isa))
        .collect();
    cols.push(("host".to_string(), tract_linalg::isa::native()));
    cols
}

/// What each column stands for, in the form `--isa` takes.
fn key(cols: &[(String, IsaSet)]) {
    for (label, isa) in cols {
        println!("  {}   {}", centre_in(label, COLUMN), DarkGray.paint(machine_name(isa)));
    }
}

/// The architecture over every column, then the rung. Two lines, so a column is only as wide as
/// one rung's nickname.
fn header(cols: &[(String, IsaSet)], chosen: usize) {
    print!("{:<ROW$}", "");
    for (_, isa) in cols {
        print!("{}", DarkGray.paint(centre_in(&arch_name(isa), COLUMN)));
    }
    println!();
    print!("{:<ROW$}", "");
    for (i, (label, _)) in cols.iter().enumerate() {
        let label = centre_in(label, COLUMN);
        if i == chosen { print!("{}", White.bold().paint(label)) } else { print!("{label}") }
    }
    println!();
}

pub fn dump(isa: Option<&str>) -> TractResult<()> {
    let machine = match isa {
        Some(spec) => spec.parse::<IsaSet>()?,
        None => tract_linalg::isa::native(),
    };
    let rows = rows();
    let cols = columns();
    let chosen = cols.iter().position(|(_, isa)| *isa == machine).unwrap_or(cols.len());

    println!();
    println!("{}", White.bold().paint("# Routine kernels by machine"));
    println!(
        "  {} written for this machine   {} not written for it   {} f16 converted per operation   {} settled on purpose   {} nothing implements it",
        Green.paint(FILLED),
        White.paint(FILLED),
        LightRed.paint(FILLED),
        Blue.paint(FILLED),
        DarkGray.paint(HOLLOW),
    );
    println!();
    key(&cols);
    println!();
    header(&cols, chosen);

    let mut group = "";
    for (func, dt, _, this_group) in &rows {
        if *this_group != group {
            group = this_group;
            println!("{}", DarkGray.paint(format!("── {group} ──")));
        }
        print!("{:<ROW$}", format!("{}/{dt:?}", func.name()));
        for (i, (_, isa)) in cols.iter().enumerate() {
            let standing = standing(*func, *dt, isa);
            let cell =
                centre_in(if standing == Standing::Missing { HOLLOW } else { FILLED }, COLUMN);
            if i == chosen {
                print!("{}", ink(standing).bold().paint(cell))
            } else {
                print!("{}", ink(standing).paint(cell))
            }
        }
        println!();
    }

    machine_list(&machine, &rows);
    Ok(())
}

/// Every kernel one machine runs, named, and what closed the cells that are closed.
fn machine_list(machine: &IsaSet, rows: &[(Func, DatumType, u8, &'static str)]) {
    println!();
    println!("{}", White.bold().paint(format!("# {}", machine_name(machine))));
    println!("  {}", DarkGray.paint("--isa takes any machine named above"));
    let mut group = "";
    for (func, dt, _, this_group) in rows {
        if *this_group != group {
            group = this_group;
            println!("{}", DarkGray.paint(format!("── {group} ──")));
        }
        let standing = standing(*func, *dt, machine);
        let kernel = best_for(*func, *dt, machine).map_or("—", |r| r.name());
        let pair = format!("{}/{dt:?}", func.name());
        match settled_why(*func, *dt, machine) {
            Some(why) => println!(
                "  {pair:<ROW$}{} {}",
                ink(standing).paint(format!("{kernel:<38}")),
                DarkGray.paint(why)
            ),
            None => println!("  {pair:<ROW$}{}", ink(standing).paint(kernel)),
        }
    }
}
