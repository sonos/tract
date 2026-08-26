//! Cross-arch registry of the single-winner kernels: one function, one datum type, one best
//! implementation per machine.
//!
//! Every element-wise activation, scalar-parameter kernel and reduction declares itself here as
//! data, so the whole function x target matrix is enumerable on any host -- with the
//! `foreign-inventory` feature, including the trees this build cannot run. [`best_for`] answers
//! which one a machine would use, and it takes the machine as an argument rather than reading
//! the host, so the same query serves dispatch and introspection.
//!
//! This is deliberately not the model [`crate::mmm_routines`] uses. A matmul has many co-valid
//! kernels per machine and the winner depends on the shape, so mmm keeps a pool and a tier
//! ladder. These have no shape to weigh: one kernel wins outright, by what it is written
//! against.
//!
//! A tree's kernels are declared whether or not this build compiled their bodies, so a
//! descriptor being here means "such a kernel exists", not "it runs here". Only [`best_for`]'s
//! answer for the *native* machine may be executed; anything else is metadata and would bail.
//! What a build could not assemble at all is not declared, so an absent descriptor means "no
//! such kernel", never "this toolchain skipped it".
use crate::element_wise::{ElementWise, ElementWiseKer};
use crate::isa::{Arch, Isa, IsaReq, IsaSet, LEVEL_BOOST};
use crate::lut::Lut;
use crate::reduce::{MapReduce, MapReduceKer, Reduce, ReduceKer};
use tract_data::internal::*;

/// A function a routine computes. One variant per function, whatever the datum types or the
/// number of implementations behind it.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Func {
    Sigmoid,
    Tanh,
    Silu,
    Gelu,
    Erf,
    Hardswish,
    LeakyRelu,
    MulByScalar,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    Softmax2,
    RmsNorm,
    Lut,
    /// A binary operation with the scalar broadcast over the whole slice.
    BinByScalar(crate::BinOp),
    /// A binary operation between two slices of the same length.
    BinUnicast(crate::BinOp),
}

impl Func {
    /// Every binary operation, in both layouts: what [`Self::ALL`] ends with.
    const BIN: [Func; 12] = {
        use crate::BinOp::*;
        [
            Func::BinByScalar(Min),
            Func::BinByScalar(Max),
            Func::BinByScalar(Add),
            Func::BinByScalar(Mul),
            Func::BinByScalar(Sub),
            Func::BinByScalar(SubF),
            Func::BinUnicast(Min),
            Func::BinUnicast(Max),
            Func::BinUnicast(Add),
            Func::BinUnicast(Mul),
            Func::BinUnicast(Sub),
            Func::BinUnicast(SubF),
        ]
    };

    pub const ALL: [Func; 26] = [
        Func::Sigmoid,
        Func::Tanh,
        Func::Silu,
        Func::Gelu,
        Func::Erf,
        Func::Hardswish,
        Func::ReduceMax,
        Func::ReduceMin,
        Func::ReduceSum,
        Func::Softmax2,
        Func::RmsNorm,
        Func::Lut,
        Func::LeakyRelu,
        Func::MulByScalar,
        Func::BIN[0],
        Func::BIN[1],
        Func::BIN[2],
        Func::BIN[3],
        Func::BIN[4],
        Func::BIN[5],
        Func::BIN[6],
        Func::BIN[7],
        Func::BIN[8],
        Func::BIN[9],
        Func::BIN[10],
        Func::BIN[11],
    ];

    /// Where this function sits in [`Self::ALL`], which is what indexes the dispatch table.
    fn slot(self) -> usize {
        match self {
            Func::Sigmoid => 0,
            Func::Tanh => 1,
            Func::Silu => 2,
            Func::Gelu => 3,
            Func::Erf => 4,
            Func::Hardswish => 5,
            Func::ReduceMax => 6,
            Func::ReduceMin => 7,
            Func::ReduceSum => 8,
            Func::Softmax2 => 9,
            Func::RmsNorm => 10,
            Func::Lut => 11,
            Func::LeakyRelu => 12,
            Func::MulByScalar => 13,
            Func::BinByScalar(op) => 14 + op as usize,
            Func::BinUnicast(op) => 20 + op as usize,
        }
    }

    /// The name the matrix and the logs use.
    pub fn name(&self) -> &'static str {
        match self {
            Func::Sigmoid => "sigmoid",
            Func::Tanh => "tanh",
            Func::Silu => "silu",
            Func::Gelu => "gelu",
            Func::Erf => "erf",
            Func::Hardswish => "hardswish",
            Func::LeakyRelu => "leaky_relu",
            Func::MulByScalar => "mul_by_scalar",
            Func::ReduceMax => "reduce_max",
            Func::ReduceMin => "reduce_min",
            Func::ReduceSum => "reduce_sum",
            Func::Softmax2 => "softmax2",
            Func::RmsNorm => "rms_norm",
            Func::Lut => "lut",
            Func::BinByScalar(op) => match op {
                crate::BinOp::Min => "by_scalar_min",
                crate::BinOp::Max => "by_scalar_max",
                crate::BinOp::Add => "by_scalar_add",
                crate::BinOp::Mul => "by_scalar_mul",
                crate::BinOp::Sub => "by_scalar_sub",
                crate::BinOp::SubF => "by_scalar_subf",
            },
            Func::BinUnicast(op) => match op {
                crate::BinOp::Min => "unicast_min",
                crate::BinOp::Max => "unicast_max",
                crate::BinOp::Add => "unicast_add",
                crate::BinOp::Mul => "unicast_mul",
                crate::BinOp::Sub => "unicast_sub",
                crate::BinOp::SubF => "unicast_subf",
            },
        }
    }

    /// The kernel this host runs for a function and datum type. An unfilled pair is an error rather
    /// than a substitution: what a machine has no kernel for is what the matrix is there to show, and
    /// a caller that quietly computed something else would hide it.
    fn best_here(self, dt: DatumType) -> TractResult<&'static Routine> {
        native_best(self, dt).with_context(|| {
            format!("No {} kernel for {dt:?} on {:?}", self.name(), crate::isa::native())
        })
    }

    /// The f32 kernel this host runs for `func`.
    pub fn ew_f32(self) -> TractResult<Box<dyn ElementWise<f32>>> {
        match self.best_here(DatumType::F32)?.factory {
            RoutineFactory::F32(f) => Ok(f()),
            // `Routine::dt` reads the arm, so `best_for` already filtered the datum type. The
            // shape it cannot filter: asking a scalar-parameter routine for a plain one is a
            // caller's mistake, not a missing kernel.
            _ => bail!("{} is not a plain element-wise kernel", self.name()),
        }
    }

    /// The f16 kernel this host runs for `func`.
    pub fn ew_f16(self) -> TractResult<Box<dyn ElementWise<f16>>> {
        match self.best_here(DatumType::F16)?.factory {
            RoutineFactory::F16(f) => Ok(f()),
            _ => bail!("{} is not a plain element-wise kernel", self.name()),
        }
    }

    /// The f32 kernel this host runs for `func`, which takes a scalar parameter.
    pub fn ew_f32_param(self) -> TractResult<Box<dyn ElementWise<f32, f32>>> {
        match self.best_here(DatumType::F32)?.factory {
            RoutineFactory::F32Param(f) => Ok(f()),
            _ => bail!("{} is not a scalar-parameter kernel", self.name()),
        }
    }

    /// The f16 kernel this host runs for `func`, which takes a scalar parameter.
    pub fn ew_f16_param(self) -> TractResult<Box<dyn ElementWise<f16, f16>>> {
        match self.best_here(DatumType::F16)?.factory {
            RoutineFactory::F16Param(f) => Ok(f()),
            _ => bail!("{} is not a scalar-parameter kernel", self.name()),
        }
    }

    /// The f32 reduction this host runs for `func`.
    pub fn reduce_f32(self) -> TractResult<Box<dyn Reduce<f32>>> {
        match self.best_here(DatumType::F32)?.factory {
            RoutineFactory::F32Reduce(f) => Ok(f()),
            _ => bail!("{} is not a reduction", self.name()),
        }
    }

    /// The f16 reduction this host runs for `func`.
    pub fn reduce_f16(self) -> TractResult<Box<dyn Reduce<f16>>> {
        match self.best_here(DatumType::F16)?.factory {
            RoutineFactory::F16Reduce(f) => Ok(f()),
            _ => bail!("{} is not a reduction", self.name()),
        }
    }

    /// The f32 map-reduction this host runs for `func`.
    pub fn map_reduce_f32(self) -> TractResult<Box<dyn MapReduce<f32, f32>>> {
        match self.best_here(DatumType::F32)?.factory {
            RoutineFactory::F32MapReduce(f) => Ok(f()),
            _ => bail!("{} is not a map-reduction", self.name()),
        }
    }

    /// The binary kernel this host runs for `func` and datum type, `None` when it has none. Unlike
    /// the other accessors this one is optional rather than fallible: its callers rewrite a model
    /// only when a kernel exists, and having none is an ordinary answer rather than a failure.
    pub fn bin(self, dt: DatumType) -> Option<Box<crate::BinFn>> {
        match native_best(self, dt)?.factory {
            RoutineFactory::BinF32 { make, .. } | RoutineFactory::BinF16 { make, .. } => {
                Some(make())
            }
            _ => None,
        }
    }
}

/// Builds the kernel behind a descriptor. The arm is what says which datum type the descriptor
/// is for, so nothing repeats it as a field.
#[allow(clippy::type_complexity)]
pub enum RoutineFactory {
    F32(fn() -> Box<dyn ElementWise<f32>>),
    F16(fn() -> Box<dyn ElementWise<f16>>),
    /// A kernel taking one scalar of its own datum type, applied to every element.
    F32Param(fn() -> Box<dyn ElementWise<f32, f32>>),
    F16Param(fn() -> Box<dyn ElementWise<f16, f16>>),
    /// A kernel folding a slice to one value.
    F32Reduce(fn() -> Box<dyn Reduce<f32>>),
    F16Reduce(fn() -> Box<dyn Reduce<f16>>),
    /// A kernel rewriting a slice and folding it in the same pass.
    F32MapReduce(fn() -> Box<dyn MapReduce<f32, f32>>),
    /// A kernel that is a plain function rather than a boxed object, so its name is a field:
    /// there is no object to ask for one.
    RmsNormF32 {
        name: &'static str,
        run: fn(&mut [f32], f32),
    },
    /// A kernel built around a table, which the caller owns and hands over per op.
    LutU8 {
        name: fn() -> &'static str,
        make: fn(&[u8]) -> Box<dyn Lut>,
    },
    /// A binary kernel, over two tensor views. Type-erased like the views themselves, so the
    /// datum type is the arm and the name a field.
    BinF32 {
        name: fn() -> &'static str,
        make: fn() -> Box<crate::BinFn>,
    },
    BinF16 {
        name: fn() -> &'static str,
        make: fn() -> Box<crate::BinFn>,
    },
}

/// One kernel, enumerable uniformly on every host.
pub struct Routine {
    pub func: Func,
    /// Architecture the kernel is written for, `None` for generic Rust every target builds.
    pub arch: Option<Arch>,
    /// What the instruction set must offer for this kernel to run at all. Runnability only:
    /// a preference spelled here would also move the kernel in the matrix.
    pub isa: IsaReq,
    /// Where this kernel sits against its siblings, when the instruction set it needs does not
    /// say it. Zero for the kernels whose ladder step already ranks them correctly; a measured
    /// exception spells the steps it disagrees with, via [`crate::isa::peer_of`] or
    /// [`crate::isa::NEVER_PREFERRED`].
    pub boost: isize,
    /// Whether it reaches an f16 answer by converting a chunk to f32, running an f32 kernel over
    /// it and converting it back. Set by the declaration that writes the round trip, never by
    /// hand.
    pub round_trip: bool,
    pub factory: RoutineFactory,
}

inventory::collect!(Routine);

impl Routine {
    pub fn dt(&self) -> DatumType {
        match self.factory {
            RoutineFactory::F32(_)
            | RoutineFactory::F32Param(_)
            | RoutineFactory::F32Reduce(_)
            | RoutineFactory::F32MapReduce(_)
            | RoutineFactory::RmsNormF32 { .. } => DatumType::F32,
            RoutineFactory::F16(_) | RoutineFactory::F16Param(_) | RoutineFactory::F16Reduce(_) => {
                DatumType::F16
            }
            RoutineFactory::LutU8 { .. } => DatumType::U8,
            RoutineFactory::BinF32 { .. } => DatumType::F32,
            RoutineFactory::BinF16 { .. } => DatumType::F16,
        }
    }

    /// The kernel's own name. Read from the built object rather than declared, so it cannot
    /// disagree with the kernel it names. Building is metadata-only work and safe anywhere;
    /// running what it builds is not.
    pub fn name(&self) -> &'static str {
        match self.factory {
            RoutineFactory::F32(f) => f().name(),
            RoutineFactory::F16(f) => f().name(),
            RoutineFactory::F32Param(f) => f().name(),
            RoutineFactory::F16Param(f) => f().name(),
            RoutineFactory::F32Reduce(f) => f().name(),
            RoutineFactory::F16Reduce(f) => f().name(),
            RoutineFactory::F32MapReduce(f) => f().name(),
            RoutineFactory::RmsNormF32 { name, .. } => name,
            RoutineFactory::LutU8 { name, .. } => name(),
            RoutineFactory::BinF32 { name, .. } | RoutineFactory::BinF16 { name, .. } => name(),
        }
    }

    /// Whether `isa` describes a machine this kernel runs on: its architecture, and every
    /// feature it needs.
    pub fn runnable_on(&self, isa: &IsaSet) -> bool {
        self.arch.is_none_or(|a| Some(a) == isa.arch()) && self.isa.satisfied_by(*isa)
    }

    /// What this kernel is worth on a machine that can run it: its ladder step, plus whatever
    /// a measurement said the step gets wrong. An arch kernel always outranks a generic one,
    /// which is a different question and is compared before this.
    fn preference(&self) -> isize {
        self.isa.level() as isize * LEVEL_BOOST + self.boost
    }
}

/// Every routine this build compiled, whichever architecture it speaks for.
pub fn declared() -> impl Iterator<Item = &'static Routine> {
    inventory::iter::<Routine>()
}

/// The kernel `isa` would run for this function and datum type: an architecture kernel over a
/// generic one, then the most capable instruction set, then the name, which only settles ties
/// `inventory`'s link order would otherwise settle differently between builds. `None` when
/// nothing is declared for the pair at all.
pub fn best_for(func: Func, dt: DatumType, isa: &IsaSet) -> Option<&'static Routine> {
    declared()
        .filter(|r| r.func == func && r.dt() == dt && r.runnable_on(isa))
        .max_by_key(|r| (r.arch.is_some(), r.preference(), r.name()))
}

/// A cell of the matrix closed on purpose: on machines offering `isa`, this pair runs the kernel
/// named here, and a kernel written for that instruction set would not be an improvement.
///
/// Declared beside the kernels of the tree it speaks for, so a build carries it exactly when it
/// carries them, and it pins its winner by name: a machine that resolves the pair to something
/// else is a different verdict, and needs its own settlement or none.
pub struct Settled {
    /// The rung this speaks for, which is also the column the matrix draws it under.
    pub isa: Isa,
    pub func: Func,
    pub dt: DatumType,
    /// The winner this keeps, by the name the kernel answers with.
    pub kernel: &'static str,
    /// One line, in the terms someone would need to disagree with it.
    pub why: &'static str,
}

inventory::collect!(Settled);

impl Settled {
    /// Whether this speaks for `isa`: a machine of its architecture, offering what it names, and
    /// resolving its pair to the kernel it pins.
    pub fn covers(&self, isa: &IsaSet) -> bool {
        Some(self.isa.arch()) == isa.arch()
            && isa.has(self.isa)
            && best_for(self.func, self.dt, isa).is_some_and(|r| r.name() == self.kernel)
    }
}

/// Every settlement this build compiled, whichever architecture it speaks for.
pub fn settlements() -> impl Iterator<Item = &'static Settled> {
    inventory::iter::<Settled>()
}

/// Why what `isa` runs for this pair is the answer we mean, when a settlement says so. `None`
/// when nothing settles it, which is what leaves such a cell a gap.
pub fn settled_for(func: Func, dt: DatumType, isa: &IsaSet) -> Option<&'static Settled> {
    settlements().find(|s| s.func == func && s.dt == dt && s.covers(isa))
}

/// What a machine's answer for one pair amounts to, which is what the matrix colours and what
/// says whether a kernel is left to write.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Standing {
    /// Nothing this build declares serves the pair here.
    Missing,
    /// A kernel written for this machine's own rung of the ladder.
    Dedicated,
    /// Correct code that was not written for this machine: portable Rust, or an architecture
    /// kernel from a rung below it.
    Unspecialized,
    /// Portable f16 code, whose every operation converts to f32 and back: the right answer at a
    /// per-element cost no machine has to pay.
    Emulated,
    /// Whatever it runs, declared as the answer we mean. [`settled_for`] says why.
    Settled,
}

/// What `isa` amounts to for this pair before any settlement speaks for it, which is the question
/// a settlement answers.
fn unsettled(func: Func, dt: DatumType, isa: &IsaSet) -> Standing {
    let Some(routine) = best_for(func, dt, isa) else { return Standing::Missing };
    // Portable f16 arithmetic goes through `f16`'s operators, which convert to f32 and back
    // around every single one, whatever the machine underneath. A conversion-free portable f16
    // kernel -- a table, or bit work -- would need saying so here.
    if routine.arch.is_none() && dt == DatumType::F16 {
        Standing::Emulated
    } else if routine.round_trip && !isa.fp16_arithmetic() {
        Standing::Settled
    } else if routine.arch.is_some() && routine.isa.level() == isa.level() {
        Standing::Dedicated
    } else {
        Standing::Unspecialized
    }
}

/// What this machine's answer for the pair amounts to.
pub fn standing(func: Func, dt: DatumType, isa: &IsaSet) -> Standing {
    let answer = unsettled(func, dt, isa);
    let settleable = matches!(answer, Standing::Unspecialized | Standing::Emulated);
    if settleable && settled_for(func, dt, isa).is_some() { Standing::Settled } else { answer }
}

/// What every chunked round trip answers for on a machine with no f16 arithmetic: converting a
/// chunk, computing it in f32 and converting it back is the technique, not a compromise.
const NO_FP16_ARITHMETIC: &str = "no f16 arithmetic here: a chunk through an f32 kernel is it";

/// Why this cell is closed, when it is: what a settlement declared, or the standing answer every
/// f32 round trip carries on a machine that cannot compute in f16.
pub fn settled_why(func: Func, dt: DatumType, isa: &IsaSet) -> Option<&'static str> {
    if standing(func, dt, isa) != Standing::Settled {
        return None;
    }
    Some(settled_for(func, dt, isa).map_or(NO_FP16_ARITHMETIC, |settled| settled.why))
}

/// The kernel this host runs for a function and datum type, resolved once. Dispatch happens per
/// eval, so the scan over every declared routine runs at first use and the answers are kept in a
/// flat table nothing has to hash.
fn native_best(func: Func, dt: DatumType) -> Option<&'static Routine> {
    const SLOTS: usize = Func::ALL.len() * 3;
    static NATIVE: std::sync::OnceLock<[Option<&'static Routine>; SLOTS]> =
        std::sync::OnceLock::new();
    let dt_slot = match dt {
        DatumType::F32 => 0,
        DatumType::F16 => 1,
        DatumType::U8 => 2,
        _ => return None,
    };
    NATIVE.get_or_init(|| {
        let isa = crate::isa::native();
        let mut table = [None; SLOTS];
        for func in Func::ALL {
            for (dt_slot, dt) in [DatumType::F32, DatumType::F16, DatumType::U8].iter().enumerate()
            {
                table[func.slot() * 3 + dt_slot] = best_for(func, *dt, &isa);
            }
        }
        table
    })[func.slot() * 3 + dt_slot]
}

/// The fused row-wise RmsNorm this host runs: the row and the epsilon, in place.
pub fn rms_norm_f32() -> TractResult<fn(&mut [f32], f32)> {
    match Func::RmsNorm.best_here(DatumType::F32)?.factory {
        RoutineFactory::RmsNormF32 { run, .. } => Ok(run),
        _ => bail!("rms_norm is not a plain function"),
    }
}

/// The look-up table kernel this host runs, over the table the caller owns.
pub fn lut_u8(table: &[u8]) -> TractResult<Box<dyn Lut>> {
    match Func::Lut.best_here(DatumType::U8)?.factory {
        RoutineFactory::LutU8 { make, .. } => Ok(make(table)),
        _ => bail!("lut is not a table kernel"),
    }
}

/// File the descriptor of a kernel declared elsewhere, under the leading architecture ident the
/// `routine_*` declaration macros take, or with no ident at all for generic Rust every target
/// builds. Those macros end here; write it directly for a kernel no shape macro emits, and it
/// carries no test module of its own. The first argument names the factory arm, which is what
/// says the kernel's shape and datum type; `isa` is omitted for a kernel whose architecture
/// needs nothing extra to run it, `boost` for one its ladder step already ranks right.
macro_rules! submit_routine {
    (arm; $($rest:tt)*) => { submit_routine!(@ Some($crate::isa::Arch::Arm); $($rest)*); };
    (aarch64; $($rest:tt)*) => { submit_routine!(@ Some($crate::isa::Arch::Aarch64); $($rest)*); };
    (x86_64; $($rest:tt)*) => { submit_routine!(@ Some($crate::isa::Arch::X86_64); $($rest)*); };
    (riscv64; $($rest:tt)*) => { submit_routine!(@ Some($crate::isa::Arch::RiscV64); $($rest)*); };
    (wasm32; $($rest:tt)*) => { submit_routine!(@ Some($crate::isa::Arch::Wasm32Simd128); $($rest)*); };
    (generic; $($rest:tt)*) => { submit_routine!(@ None; $($rest)*); };

    ($factory:ident, $($rest:tt)*) => { submit_routine!(@ None; $factory, $($rest)*); };

    // One arm per kernel shape: the trait a kernel implements decides how it is built, and
    // nothing else here varies. The clauses are spelled out rather than forwarded as tokens
    // because a `path` fragment may only be followed by a comma.
    (@ $arch:expr; RmsNormF32, $func:ident, $name:literal, $run:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::RmsNormF32 { name: $name, run: $run }
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; BinF32, BinByScalar($op:ident), $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, BinByScalar($crate::BinOp::$op),
            $crate::routines::RoutineFactory::BinF32 {
                name: <$ker as $crate::element_wise::ElementWiseKer<f32, f32>>::name,
                make: <$ker as $crate::by_scalar::ByScalarKer<f32>>::bin,
            }
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; BinF16, BinByScalar($op:ident), $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, BinByScalar($crate::BinOp::$op),
            $crate::routines::RoutineFactory::BinF16 {
                name: <$ker as $crate::element_wise::ElementWiseKer<$crate::f16, $crate::f16>>::name,
                make: <$ker as $crate::by_scalar::ByScalarKer<$crate::f16>>::bin,
            }
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; BinF32, BinUnicast($op:ident), $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, BinUnicast($crate::BinOp::$op),
            $crate::routines::RoutineFactory::BinF32 {
                name: <$ker as $crate::unicast::UnicastKer<f32>>::name,
                make: <$ker as $crate::unicast::UnicastKer<f32>>::bin,
            }
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; BinF16, BinUnicast($op:ident), $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, BinUnicast($crate::BinOp::$op),
            $crate::routines::RoutineFactory::BinF16 {
                name: <$ker as $crate::unicast::UnicastKer<$crate::f16>>::name,
                make: <$ker as $crate::unicast::UnicastKer<$crate::f16>>::bin,
            }
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; LutU8, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::LutU8 {
                name: <$ker as $crate::lut::LutKer>::name,
                make: |table| $crate::routines::lut_of::<$ker>(table),
            }
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; F32Reduce, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::F32Reduce(|| $crate::routines::reduce_of::<$ker, _>())
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; F16Reduce, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::F16Reduce(|| $crate::routines::reduce_of::<$ker, _>())
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; F32MapReduce, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::F32MapReduce(
                || $crate::routines::map_reduce_of::<$ker, _>()
            )
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; $factory:ident, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))? $(, round_trip($round_trip:expr))?) => {
        submit_routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::$factory(
                || $crate::routines::factory_of::<$ker, _, _>()
            )
            $(, isa($($isa),+))? $(, boost($boost))? $(, round_trip($round_trip))?);
    };

    (@@ $arch:expr, $func:ident $(($($payload:tt)*))?, $factory:expr
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))? $(, round_trip($round_trip:expr))?) => {
        inventory::submit! {
            $crate::routines::Routine {
                func: $crate::routines::Func::$func $(($($payload)*))?,
                arch: $arch,
                isa: $crate::isa::IsaReq::ANY $(.needing(&[$($crate::isa::Isa::$isa),+]))?,
                boost: {
                    #[allow(unused_mut, unused_assignments)]
                    let mut boost = 0;
                    $(boost = $boost;)?
                    boost
                },
                round_trip: {
                    #[allow(unused_mut, unused_assignments)]
                    let mut round_trip = false;
                    $(round_trip = $round_trip;)?
                    round_trip
                },
                factory: $factory,
            }
        }
    };
}

/// Close one cell of the matrix on purpose: the machines offering `$isa` run `$kernel` for this
/// pair, and that is the answer we mean. Write it beside the kernels of the tree it speaks for,
/// and say in `$why` what a kernel written for the instruction set would have to beat.
macro_rules! settled {
    ($isa:ident, $func:ident $(($op:ident))?, $dt:ident, $kernel:ident, $why:literal) => {
        inventory::submit! {
            $crate::routines::Settled {
                isa: $crate::isa::Isa::$isa,
                func: $crate::routines::Func::$func $(($crate::BinOp::$op))?,
                dt: tract_data::prelude::DatumType::$dt,
                kernel: stringify!($kernel),
                why: $why,
            }
        }
    };
}

/// A look-up table kernel over `table`, as its factory arm wants it.
pub fn lut_of<K: crate::lut::LutKer + 'static>(table: &[u8]) -> Box<dyn Lut> {
    Box::new(crate::lut::LutImpl::<K>::new(table))
}

/// The `red()` of a reduction kernel, as its factory arm wants it.
pub fn reduce_of<K, T>() -> Box<dyn Reduce<T>>
where
    T: crate::LADatum,
    K: ReduceKer<T> + Clone,
{
    K::red()
}

/// The `red()` of a map-reduction kernel, as its factory arm wants it.
pub fn map_reduce_of<K, T>() -> Box<dyn MapReduce<T, T>>
where
    T: crate::LADatum,
    K: MapReduceKer<T, T> + Clone,
{
    K::red()
}

pub fn factory_of<K, T, P>() -> Box<dyn ElementWise<T, P>>
where
    T: crate::LADatum,
    P: Copy + Send + Sync + std::fmt::Debug + 'static + Default,
    K: ElementWiseKer<T, P> + Clone,
{
    K::ew()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dispatch table is indexed by [`Func::slot`], so every function must own one slot
    /// inside it, and the cache must answer what a fresh scan would.
    #[test]
    fn every_func_owns_a_slot() {
        let mut slots = std::collections::HashSet::new();
        for func in Func::ALL {
            assert!(func.slot() < Func::ALL.len(), "{} is out of the table", func.name());
            assert!(slots.insert(func.slot()), "{} shares a slot", func.name());
        }
        assert_eq!(slots.len(), Func::ALL.len());
        let isa = crate::isa::native();
        for func in Func::ALL {
            for dt in [DatumType::F32, DatumType::F16, DatumType::U8] {
                assert_eq!(
                    native_best(func, dt).map(|r| r.name()),
                    best_for(func, dt, &isa).map(|r| r.name()),
                    "{} {dt:?}",
                    func.name()
                );
            }
        }
    }

    /// A declared kernel no machine would ever choose is either a mistake -- the wrong instruction
    /// set, or a sibling that dominates it -- or one kept for its tests, which says so by
    /// declining. Nothing else is reachable, and nothing would notice.
    #[test]
    fn a_kernel_nothing_can_choose_says_so() {
        let mut chosen = std::collections::HashSet::new();
        for isa in IsaSet::every_ladder() {
            for func in Func::ALL {
                for dt in [DatumType::F32, DatumType::F16, DatumType::U8] {
                    if let Some(r) = best_for(func, dt, &isa) {
                        chosen.insert((func, dt, r.name()));
                    }
                }
            }
        }
        for r in declared() {
            assert!(
                chosen.contains(&(r.func, r.dt(), r.name())) || r.boost < 0,
                "{} {:?} {} can never be chosen, and does not decline",
                r.func.name(),
                r.dt(),
                r.name()
            );
        }
    }

    /// A pair with no kernel fails rather than falling back on something that computes a
    /// different thing. f16 erf is the standing example: no tree has one, and core builds a
    /// look-up table from the f32 kernel instead of asking for it.
    #[test]
    fn an_unfilled_pair_fails() {
        let err = Func::Erf.ew_f16().unwrap_err().to_string();
        assert!(err.starts_with("No erf kernel for F16 on "), "{err}");
        // No tree has an f16 minimum either, and no consumer asks for one.
        let err = Func::ReduceMin.reduce_f16().unwrap_err().to_string();
        assert!(err.starts_with("No reduce_min kernel for F16 on "), "{err}");
        // Asking for the wrong shape is a caller's mistake, and says so.
        let err = Func::ReduceMax.ew_f32().unwrap_err().to_string();
        assert_eq!(err, "reduce_max is not a plain element-wise kernel");
    }
    /// Whatever this machine declares, it can build and run: the registry is dispatch now, so a
    /// pair whose accessor fails is a cell nothing would notice was dead. Also holds `best_for`
    /// and the accessors to the same answer, they being two ways to ask one question.
    #[test]
    fn what_this_machine_declares_it_can_build() {
        let isa = crate::isa::native();
        for func in Func::ALL {
            for dt in [DatumType::F32, DatumType::F16, DatumType::U8] {
                let Some(routine) = best_for(func, dt, &isa) else { continue };
                let built = match routine.factory {
                    RoutineFactory::F32(_) => func.ew_f32().map(|k| k.name()),
                    RoutineFactory::F16(_) => func.ew_f16().map(|k| k.name()),
                    RoutineFactory::F32Param(_) => func.ew_f32_param().map(|k| k.name()),
                    RoutineFactory::F16Param(_) => func.ew_f16_param().map(|k| k.name()),
                    RoutineFactory::F32Reduce(_) => func.reduce_f32().map(|k| k.name()),
                    RoutineFactory::F16Reduce(_) => func.reduce_f16().map(|k| k.name()),
                    RoutineFactory::F32MapReduce(_) => func.map_reduce_f32().map(|k| k.name()),
                    RoutineFactory::RmsNormF32 { name, .. } => Ok(name),
                    RoutineFactory::LutU8 { name, .. } => lut_u8(&[0u8; 256]).map(|_| name()),
                    RoutineFactory::BinF32 { name, .. } | RoutineFactory::BinF16 { name, .. } => {
                        func.bin(dt).map(|_| name()).ok_or_else(|| format_err!("no bin kernel"))
                    }
                };
                assert_eq!(
                    built.map_err(|e| e.to_string()),
                    Ok(routine.name()),
                    "{} {dt:?}",
                    func.name()
                );
            }
        }
    }

    /// A settlement answers for a cell the matrix would otherwise read as an invitation, so one
    /// closing no such cell is either wrong about its machine or has outlived the kernel it
    /// pinned -- someone wrote the kernel it says nobody should.
    #[test]
    fn every_settlement_closes_a_cell() {
        for s in settlements() {
            assert!(
                IsaSet::every_ladder().any(|m| s.covers(&m)
                    && matches!(
                        unsettled(s.func, s.dt, &m),
                        Standing::Unspecialized | Standing::Emulated
                    )),
                "{} {:?} on {} settles nothing, {} being what it keeps",
                s.func.name(),
                s.dt,
                s.isa,
                s.kernel
            );
        }
    }

    /// Two settlements over one cell would each answer for it, and the matrix would print
    /// whichever `inventory` happened to link first.
    #[test]
    fn a_cell_is_settled_once() {
        for isa in IsaSet::every_ladder() {
            for func in Func::ALL {
                for dt in [DatumType::F32, DatumType::F16, DatumType::U8] {
                    let count = settlements()
                        .filter(|s| s.func == func && s.dt == dt && s.covers(&isa))
                        .count();
                    assert!(
                        count <= 1,
                        "{} {dt:?} on {isa:?} is settled {count} times",
                        func.name()
                    );
                }
            }
        }
    }
}
