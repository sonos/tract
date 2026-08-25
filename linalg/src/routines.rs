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
use crate::isa::{Arch, IsaReq, IsaSet, LEVEL_BOOST};
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
    /// Architecture the kernel is written for, `None` for portable Rust every target builds.
    pub arch: Option<Arch>,
    /// What the instruction set must offer for this kernel to run at all. Runnability only:
    /// a preference spelled here would also move the kernel in the matrix.
    pub isa: IsaReq,
    /// Where this kernel sits against its siblings, when the instruction set it needs does not
    /// say it. Zero for the kernels whose ladder step already ranks them correctly; a measured
    /// exception spells the steps it disagrees with, via [`crate::isa::peer_of`] or
    /// [`crate::isa::NEVER_PREFERRED`].
    pub boost: isize,
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
    /// a measurement said the step gets wrong. An arch kernel always outranks a portable one,
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
/// portable one, then the most capable instruction set, then the name, which only settles ties
/// `inventory`'s link order would otherwise settle differently between builds. `None` when
/// nothing is declared for the pair at all.
pub fn best_for(func: Func, dt: DatumType, isa: &IsaSet) -> Option<&'static Routine> {
    declared()
        .filter(|r| r.func == func && r.dt() == dt && r.runnable_on(isa))
        .max_by_key(|r| (r.arch.is_some(), r.preference(), r.name()))
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
/// `routine_*` declaration macros take, or with no ident at all for portable Rust every target
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
    (portable; $($rest:tt)*) => { submit_routine!(@ None; $($rest)*); };

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
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::$factory(
                || $crate::routines::factory_of::<$ker, _, _>()
            )
            $(, isa($($isa),+))? $(, boost($boost))?);
    };

    (@@ $arch:expr, $func:ident $(($($payload:tt)*))?, $factory:expr
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
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
                factory: $factory,
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

    /// What each x86 generation runs, for the two f16 kernels the AVX-512_FP16 tier has: the
    /// native hardswish wins on its ladder step, and the native leaky_relu loses every tie to the
    /// f32 round-trip it was measured against, by the boost it declares. Both are the same
    /// instruction set, so nothing but that declaration separates them.
    #[cfg(any(target_arch = "x86_64", feature = "foreign-inventory"))]
    #[test]
    fn what_the_x86_ladder_runs() {
        for (level, func, dt, expected) in [
            (1, Func::MulByScalar, DatumType::F32, "x86_64_avx_f32_mul_by_scalar_32n"),
            (1, Func::LeakyRelu, DatumType::F32, "generic"),
            (1, Func::Sigmoid, DatumType::F32, "avx_sigmoid_f32"),
            (2, Func::Sigmoid, DatumType::F32, "fma_sigmoid_f32"),
            (3, Func::Sigmoid, DatumType::F32, "avx512_sigmoid_f32"),
            (3, Func::LeakyRelu, DatumType::F32, "x86_64_avx512_leaky_relu_f32_64n"),
            (3, Func::LeakyRelu, DatumType::F16, "x86_64_avx512_leaky_relu_f16_64n"),
            (3, Func::Hardswish, DatumType::F16, "x86_64_avx512_hardswish_f16_64n"),
            (4, Func::LeakyRelu, DatumType::F16, "x86_64_avx512_leaky_relu_f16_64n"),
            (4, Func::Hardswish, DatumType::F16, "x86_64_avx512fp16_hardswish_f16_128n"),
            // The two reductions x86 has are plain AVX whatever their names say, and nothing
            // here sums: that column is portable on every x86 part.
            (1, Func::ReduceMax, DatumType::F32, "x86_64_fma_max_f32_32n"),
            (1, Func::ReduceMin, DatumType::F32, "x86_64_fma_min_f32_32n"),
            (1, Func::ReduceSum, DatumType::F32, "SSum4"),
            (3, Func::ReduceMax, DatumType::F32, "x86_64_avx512_max_f32_64n"),
            (1, Func::Softmax2, DatumType::F32, "SSoftMaxL2Accurate"),
            (2, Func::Softmax2, DatumType::F32, "x86_64_fma_softmax2_f32_32n"),
            (3, Func::Softmax2, DatumType::F32, "x86_64_avx512_softmax2_f32_64n"),
            (1, Func::RmsNorm, DatumType::F32, "generic"),
            (3, Func::RmsNorm, DatumType::F32, "x86_64_avx512_rms_norm_f32"),
            // No binary kernel at any x86 tier: both layouts are portable everywhere.
            (3, Func::BinByScalar(crate::BinOp::Mul), DatumType::F32, "SMulByScalar4"),
            (3, Func::BinUnicast(crate::BinOp::Add), DatumType::F32, "SUnicastAdd4"),
        ] {
            let isa = IsaSet::ladder(Arch::X86_64, level);
            assert_eq!(
                best_for(func, dt, &isa).map(|r| r.name()),
                Some(expected),
                "{} {dt:?} on the x86 ladder at level {level}",
                func.name()
            );
        }
    }

    /// What each aarch64 generation runs. Which kernel a machine picks is a function of its
    /// instruction set and not of the machine asking, so these answers are checked from any host
    /// that compiled the tree -- including for hardware nobody here has.
    #[cfg(any(target_arch = "aarch64", feature = "foreign-inventory"))]
    #[test]
    fn what_the_aarch64_ladder_runs() {
        let neon = IsaSet::of_arch(Arch::Aarch64);
        for (func, dt, expected) in [
            (Func::Sigmoid, DatumType::F32, Some("arm64simd_sigmoid_f32_4n")),
            (Func::Tanh, DatumType::F32, Some("arm64simd_tanh_f32_4n")),
            (Func::Silu, DatumType::F32, Some("arm64simd_silu_f32_4n_fused")),
            (Func::Gelu, DatumType::F32, Some("arm64simd_gelu_f32_4n_fused")),
            (Func::Hardswish, DatumType::F32, Some("arm64simd_hardswish_f32_8n")),
            (Func::Erf, DatumType::F32, Some("generic")),
            (Func::Sigmoid, DatumType::F16, Some("arm64simd_sigmoid_f16_4n")),
            (Func::Tanh, DatumType::F16, Some("arm64simd_tanh_f16_4n")),
            (Func::Silu, DatumType::F16, Some("arm64simd_silu_f16_lut_8n")),
            (Func::Gelu, DatumType::F16, Some("generic")),
            (Func::Hardswish, DatumType::F16, Some("generic")),
            (Func::Erf, DatumType::F16, None),
            (Func::LeakyRelu, DatumType::F32, Some("arm64simd_leaky_relu_f32_8n")),
            (Func::MulByScalar, DatumType::F32, Some("arm64simd_mul_by_scalar_f32_16n")),
            (Func::LeakyRelu, DatumType::F16, Some("generic")),
            (Func::MulByScalar, DatumType::F16, Some("HMulByScalar8")),
            (Func::ReduceMax, DatumType::F32, Some("arm64simd_max_f32_16n")),
            (Func::ReduceMin, DatumType::F32, Some("arm64simd_min_f32_16n")),
            (Func::ReduceSum, DatumType::F32, Some("arm64simd_sum_f32_16n")),
            (Func::RmsNorm, DatumType::F32, Some("arm64simd_rms_norm_f32")),
            // No arm64 softmax2 kernel, and no f16 reduction below fp16.
            (Func::Softmax2, DatumType::F32, Some("SSoftMaxL2Accurate")),
            (Func::ReduceMax, DatumType::F16, Some("HMax8")),
            (
                Func::BinByScalar(crate::BinOp::Mul),
                DatumType::F32,
                Some("arm64simd_mul_by_scalar_f32_16n"),
            ),
            (
                Func::BinUnicast(crate::BinOp::Add),
                DatumType::F32,
                Some("arm64simd_unicast_add_f32_16n"),
            ),
            // Nothing computes an f16 binary op without the fp16 tree: the portable kernels for
            // these were deleted, nothing having been able to reach them.
            (Func::BinByScalar(crate::BinOp::Mul), DatumType::F16, None),
            (Func::BinUnicast(crate::BinOp::Add), DatumType::F16, None),
        ] {
            assert_eq!(
                best_for(func, dt, &neon).map(|r| r.name()),
                expected,
                "{} {dt:?} on plain aarch64",
                func.name()
            );
        }
        // The fp16 tree speaks for the two functions it has kernels for and no more: the NEON
        // look-up-table silu stays ahead on fp16 hardware, having no fp16 rival at all.
        #[cfg(not(feature = "no_fp16"))]
        {
            let fp16 = neon.with(crate::isa::Isa::Aarch64Fp16);
            for (func, dt, expected) in [
                (Func::Sigmoid, DatumType::F16, "arm64fp16_sigmoid_f16_8n"),
                (Func::Tanh, DatumType::F16, "arm64fp16_tanh_f16_8n"),
                (Func::Silu, DatumType::F16, "arm64simd_silu_f16_lut_8n"),
                (Func::Sigmoid, DatumType::F32, "arm64simd_sigmoid_f32_4n"),
                (Func::LeakyRelu, DatumType::F16, "arm64fp16_leaky_relu_f16_16n"),
                (Func::MulByScalar, DatumType::F16, "arm64fp16_mul_by_scalar_f16_32n"),
                (Func::ReduceMax, DatumType::F16, "arm64fp16_max_f16_32n"),
                (Func::ReduceSum, DatumType::F16, "arm64fp16_sum_f16_32n"),
                (
                    Func::BinByScalar(crate::BinOp::Mul),
                    DatumType::F16,
                    "arm64fp16_mul_by_scalar_f16_32n",
                ),
                (
                    Func::BinUnicast(crate::BinOp::Add),
                    DatumType::F16,
                    "arm64fp16_unicast_add_f16_32n",
                ),
            ] {
                assert_eq!(
                    best_for(func, dt, &fp16).map(|r| r.name()),
                    Some(expected),
                    "{} {dt:?} on aarch64+fp16",
                    func.name()
                );
            }
        }
    }

    /// What wasm runs. Its two steps are a build question rather than a probe, but they rank
    /// like any other ladder: the relaxed kernels, which have the fused multiply-add, sit above a
    /// baseline simd128 build that has no activation kernel of its own at all.
    #[cfg(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        feature = "foreign-inventory"
    ))]
    #[test]
    fn what_the_wasm_ladder_runs() {
        let simd128 = IsaSet::ladder(Arch::Wasm32Simd128, 0);
        let relaxed = IsaSet::ladder(Arch::Wasm32Simd128, 1);
        for (func, dt, expected) in [
            (Func::ReduceMax, DatumType::F32, "wasm_max_f32_32n"),
            (Func::ReduceMin, DatumType::F32, "wasm_min_f32_32n"),
            (Func::ReduceSum, DatumType::F32, "wasm_sum_f32_32n"),
            (Func::ReduceMax, DatumType::F16, "wasm_max_f16_32n"),
            (Func::ReduceSum, DatumType::F16, "wasm_sum_f16_32n"),
            (Func::RmsNorm, DatumType::F32, "wasm_rms_norm_f32"),
        ] {
            assert_eq!(
                best_for(func, dt, &simd128).map(|r| r.name()),
                Some(expected),
                "{} {dt:?} on plain simd128",
                func.name()
            );
        }
        for func in [Func::Sigmoid, Func::Tanh] {
            assert_eq!(
                best_for(func, DatumType::F32, &simd128).map(|r| r.name()),
                Some("generic"),
                "{} f32 on plain simd128",
                func.name()
            );
            assert_eq!(
                best_for(func, DatumType::F32, &relaxed).map(|r| r.name()),
                Some("wasm_relaxed_simd"),
                "{} f32 on wasm+relaxed-simd",
                func.name()
            );
        }
    }
    /// What armv7 runs, with and without NEON: three f32 kernels, and the portable floor for
    /// everything else. The f16 side has no armv7 kernel at all.
    #[cfg(any(target_arch = "arm", feature = "foreign-inventory"))]
    #[test]
    fn what_the_armv7_ladder_runs() {
        let vfp = IsaSet::of_arch(Arch::Arm);
        let neon = vfp.with(crate::isa::Isa::ArmNeon);
        for (func, expected) in [
            (Func::Sigmoid, "armv7neon_sigmoid_f32_4n"),
            (Func::Tanh, "armv7neon_tanh_f32_4n"),
            (Func::Silu, "armv7neon_silu_f32_4n"),
        ] {
            assert_eq!(
                best_for(func, DatumType::F32, &vfp).map(|r| r.name()),
                Some("generic"),
                "{} f32 on armv7 without neon",
                func.name()
            );
            assert_eq!(
                best_for(func, DatumType::F32, &neon).map(|r| r.name()),
                Some(expected),
                "{} f32 on armv7+neon",
                func.name()
            );
            assert_eq!(
                best_for(func, DatumType::F16, &neon).map(|r| r.name()),
                Some("generic"),
                "{} f16 on armv7+neon",
                func.name()
            );
        }
    }
}
