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
    HardSwish,
    LeakyRelu,
    MulByScalar,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    Softmax2,
    RmsNorm,
}

impl Func {
    pub const ALL: [Func; 13] = [
        Func::Sigmoid,
        Func::Tanh,
        Func::Silu,
        Func::Gelu,
        Func::Erf,
        Func::HardSwish,
        Func::ReduceMax,
        Func::ReduceMin,
        Func::ReduceSum,
        Func::Softmax2,
        Func::RmsNorm,
        Func::LeakyRelu,
        Func::MulByScalar,
    ];

    /// The name the matrix and the logs use.
    pub fn name(&self) -> &'static str {
        match self {
            Func::Sigmoid => "sigmoid",
            Func::Tanh => "tanh",
            Func::Silu => "silu",
            Func::Gelu => "gelu",
            Func::Erf => "erf",
            Func::HardSwish => "hardswish",
            Func::LeakyRelu => "leaky_relu",
            Func::MulByScalar => "mul_by_scalar",
            Func::ReduceMax => "reduce_max",
            Func::ReduceMin => "reduce_min",
            Func::ReduceSum => "reduce_sum",
            Func::Softmax2 => "softmax2",
            Func::RmsNorm => "rms_norm",
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

/// The kernel this host runs for a function and datum type. An unfilled pair is an error rather
/// than a substitution: what a machine has no kernel for is what the matrix is there to show, and
/// a caller that quietly computed something else would hide it.
fn best_here(func: Func, dt: DatumType) -> TractResult<&'static Routine> {
    let isa = crate::isa::native();
    best_for(func, dt, &isa)
        .with_context(|| format!("No {} kernel for {dt:?} on {isa:?}", func.name()))
}

/// The f32 kernel this host runs for `func`.
pub fn ew_f32(func: Func) -> TractResult<Box<dyn ElementWise<f32>>> {
    match best_here(func, DatumType::F32)?.factory {
        RoutineFactory::F32(f) => Ok(f()),
        // `Routine::dt` reads the arm, so `best_for` already filtered the datum type. The
        // shape it cannot filter: asking a scalar-parameter routine for a plain one is a
        // caller's mistake, not a missing kernel.
        _ => bail!("{} is not a plain element-wise kernel", func.name()),
    }
}

/// The f16 kernel this host runs for `func`.
pub fn ew_f16(func: Func) -> TractResult<Box<dyn ElementWise<f16>>> {
    match best_here(func, DatumType::F16)?.factory {
        RoutineFactory::F16(f) => Ok(f()),
        _ => bail!("{} is not a plain element-wise kernel", func.name()),
    }
}

/// The f32 kernel this host runs for `func`, which takes a scalar parameter.
pub fn ew_f32_param(func: Func) -> TractResult<Box<dyn ElementWise<f32, f32>>> {
    match best_here(func, DatumType::F32)?.factory {
        RoutineFactory::F32Param(f) => Ok(f()),
        _ => bail!("{} is not a scalar-parameter kernel", func.name()),
    }
}

/// The f16 kernel this host runs for `func`, which takes a scalar parameter.
pub fn ew_f16_param(func: Func) -> TractResult<Box<dyn ElementWise<f16, f16>>> {
    match best_here(func, DatumType::F16)?.factory {
        RoutineFactory::F16Param(f) => Ok(f()),
        _ => bail!("{} is not a scalar-parameter kernel", func.name()),
    }
}

/// The f32 reduction this host runs for `func`.
pub fn reduce_f32(func: Func) -> TractResult<Box<dyn Reduce<f32>>> {
    match best_here(func, DatumType::F32)?.factory {
        RoutineFactory::F32Reduce(f) => Ok(f()),
        _ => bail!("{} is not a reduction", func.name()),
    }
}

/// The f16 reduction this host runs for `func`.
pub fn reduce_f16(func: Func) -> TractResult<Box<dyn Reduce<f16>>> {
    match best_here(func, DatumType::F16)?.factory {
        RoutineFactory::F16Reduce(f) => Ok(f()),
        _ => bail!("{} is not a reduction", func.name()),
    }
}

/// The f32 map-reduction this host runs for `func`.
pub fn map_reduce_f32(func: Func) -> TractResult<Box<dyn MapReduce<f32, f32>>> {
    match best_here(func, DatumType::F32)?.factory {
        RoutineFactory::F32MapReduce(f) => Ok(f()),
        _ => bail!("{} is not a map-reduction", func.name()),
    }
}

/// The fused row-wise RmsNorm this host runs: the row and the epsilon, in place.
pub fn rms_norm_f32() -> TractResult<fn(&mut [f32], f32)> {
    match best_here(Func::RmsNorm, DatumType::F32)?.factory {
        RoutineFactory::RmsNormF32 { run, .. } => Ok(run),
        _ => bail!("rms_norm is not a plain function"),
    }
}

/// Declare one kernel, under the leading architecture ident the kernel-declaration macros take,
/// or with no ident at all for portable Rust every target builds. The first argument names the
/// factory arm, which is what says the kernel's shape and datum type; `isa` is omitted for a
/// kernel whose architecture needs nothing extra to run it, `boost` for one its ladder step
/// already ranks right.
macro_rules! routine {
    (arm; $($rest:tt)*) => { routine!(@ Some($crate::isa::Arch::Arm); $($rest)*); };
    (aarch64; $($rest:tt)*) => { routine!(@ Some($crate::isa::Arch::Aarch64); $($rest)*); };
    (x86_64; $($rest:tt)*) => { routine!(@ Some($crate::isa::Arch::X86_64); $($rest)*); };
    (riscv64; $($rest:tt)*) => { routine!(@ Some($crate::isa::Arch::RiscV64); $($rest)*); };
    (wasm32; $($rest:tt)*) => { routine!(@ Some($crate::isa::Arch::Wasm32Simd128); $($rest)*); };

    ($factory:ident, $($rest:tt)*) => { routine!(@ None; $factory, $($rest)*); };

    // One arm per kernel shape: the trait a kernel implements decides how it is built, and
    // nothing else here varies. The clauses are spelled out rather than forwarded as tokens
    // because a `path` fragment may only be followed by a comma.
    (@ $arch:expr; RmsNormF32, $func:ident, $name:literal, $run:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::RmsNormF32 { name: $name, run: $run }
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; F32Reduce, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::F32Reduce(|| $crate::routines::reduce_of::<$ker, _>())
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; F16Reduce, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::F16Reduce(|| $crate::routines::reduce_of::<$ker, _>())
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; F32MapReduce, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::F32MapReduce(
                || $crate::routines::map_reduce_of::<$ker, _>()
            )
            $(, isa($($isa),+))? $(, boost($boost))?);
    };
    (@ $arch:expr; $factory:ident, $func:ident, $ker:path
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        routine!(@@ $arch, $func,
            $crate::routines::RoutineFactory::$factory(
                || $crate::routines::factory_of::<$ker, _, _>()
            )
            $(, isa($($isa),+))? $(, boost($boost))?);
    };

    (@@ $arch:expr, $func:ident, $factory:expr
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        inventory::submit! {
            $crate::routines::Routine {
                func: $crate::routines::Func::$func,
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

    /// A pair with no kernel fails rather than falling back on something that computes a
    /// different thing. f16 erf is the standing example: no tree has one, and core builds a
    /// look-up table from the f32 kernel instead of asking for it.
    #[test]
    fn an_unfilled_pair_fails() {
        let err = ew_f16(Func::Erf).unwrap_err().to_string();
        assert!(err.starts_with("No erf kernel for F16 on "), "{err}");
        // No tree has an f16 minimum either, and no consumer asks for one.
        let err = reduce_f16(Func::ReduceMin).unwrap_err().to_string();
        assert!(err.starts_with("No reduce_min kernel for F16 on "), "{err}");
        // Asking for the wrong shape is a caller's mistake, and says so.
        let err = ew_f32(Func::ReduceMax).unwrap_err().to_string();
        assert_eq!(err, "reduce_max is not a plain element-wise kernel");
    }
    /// The registry picks exactly what `plug` installed, on whatever machine the test runs. This
    /// is the whole safety argument for the flip: the two mechanisms answer the same question, so
    /// the imperative one can go.
    #[test]
    fn picks_what_plug_installed() {
        let ops = crate::ops();
        for (func, plugged) in [
            (Func::Sigmoid, format!("{:?}", (ops.sigmoid_f32)())),
            (Func::Tanh, format!("{:?}", (ops.tanh_f32)())),
            (Func::Silu, format!("{:?}", (ops.silu_f32)())),
            (Func::Gelu, format!("{:?}", (ops.gelu_f32)())),
            (Func::Erf, format!("{:?}", (ops.erf_f32)())),
            (Func::HardSwish, format!("{:?}", (ops.hardswish_f32)())),
        ] {
            assert_eq!(
                ew_f32(func).map(|k| format!("{k:?}")).ok(),
                Some(plugged),
                "{} f32 disagrees with plug",
                func.name()
            );
        }
        for (func, plugged) in [
            (Func::LeakyRelu, format!("{:?}", (ops.leaky_relu_f32)())),
            (Func::MulByScalar, format!("{:?}", (ops.mul_by_scalar_f32)())),
        ] {
            assert_eq!(
                ew_f32_param(func).map(|k| format!("{k:?}")).ok(),
                Some(plugged),
                "{} f32 disagrees with plug",
                func.name()
            );
        }
        for (func, plugged) in [
            (Func::LeakyRelu, format!("{:?}", (ops.leaky_relu_f16)())),
            (Func::MulByScalar, format!("{:?}", (ops.mul_by_scalar_f16)())),
        ] {
            assert_eq!(
                ew_f16_param(func).map(|k| format!("{k:?}")).ok(),
                Some(plugged),
                "{} f16 disagrees with plug",
                func.name()
            );
        }
        for (func, plugged) in [
            (Func::ReduceMax, format!("{:?}", (ops.max_f32)())),
            (Func::ReduceMin, format!("{:?}", (ops.min_f32)())),
            (Func::ReduceSum, format!("{:?}", (ops.sum_f32)())),
        ] {
            assert_eq!(
                reduce_f32(func).map(|k| format!("{k:?}")).ok(),
                Some(plugged),
                "{} f32 disagrees with plug",
                func.name()
            );
        }
        for (func, plugged) in [
            (Func::ReduceMax, format!("{:?}", (ops.max_f16)())),
            (Func::ReduceSum, format!("{:?}", (ops.sum_f16)())),
        ] {
            assert_eq!(
                reduce_f16(func).map(|k| format!("{k:?}")).ok(),
                Some(plugged),
                "{} f16 disagrees with plug",
                func.name()
            );
        }
        assert_eq!(
            map_reduce_f32(Func::Softmax2).map(|k| format!("{k:?}")).ok(),
            Some(format!("{:?}", (ops.softmax2_f32)())),
            "softmax2 disagrees with plug"
        );
        // RmsNorm is a plain function, so the comparison is what it computes: there is no object
        // whose identity could be read instead.
        let mut from_registry = [0.5f32, -1.5, 2.0, 3.25, -0.75, 8.0, 1.0, -2.5];
        let mut from_plug = from_registry;
        (rms_norm_f32().unwrap())(&mut from_registry, 1e-6);
        (ops.rms_norm_f32)(&mut from_plug, 1e-6);
        assert_eq!(from_registry, from_plug, "rms_norm disagrees with plug");
        for (func, plugged) in [
            (Func::Sigmoid, format!("{:?}", (ops.sigmoid_f16)())),
            (Func::Tanh, format!("{:?}", (ops.tanh_f16)())),
            (Func::Silu, format!("{:?}", (ops.silu_f16)())),
            (Func::Gelu, format!("{:?}", (ops.gelu_f16)())),
            (Func::HardSwish, format!("{:?}", (ops.hardswish_f16)())),
        ] {
            assert_eq!(
                ew_f16(func).map(|k| format!("{k:?}")).ok(),
                Some(plugged),
                "{} f16 disagrees with plug",
                func.name()
            );
        }
    }
    /// Two kernels a machine can run must not come out equal: [`best_for`] would then pick by
    /// `inventory`'s link order, which is not stable across builds, and dispatch would be a
    /// coin toss the tests could not see.
    #[test]
    fn nothing_ties_for_the_best() {
        for isa in IsaSet::every_ladder() {
            for func in Func::ALL {
                for dt in [DatumType::F32, DatumType::F16] {
                    let mut best: Vec<&Routine> = declared()
                        .filter(|r| r.func == func && r.dt() == dt && r.runnable_on(&isa))
                        .collect();
                    let Some(top) = best.iter().map(|r| (r.arch.is_some(), r.preference())).max()
                    else {
                        continue;
                    };
                    best.retain(|r| (r.arch.is_some(), r.preference()) == top);
                    assert!(
                        best.len() == 1,
                        "{} {dt:?} on {isa:?} ties between {:?}",
                        func.name(),
                        best.iter().map(|r| r.name()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    /// What each x86 generation runs, for the two f16 kernels the AVX-512_FP16 tier has: the
    /// native hardswish wins on its ladder step, and the native leaky_relu loses every tie to the
    /// f32 round-trip it was measured against, by the boost it declares. Both are the same
    /// instruction set, so nothing but that declaration separates them.
    #[cfg(any(target_arch = "x86_64", feature = "foreign-inventory"))]
    #[test]
    fn the_x86_ladder_picks_what_its_plug_installs() {
        for (level, func, dt, expected) in [
            (1, Func::MulByScalar, DatumType::F32, "x86_64_avx_f32_mul_by_scalar_32n"),
            (1, Func::LeakyRelu, DatumType::F32, "generic"),
            (1, Func::Sigmoid, DatumType::F32, "avx_sigmoid_f32"),
            (2, Func::Sigmoid, DatumType::F32, "fma_sigmoid_f32"),
            (3, Func::Sigmoid, DatumType::F32, "avx512_sigmoid_f32"),
            (3, Func::LeakyRelu, DatumType::F32, "x86_64_avx512_leaky_relu_f32_64n"),
            (3, Func::LeakyRelu, DatumType::F16, "x86_64_avx512_leaky_relu_f16_64n"),
            (3, Func::HardSwish, DatumType::F16, "x86_64_avx512_hardswish_f16_64n"),
            (4, Func::LeakyRelu, DatumType::F16, "x86_64_avx512_leaky_relu_f16_64n"),
            (4, Func::HardSwish, DatumType::F16, "x86_64_avx512fp16_hardswish_f16_128n"),
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
    fn the_aarch64_ladder_picks_what_its_plug_installs() {
        let neon = IsaSet::of_arch(Arch::Aarch64);
        for (func, dt, expected) in [
            (Func::Sigmoid, DatumType::F32, Some("arm64simd_sigmoid_f32_4n")),
            (Func::Tanh, DatumType::F32, Some("arm64simd_tanh_f32_4n")),
            (Func::Silu, DatumType::F32, Some("arm64simd_silu_f32_4n_fused")),
            (Func::Gelu, DatumType::F32, Some("arm64simd_gelu_f32_4n_fused")),
            (Func::HardSwish, DatumType::F32, Some("arm64simd_hardswish_f32_8n")),
            (Func::Erf, DatumType::F32, Some("generic")),
            (Func::Sigmoid, DatumType::F16, Some("arm64simd_sigmoid_f16_4n")),
            (Func::Tanh, DatumType::F16, Some("arm64simd_tanh_f16_4n")),
            (Func::Silu, DatumType::F16, Some("arm64simd_silu_f16_lut_8n")),
            (Func::Gelu, DatumType::F16, Some("generic")),
            (Func::HardSwish, DatumType::F16, Some("generic")),
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
    fn the_wasm_ladder_picks_what_its_plug_installs() {
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
    fn the_armv7_ladder_picks_what_its_plug_installs() {
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
