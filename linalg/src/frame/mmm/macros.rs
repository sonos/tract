// An mmm kernel whose inner loop is asm. The leading ident names the arch that asm was
// written for: the extern symbol is emitted only in builds carrying that arch's
// instructions, replaced elsewhere by a bail stub, so the tree links everywhere. The
// `MmmRoutine` submit is unconditional (pure data); the kernel itself records whether this
// build assembled it.
macro_rules! MMMExternKernel {
    // Map the arch ident to its inventory label and to the cfg predicate for "built here".
    (arm; $($rest:tt)*)     => { MMMExternKernel!(@ Some($crate::isa::Arch::Arm), target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { MMMExternKernel!(@ Some($crate::isa::Arch::Aarch64), target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*)  => { MMMExternKernel!(@ Some($crate::isa::Arch::X86_64), target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { MMMExternKernel!(@ Some($crate::isa::Arch::RiscV64), target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*)  => { MMMExternKernel!(@ Some($crate::isa::Arch::Wasm32Simd128), all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $target:expr, $built:meta;
        $func:ident<$ti:ident>($mr:expr, $nr:expr)
        $($rest:tt)*
    ) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use super::*;
                #[allow(unused_imports)]
                use crate::frame::mmm::*;

                #[cfg($built)]
                extern_kernel!(fn $func(op: *const FusedKerSpec<$ti>) -> isize);

                #[cfg(not($built))]
                #[allow(dead_code)]
                pub unsafe fn $func(_op: *const FusedKerSpec<$ti>) -> isize {
                    panic!(concat!(stringify!($func), ": mmm kernel not built for this target arch"))
                }

                #[inline]
                pub unsafe fn rusty(op: &[FusedKerSpec<$ti>]) -> isize {
                    unsafe { $func(op.as_ptr()) }
                }
            }

            MMMKernel!([<sys_ $func>]::rusty as $func<$ti>($mr, $nr)
                built(cfg!($built)) arch($target) $($rest)*);

            inventory::submit! {
                $crate::mmm_routines::MmmRoutine { make: || $func.mmm() }
            }
        }
    };
}

// An mmm kernel whose inner loop is Rust — intrinsics, or a C extern block like SVE's.
// Given a leading arch ident it also registers an introspection descriptor; whether this build
// compiled the kernel is the kernel's own `built`, and the kernel fn is gated by its module.
// Without the ident it is a generic kernel, built everywhere.
macro_rules! MMMRustKernel {
    // Generic Rust, built and dispatchable everywhere.
    (generic; $($rest:tt)*) => { MMMRustKernel!(@ None, all(); $($rest)*); };
    (arm; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::isa::Arch::Arm), target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::isa::Arch::Aarch64), target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::isa::Arch::X86_64), target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::isa::Arch::RiscV64), target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::isa::Arch::Wasm32Simd128), all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $target:expr, $built:meta; $func:path => $id:ident<$ti:ident>($mr:expr, $nr:expr) $($rest:tt)*) => {
        MMMRustKernel!($func => $id<$ti>($mr, $nr) built(cfg!($built)) arch($target) $($rest)*);
        inventory::submit! {
            $crate::mmm_routines::MmmRoutine { make: || $id.mmm() }
        }
    };

    (       $func: path =>
            $id:ident<$ti:ident>($mr: expr, $nr: expr)
            $(built($built_here:expr))?
            $(arch($arch:expr))?
            $(@($align_a:expr, $align_b:expr))?
            $(isa($($isa:ident),+))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(emulated($emulated:expr))?
            $(boost($boost:expr))?
            $(store($($store:ty),*))?
            $(row_major_store($rms:expr))?
     ) => {
        paste! {
            mod [<sys_ $id>] {
                #[allow(unused_imports)]
                use crate::frame::mmm::*;
                use super::*;
                #[inline]
                pub unsafe fn rusty(op: &[FusedKerSpec<$ti>]) -> isize {
                    unsafe { $func(op.as_ptr()) }
                }
            }
            MMMKernel!([<sys_$id>]::rusty as $id<$ti>($mr, $nr)
                $(built($built_here))?
                $(arch($arch))?
                $(@($align_a, $align_b))?
                $(isa($($isa),+))?
                $(can_fuse($can_fuse))?
                $(packing[$pnum] = $pid => $packing;)*
                $(emulated($emulated))?
                $(boost($boost))?
                $(store($($store),*))?
                $(row_major_store($rms))?
            );
        }
    }
}

macro_rules! MMMKernel {
    (
            $func: path as
            $id:ident<$ti:ident>($mr: expr, $nr: expr)
            $(built($built_here:expr))?
            $(arch($arch:expr))?
            $(@($align_a:expr, $align_b:expr))?
            $(isa($($isa:ident),+))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(emulated($emulated:expr))?
            $(boost($boost:expr))?
            $(store($($store:ty),*))?
            $(row_major_store($rms:expr))?
     ) => {
        paste! {
            lazy_static::lazy_static! {
                pub static ref $id: $crate::mmm::DynKernel<$mr, $nr, $ti> = {
                    use $crate::mmm::DynKernel;
                    #[allow(unused_imports)]
                    use tract_data::prelude::*;
                    use $crate::pack::Packing;
                    #[allow(unused_mut)]
                    let (mut packing_a, mut packing_b) = ($ti::packing($mr), $ti::packing($nr));
                    $(
                        packing_a = packing_a.align($align_a);
                        packing_b = packing_b.align($align_b);
                    )?
                    #[allow(unused_mut)]
                    let mut k = DynKernel::<$mr, $nr, $ti>::new(stringify!($id), $func, packing_a, packing_b);
                    $(k.built = $built_here;)?
                    $(k.arch = $arch;)?
                    k = k.with_isa(
                        $crate::isa::IsaReq::ANY
                            $(.needing(&[$($crate::isa::Isa::$isa),+]))?
                    );
                    $(
                        assert!(k.packings.len() == $pnum);
                        let f: fn(DynKernel<$mr, $nr, $ti>) -> DynKernel<$mr, $nr, $ti> = $packing;
                        k = f(k);
                    )*
                    $($(
                        k.stores.push(<$store>::datum_type());
                    )*)?
                    $(k.can_fuse = $can_fuse;)?
                    $(k.emulated = $emulated;)?
                    $(k = k.with_boost($boost);)?
                    $(k.row_major_store = $rms;)?
                    k
                };
            }

            #[cfg(test)]
            mod [<test_$id>] {
                use super::$id;
                test_mmm_kernel!($ti, &*super::$id);
                $(mmm_packed_packed_tests!(&*super::$id, $pid : $pnum);)*
                $($(mmm_store_test!(&*super::$id, $store);)*)?
            }
        }
    };
}
