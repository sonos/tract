// An mmm kernel whose inner loop is asm. The leading ident names the arch that asm was
// written for: the extern symbol is emitted only in builds carrying that arch's
// instructions, replaced elsewhere by a bail stub, so the tree links everywhere. The
// `MmmRoutine` submit is unconditional (pure data), with `bound` recording whether this
// build assembled the kernel.
macro_rules! MMMExternKernel {
    // Map the arch ident to its inventory label and to the cfg predicate for "built here".
    (arm; $($rest:tt)*)     => { MMMExternKernel!(@ Some($crate::platform::Target::Arm), target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { MMMExternKernel!(@ Some($crate::platform::Target::Aarch64), target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*)  => { MMMExternKernel!(@ Some($crate::platform::Target::X86_64), target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { MMMExternKernel!(@ Some($crate::platform::Target::RiscV64), target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*)  => { MMMExternKernel!(@ Some($crate::platform::Target::Wasm32Simd128), all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

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
                bound(cfg!($built)) $($rest)*);

            inventory::submit! {
                $crate::mmm_routines::MmmRoutine {
                    target: $target,
                    make: || $func.mmm(),
                }
            }
        }
    };
}

// An mmm kernel whose inner loop is Rust — intrinsics, or a C extern block like SVE's.
// Given a leading arch ident it also registers an introspection descriptor, `bound` telling
// whether this build compiled the kernel; the kernel fn itself is gated by its own module.
// Without the ident it is a portable kernel, built everywhere.
macro_rules! MMMRustKernel {
    // Portable Rust, built and dispatchable everywhere.
    (generic; $($rest:tt)*) => { MMMRustKernel!(@ None, all(); $($rest)*); };
    (arm; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::platform::Target::Arm), target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::platform::Target::Aarch64), target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::platform::Target::X86_64), target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::platform::Target::RiscV64), target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { MMMRustKernel!(@ Some($crate::platform::Target::Wasm32Simd128), all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $target:expr, $built:meta; $func:path => $id:ident<$ti:ident>($mr:expr, $nr:expr) $($rest:tt)*) => {
        MMMRustKernel!($func => $id<$ti>($mr, $nr) bound(cfg!($built)) $($rest)*);
        inventory::submit! {
            $crate::mmm_routines::MmmRoutine {
                target: $target,
                make: || $id.mmm(),
            }
        }
    };

    (       $func: path =>
            $id:ident<$ti:ident>($mr: expr, $nr: expr)
            $(bound($bound:expr))?
            $(@($align_a:expr, $align_b:expr))?
            $(where($where:expr))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(quality($quality:expr))?
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
                $(bound($bound))?
                $(@($align_a, $align_b))?
                generic(true)
                $(where($where))?
                $(can_fuse($can_fuse))?
                $(packing[$pnum] = $pid => $packing;)*
                $(quality($quality))?
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
            $(bound($bound:expr))?
            $(@($align_a:expr, $align_b:expr))?
            $(generic($generic:expr))?
            $(where($where:expr))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(quality($quality:expr))?
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
                    let mut k = DynKernel::<$mr, $nr, $ti>::new(stringify!($id), $func, packing_a, packing_b, $crate::frame::mmm::ImplementationQuality::Dreadful);
                    $(k.bound = $bound;)?
                    $(k = k.with_platform_condition($where);)?
                    $(
                        assert!(k.packings.len() == $pnum);
                        let f: fn(DynKernel<$mr, $nr, $ti>) -> DynKernel<$mr, $nr, $ti> = $packing;
                        k = f(k);
                    )*
                    $($(
                        k.stores.push(<$store>::datum_type());
                    )*)?
                    $(k.can_fuse = $can_fuse;)?
                    $(k.quality = $quality;)?
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
