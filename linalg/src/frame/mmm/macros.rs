macro_rules! MMMExternKernel {
    (
            $func:ident<$ti:ident>($mr: expr, $nr: expr)
            $(@($align_a:expr, $align_b:expr))?
            $(where($where:expr))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(quality($quality:expr))?
            $(boost($boost:expr))?
            $(store($($store:ty),*))?
            $(row_major_store($rms:expr))?
     ) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use super::*;
                #[allow(unused_imports)]
                use crate::frame::mmm::*;
                extern_kernel!(fn $func(op: *const FusedKerSpec<$ti>) -> isize);

                #[inline]
                pub unsafe fn rusty(op: &[FusedKerSpec<$ti>]) -> isize {
                    unsafe { $func(op.as_ptr()) }
                }
            }

            MMMKernel!([<sys_$func>]::rusty as $func<$ti>($mr, $nr)
                $(@($align_a, $align_b))?
                $(where($where))?
                $(can_fuse($can_fuse))?
                $(packing[$pnum] = $pid => $packing;)*
                $(quality($quality))?
                $(boost($boost))?
                $(store($($store),*))?
                $(row_major_store($rms))?
            );
        }
    };
}
// Temporary: like `MMMExternKernel!`, but compiles on any host and registers an
// introspection descriptor. The leading ident names the target arch the asm was written
// for; the extern symbol is emitted only there, replaced elsewhere by a bail stub so the
// whole module links everywhere. The `MmmRoutine` submit is unconditional (pure data),
// with `bound` recording whether this build actually assembled the kernel.
macro_rules! MMMExternKernel2 {
    // Map the arch ident to its `target_arch` string literal (cfg needs a literal).
    (arm; $($rest:tt)*)     => { MMMExternKernel2!(@ "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { MMMExternKernel2!(@ "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*)  => { MMMExternKernel2!(@ "x86_64"; $($rest)*); };

    (@ $arch:literal;
        $func:ident<$ti:ident>($mr:expr, $nr:expr)
        $($rest:tt)*
    ) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use super::*;
                #[allow(unused_imports)]
                use crate::frame::mmm::*;

                #[cfg(target_arch = $arch)]
                extern_kernel!(fn $func(op: *const FusedKerSpec<$ti>) -> isize);

                #[cfg(not(target_arch = $arch))]
                #[allow(dead_code)]
                pub unsafe fn $func(_op: *const FusedKerSpec<$ti>) -> isize {
                    panic!(concat!(stringify!($func), ": mmm kernel not built for this target arch"))
                }

                #[inline]
                pub unsafe fn rusty(op: &[FusedKerSpec<$ti>]) -> isize {
                    unsafe { $func(op.as_ptr()) }
                }
            }

            MMMKernel!([<sys_ $func>]::rusty as $func<$ti>($mr, $nr) $($rest)*);

            inventory::submit! {
                $crate::mmm_routines::MmmRoutine {
                    target: $arch,
                    bound: cfg!(target_arch = $arch),
                    make: || $func.mmm(),
                }
            }
        }
    };
}

macro_rules! MMMRustKernel {
    (       $func: path =>
            $id:ident<$ti:ident>($mr: expr, $nr: expr)
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
