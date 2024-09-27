macro_rules! MMMExternKernel {
    (
            $func:ident<$ti:ident>($mr: expr, $nr: expr)@($align_a:expr, $align_b:expr)
            $(where($where:expr))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(store($($store:ty),*))?
            $(tweak($tweak: expr))*
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
                    $func(op.as_ptr())
                }
            }

            MMMKernel!([<sys_$func>]::rusty as $func<$ti>($mr, $nr)@($align_a, $align_b)
                $(where($where))?
                $(can_fuse($can_fuse))?
                $(packing[$pnum] = $pid => $packing;)*
                $(store($($store),*))?
            );
        }
    };
}
macro_rules! MMMRustKernel {
    (       $func: path =>
            $id:ident<$ti:ident>($mr: expr, $nr: expr)@($align_a:expr, $align_b:expr)
            $(where($where:expr))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(store($($store:ty),*))?
            $(tweak($tweak: expr))*
     ) => {
        paste! {
            mod [<sys_ $id>] {
                #[allow(unused_imports)]
                use crate::frame::mmm::*;
                use super::*;
                #[inline]
                pub unsafe fn rusty(op: &[FusedKerSpec<$ti>]) -> isize {
                    $func(op.as_ptr())
                }
            }
            MMMKernel!([<sys_$id>]::rusty as $id<$ti>($mr, $nr)@($align_a, $align_b)
                $(where($where))?
                $(can_fuse($can_fuse))?
                $(packing[$pnum] = $pid => $packing;)*
                $(store($($store),*))?
            );
        }
    }
}

macro_rules! MMMKernel {
    (
            $func: path as
            $id:ident<$ti:ident>($mr: expr, $nr: expr)@($align_a:expr, $align_b:expr)
            $(where($where:expr))?
            $(can_fuse($can_fuse:expr))?
            $(packing[$pnum:literal] = $pid:ident => $packing:expr;)*
            $(store($($store:ty),*))?
            $(tweak($tweak: expr))*
     ) => {
        paste! {
            lazy_static::lazy_static! {
                pub static ref $id: $crate::mmm::DynKernel<$mr, $nr, $ti> = {
                    use $crate::mmm::DynKernel;
                    #[allow(unused_imports)]
                    use tract_data::prelude::*;
                    #[allow(unused_mut)]
                    let mut k = DynKernel::<$mr, $nr, $ti>::new(stringify!($id), $func, ($align_a, $align_b));
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
                    $($tweak(&mut k);)*
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

