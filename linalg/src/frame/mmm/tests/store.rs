use crate::frame::mmm::fuse::FusedKerSpec;
use crate::frame::mmm::storage::*;
use crate::frame::mmm::tests::display_error;
use crate::frame::mmm::*;
use crate::LADatum;
use num_traits::Bounded;
use tract_data::internal::*;
use tract_itertools::Itertools;

#[macro_export]
macro_rules! mmm_store_test {
    ($ker:expr, $tc:ident) => {
        paste! {
            mod [<store_$tc>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;

                #[test] fn store_zeros() {
                    $crate::frame::mmm::tests::store::store_zeros::<_,$tc,_>($ker);
                }

                #[test] fn store_col_major() {
                    $crate::frame::mmm::tests::store::store_pattern::<_,$tc,_>($ker, false);
                }

                #[test] fn store_row_major() {
                    $crate::frame::mmm::tests::store::store_pattern::<_,$tc,_>($ker, true);
                }
            }
        }
    };
}

pub fn mmm_stride_storage<T: Copy>(v: &[T], rsc: usize) -> OutputStoreKer {
    OutputStoreKer {
        ptr: v.as_ptr() as _,
        row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
        col_byte_stride: std::mem::size_of::<T>() as isize,
        item_size: std::mem::size_of::<T>(),
    }
}

pub fn store_zeros<K, TC, TI>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TC: LADatum,
    TI: LADatum + Bounded + PartialEq,
{
    if !ker.is_supported_here() {
        return;
    }
    let v = vec![TC::max_value(); ker.mr() * ker.nr()];
    let c = mmm_stride_storage(&v, ker.nr());
    let non_linear = tvec![FusedKerSpec::Clear, FusedKerSpec::Store(c), FusedKerSpec::Done];
    let err = ker.kernel(&non_linear);
    assert_eq!(err, 0);
    let expected = vec![TC::zero(); v.len()];
    display_error(&v, &expected, ker.mr(), ker.nr());
    assert_eq!(v, expected);
}

pub fn store_pattern<K, TC, TI>(ker: &K, row_major: bool)
where
    K: MatMatMulKer<Acc = TI>,
    TC: LADatum,
    TI: LADatum + Bounded + PartialEq,
{
    if !ker.is_supported_here() {
        return;
    }
    let len = ker.mr() * ker.nr();
    let pattern = tensor1(&(0..).take(len).collect_vec())
        .cast_to::<TI>()
        .unwrap()
        .into_owned()
        .into_shape(&[ker.mr(), ker.nr()])
        .unwrap();
    let pattern_col_major = pattern.clone().permute_axes(&[1, 0]).unwrap();
    let mut result = tensor0(TC::max_value()).broadcast_to_shape(&[len]).unwrap();
    let size_of_tc = std::mem::size_of::<TC>();
    let (row_byte_stride, col_byte_stride) = if row_major {
        ((size_of_tc * ker.nr()) as isize, size_of_tc as isize)
    } else {
        (size_of_tc as isize, (size_of_tc * ker.mr()) as isize)
    };
    let non_linear = tvec![
        unsafe {
            FusedKerSpec::LoadTile(pattern_col_major.as_ptr_unchecked(), pattern.as_ptr_unchecked())
        },
        FusedKerSpec::Store(OutputStoreKer {
            ptr: result.as_bytes_mut().as_mut_ptr(),
            row_byte_stride,
            col_byte_stride,
            item_size: size_of_tc,
        }),
        FusedKerSpec::Done
    ];
    let err = ker.kernel(&non_linear);
    assert_eq!(err, 0);
    let expected = pattern.cast_to::<TC>().unwrap().into_owned();
    if !row_major {
        result =
            result.into_shape(&[ker.nr(), ker.mr()]).unwrap().permute_axes(&[1, 0]).unwrap();
    }
    let expected = expected.as_slice::<TC>().unwrap();
    let result = result.as_slice::<TC>().unwrap();
    display_error(result, expected, ker.mr(), ker.nr());
    assert_eq!(result, expected);
}
