use crate::frame::mmm::fuse::FusedKerSpec;
use crate::frame::mmm::storage::*;
use crate::frame::mmm::tests::display_error;
use crate::frame::mmm::*;
use crate::LADatum;
use num_traits::Bounded;
use tract_data::internal::*;
use tract_itertools::Itertools;
use tract_ndarray::Axis;

#[macro_export]
macro_rules! mmm_store_test {
    ($ker:expr, $tc:ident) => {
        paste! {
            mod [<store_$tc>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                use $crate::frame::mmm::tests::store::StoreLayout;

                #[test] fn store_zeros() {
                    $crate::frame::mmm::tests::store::store_zeros::<_,$tc,_>($ker);
                }

                #[test] fn store_col_major() {
                    $crate::frame::mmm::tests::store::store_pattern::<_,$tc,_>($ker, StoreLayout::ColMajor);
                }

                #[test] fn store_row_major() {
                    $crate::frame::mmm::tests::store::store_pattern::<_,$tc,_>($ker, StoreLayout::RowMajor);
                }

                #[test] fn store_arbitrary() {
                    $crate::frame::mmm::tests::store::store_pattern::<_,$tc,_>($ker, StoreLayout::Arbitrary);
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

pub enum StoreLayout {
    ColMajor,
    RowMajor,
    Arbitrary,
}

pub fn store_pattern<K, TC, TI>(ker: &K, layout: StoreLayout)
where
    K: MatMatMulKer<Acc = TI>,
    TC: LADatum,
    TI: LADatum + Bounded + PartialEq,
{
    if !ker.is_supported_here() {
        return;
    }
    let (mr, nr) = (ker.mr(), ker.nr());
    let pattern = tensor1(&(0..).take(mr * nr).collect_vec())
        .cast_to::<TI>()
        .unwrap()
        .into_owned()
        .into_shape(&[mr, nr])
        .unwrap();
    let pattern_col_major = pattern.clone().permute_axes(&[1, 0]).unwrap();
    let size_of_tc = std::mem::size_of::<TC>();
    let (row_stride, col_stride, result_size) = match layout {
        StoreLayout::RowMajor => (nr, 1, mr * nr),
        StoreLayout::ColMajor => (1, mr, mr * nr),
        // like row major, but storing every other third column
        StoreLayout::Arbitrary => (nr * 3, 3, mr * nr * 3),
    };
    let mut result = tensor0(TC::max_value()).broadcast_to_shape(&[result_size]).unwrap();
    let non_linear = tvec![
        unsafe {
            FusedKerSpec::LoadTile(pattern_col_major.as_ptr_unchecked(), pattern.as_ptr_unchecked())
        },
        FusedKerSpec::Store(OutputStoreKer {
            ptr: result.as_bytes_mut().as_mut_ptr(),
            row_byte_stride: (size_of_tc * row_stride) as isize,
            col_byte_stride: (size_of_tc * col_stride) as isize,
            item_size: size_of_tc,
        }),
        FusedKerSpec::Done
    ];
    let err = ker.kernel(&non_linear);
    assert_eq!(err, 0);
    let expected = pattern.cast_to::<TC>().unwrap().into_owned();
    let result = match layout {
        StoreLayout::RowMajor => result,
        StoreLayout::ColMajor => {
            result.into_shape(&[ker.nr(), ker.mr()]).unwrap().permute_axes(&[1, 0]).unwrap()
        }
        StoreLayout::Arbitrary => result
            .into_array::<TC>()
            .unwrap()
            .into_shape_with_order((mr, nr, 3))
            .unwrap()
            .index_axis_move(Axis(2), 0)
            .into_tensor(),
    };
    let expected = expected.as_slice::<TC>().unwrap();
    let result = result.as_slice::<TC>().unwrap();
    display_error(result, expected, ker.mr(), ker.nr());
    assert_eq!(result, expected);
}
