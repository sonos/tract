use crate::LADatum;
use crate::frame::mmm::fuse::FusedKerSpec;
use crate::frame::mmm::storage::*;
use crate::frame::mmm::tests::display_error;
use crate::frame::mmm::*;
use num_traits::{AsPrimitive, Bounded};
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

                #[test] fn add_unicast_dt() {
                    $crate::frame::mmm::tests::fuse::return_c_plus_d::<_, _, $tc>($ker);
                }

                #[test] fn add_unicast_col_major() {
                    $crate::frame::mmm::tests::store::add_unicast_pattern::<_,$tc,_>($ker, StoreLayout::ColMajor);
                }

                #[test] fn add_unicast_row_major() {
                    $crate::frame::mmm::tests::store::add_unicast_pattern::<_,$tc,_>($ker, StoreLayout::RowMajor);
                }

                #[test] fn add_unicast_arbitrary() {
                    $crate::frame::mmm::tests::store::add_unicast_pattern::<_,$tc,_>($ker, StoreLayout::Arbitrary);
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
    let pattern_aligned = Blob::from_bytes_alignment(pattern.as_bytes(), 128).unwrap();
    let pattern_col_major = pattern.clone().permute_axes(&[1, 0]).unwrap();
    let pattern_col_major_aligned =
        Blob::from_bytes_alignment(pattern_col_major.as_bytes(), 128).unwrap();
    let size_of_tc = std::mem::size_of::<TC>();
    let (row_stride, col_stride, result_size) = match layout {
        StoreLayout::RowMajor => (nr, 1, mr * nr),
        StoreLayout::ColMajor => (1, mr, mr * nr),
        // like row major, but storing every other third column
        StoreLayout::Arbitrary => (nr * 3, 3, mr * nr * 3),
    };
    let mut result = tensor0(TC::max_value()).broadcast_to_shape(&[result_size]).unwrap();
    let non_linear = tvec![
        FusedKerSpec::LoadTile(
            pattern_col_major_aligned.as_ptr() as *const TI,
            pattern_aligned.as_ptr() as *const TI,
        ),
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
            .into_plain_array::<TC>()
            .unwrap()
            .into_shape_with_order((mr, nr, 3))
            .unwrap()
            .index_axis_move(Axis(2), 0)
            .into_tensor(),
    };
    let expected = expected.try_as_plain().unwrap().as_slice::<TC>().unwrap();
    let result = result.try_as_plain().unwrap().as_slice::<TC>().unwrap();
    display_error(result, expected, ker.mr(), ker.nr());
    assert_eq!(result, expected);
}

/// `Clear` + `AddUnicast(operand of dtype TC)` + `Store(Acc)` and check the
/// operand reached the accumulator unchanged. The unicast operand carries the
/// kernel's declared output dtype `TC` (e.g. f16 for a kernel that stores f16
/// while accumulating in f32), across the three `StoreLayout`s so both the
/// contiguous and strided load paths are exercised. `add_unicast_dt` only
/// covers one layout per kernel; without a dtype-aware load an f16 operand
/// read as f32 saturates, so this catches `add_unicast` ignoring `item_size`.
pub fn add_unicast_pattern<K, TC, TI>(ker: &K, layout: StoreLayout)
where
    K: MatMatMulKer<Acc = TI>,
    TC: LADatum + AsPrimitive<TI>,
    TI: LADatum + Bounded + PartialEq,
    usize: AsPrimitive<TC>,
{
    if !ker.is_supported_here() {
        return;
    }
    let (mr, nr) = (ker.mr(), ker.nr());
    let size_of_tc = std::mem::size_of::<TC>();
    let (row_stride, col_stride, operand_size) = match layout {
        StoreLayout::RowMajor => (nr, 1, mr * nr),
        StoreLayout::ColMajor => (1, mr, mr * nr),
        // like row major, but reading every other third column
        StoreLayout::Arbitrary => (nr * 3, 3, mr * nr * 3),
    };
    // Distinct value per (r, c) cell in the operand's dtype, with sentinel
    // garbage in the unread cells of the strided layouts. Values round-trip
    // through TC (so small integer output dtypes like i8 don't overflow).
    let cell: Vec<TC> = (0..mr * nr).map(|i| (1 + i).as_()).collect();
    let mut operand = vec![TC::max_value(); operand_size];
    for r in 0..mr {
        for c in 0..nr {
            operand[r * row_stride + c * col_stride] = cell[r * nr + c];
        }
    }
    let mut result = vec![TI::min_value(); mr * nr];
    let non_linear = tvec![
        FusedKerSpec::Clear,
        FusedKerSpec::AddUnicast(OutputStoreKer {
            ptr: operand.as_ptr() as _,
            row_byte_stride: (size_of_tc * row_stride) as isize,
            col_byte_stride: (size_of_tc * col_stride) as isize,
            item_size: size_of_tc,
        }),
        FusedKerSpec::Store(OutputStoreKer {
            ptr: result.as_mut_ptr() as _,
            row_byte_stride: (std::mem::size_of::<TI>() * nr) as isize,
            col_byte_stride: std::mem::size_of::<TI>() as isize,
            item_size: std::mem::size_of::<TI>(),
        }),
        FusedKerSpec::Done,
    ];
    let err = ker.kernel(&non_linear);
    assert_eq!(err, 0);
    let expected: Vec<TI> = cell.iter().map(|&v| v.as_()).collect();
    display_error(&result, &expected, mr, nr);
    assert_eq!(result, expected);
}
