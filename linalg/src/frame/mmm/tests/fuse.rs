use crate::frame::mmm::fuse::FusedKerSpec;
use crate::frame::mmm::storage::*;
use crate::frame::mmm::tests::display_error;
use crate::frame::mmm::tests::store::mmm_stride_storage;
use crate::frame::mmm::*;
use num_traits::{AsPrimitive, Bounded};
use proptest::prelude::*;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_kernel_fuse_tests {
    ($ker:expr, $tc:ty, $ti: ty) => {
        mod fuse {
            use num_traits::Zero;
            #[allow(unused_imports)]
            use tract_data::prelude::f16;
            use tract_data::prelude::tensor0;
            use $crate::frame::mmm::MatMatMulKer;
            use $crate::frame::mmm::tests::fuse as test;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::fuse::*;

            $crate::mmm_test_case!($ker, "fuse", "return_zeros", {
                test::return_zeros::<_, $tc, $ti>($ker);
                Ok(())
            });

            $crate::mmm_test_case!($ker, "fuse", "store_non_contiguous", {
                test::store_non_contiguous::<_, $tc, $ti>($ker);
                Ok(())
            });

            $crate::mmm_test_case!($ker, "fuse", "add_unicast_non_contiguous", {
                test::add_unicast_non_contiguous::<_, $ti>($ker);
                Ok(())
            });

            $crate::mmm_test_case!($ker, "fuse", "return_c_prop", {
                $crate::mmm::tests::run_proptest(file!(), tile::<_, $ti>($ker), |c| {
                    test::return_c::<_, $ti>($ker, &c);
                    Ok(())
                })
            });

            fn fmin<T: PartialOrd>(a: T, b: T) -> T {
                if a < b { a } else { b }
            }

            fn fmax<T: PartialOrd>(a: T, b: T) -> T {
                if a > b { a } else { b }
            }

            macro_rules! bin {
                        ($FKS:ident, $case:expr, $geo:ident, $f:expr, $extra_cond:expr) => {
                            $crate::mmm_test_case!($ker, "fuse", $case, if($extra_cond), {
                                test::$geo::<_, $ti>($ker, $crate::mmm::FusedKerSpec::$FKS, $f);
                                Ok(())
                            });
                        };
                    }

            bin!(PerColMin, "per_col_min", per_col, fmin, true);
            bin!(PerColMax, "per_col_max", per_col, fmax, true);
            bin!(PerColAdd, "per_col_add", per_col, |a, b| a + b, true);
            bin!(PerColMul, "per_col_mul", per_col, |a, b| a * b, true);
            bin!(PerColSub, "per_col_sub", per_col, |a, b| a - b, true);
            bin!(PerColSubF, "per_col_sub_f", per_col, |a, b| b - a, true);

            bin!(PerRowMin, "per_row_min", per_row, fmin, true);
            bin!(PerRowMax, "per_row_max", per_row, fmax, true);
            bin!(PerRowAdd, "per_row_add", per_row, |a, b| a + b, true);
            bin!(PerRowMul, "per_row_mul", per_row, |a, b| a * b, true);
            bin!(PerRowSub, "per_row_sub", per_row, |a, b| a - b, true);
            bin!(PerRowSubF, "per_row_sub_f", per_row, |a, b| b - a, true);

            bin!(ScalarMin, "scalar_min", scalar, fmin, true);
            bin!(ScalarMax, "scalar_max", scalar, fmax, true);
            bin!(ScalarAdd, "scalar_add", scalar, |a, b| a + b, true);
            bin!(ScalarMul, "scalar_mul", scalar, |a, b| a * b, true);
            bin!(ScalarSub, "scalar_sub", scalar, |a, b| a - b, true);
            bin!(ScalarSubF, "scalar_sub_f", scalar, |a, b| b - a, true);

            bin!(
                LeakyRelu,
                "leaky_relu",
                scalar,
                |a, b| if b > <$ti>::zero() { b } else { a * b },
                ($ker).can_fuse(&$crate::mmm::FusedSpec::LeakyRelu(&tensor0(<$ti>::from(1_u8))))
            );

            $crate::mmm_test_case!($ker, "fuse", "return_c_add_row_col_product", {
                test::return_c_add_row_col_product::<_, $ti>($ker);
                Ok(())
            });

            $crate::mmm_test_case!($ker, "fuse", "return_c_plus_d", {
                test::return_c_plus_d::<_, $ti, $ti>($ker);
                Ok(())
            });

            $crate::mmm_test_case!($ker, "fuse", "return_c_clear", {
                test::return_c_clear::<_, $ti>($ker);
                Ok(())
            });
        }
    };
}

use crate::LADatum;
pub fn return_zeros<K, TC, TI>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TC: LADatum,
    TI: LADatum + Bounded + PartialEq,
{
    let v = vec![TC::max_value(); ker.mr() * ker.nr()];
    let c = mmm_stride_storage(&v, ker.nr());
    let non_linear = tvec![FusedKerSpec::Clear, FusedKerSpec::Store(c), FusedKerSpec::Done];
    let err = ker.kernel(&non_linear);
    assert_eq!(err, 0);
    let expected = vec![TC::zero(); v.len()];
    display_error(&v, &expected, ker.mr(), ker.nr());
    assert_eq!(v, expected);
}

pub fn store_non_contiguous<K, TC, TI>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TC: LADatum,
    TI: LADatum + Bounded + PartialEq,
{
    let v = vec![TC::max_value(); ker.mr() * 5 * ker.nr() * 3];
    let c = OutputStoreKer {
        ptr: v.as_ptr() as _,
        row_byte_stride: (std::mem::size_of::<TC>() * 3 * ker.nr() * 5) as isize,
        col_byte_stride: std::mem::size_of::<TC>() as isize * 3,
        item_size: std::mem::size_of::<TC>(),
    };
    let non_linear = tvec![FusedKerSpec::Clear, FusedKerSpec::Store(c), FusedKerSpec::Done];
    let err = ker.kernel(&non_linear);
    assert_eq!(err, 0);
    let mut expected = vec![TC::max_value(); v.len()];
    for c in 0..ker.nr() {
        for r in 0..ker.mr() {
            expected[c * 3 + r * 3 * 5 * ker.nr()] = TC::zero();
        }
    }
    assert_eq!(v, expected);
}

/// `Clear` + `AddUnicast(strided)` + `Store(contiguous)` and check the
/// source pattern reaches the destination. Counterpart of
/// `store_non_contiguous` on the read side; `return_c_plus_d` uses
/// `mmm_stride_storage` (tightly packed) and so doesn't exercise this.
pub fn add_unicast_non_contiguous<K, TI>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum + AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    let item = std::mem::size_of::<TI>();
    let row_stride_items = 3 * ker.nr() * 5;
    let col_stride_items = 3;
    // Source: a non-contiguous buffer with distinct values at the used
    // (r, c) cells and sentinel garbage everywhere else.
    let mut src: Vec<TI> = vec![TI::max_value(); ker.mr() * row_stride_items];
    for r in 0..ker.mr() {
        for c in 0..ker.nr() {
            src[r * row_stride_items + c * col_stride_items] = (1 + c + r * ker.nr()).as_();
        }
    }
    let src_store = OutputStoreKer {
        ptr: src.as_ptr() as _,
        row_byte_stride: (item * row_stride_items) as isize,
        col_byte_stride: (item * col_stride_items) as isize,
        item_size: item,
    };
    // Destination: tightly-packed output for easy comparison.
    let mut dst: Vec<TI> = vec![TI::min_value(); ker.mr() * ker.nr()];
    let dst_store = OutputStoreKer {
        ptr: dst.as_mut_ptr() as _,
        row_byte_stride: (item * ker.nr()) as isize,
        col_byte_stride: item as isize,
        item_size: item,
    };
    let non_linear = tvec![
        FusedKerSpec::Clear,
        FusedKerSpec::AddUnicast(src_store),
        FusedKerSpec::Store(dst_store),
        FusedKerSpec::Done,
    ];
    let err = ker.kernel(&non_linear);
    assert_eq!(err, 0);
    let expected: Vec<TI> = (0..ker.mr() * ker.nr()).map(|i| (1 + i).as_()).collect();
    display_error(&dst, &expected, ker.mr(), ker.nr());
    assert_eq!(dst, expected);
}

pub fn fused_ops<K, TI, E>(ker: &K, c: &[TI], ops: &[FusedKerSpec<TI>], expect: E)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    E: Fn(usize, usize, TI) -> TI,
{
    assert!(c.len() == ker.mr() * ker.nr());
    let v = c.to_vec();
    let c = mmm_stride_storage(&v, ker.nr());
    let mut ops = ops.to_vec();
    ops.insert(0, FusedKerSpec::AddUnicast(c));
    ops.insert(0, FusedKerSpec::Clear);
    ops.push(FusedKerSpec::Store(c));
    ops.push(FusedKerSpec::Done);
    let expected =
        (0..v.len()).map(|ix| expect(ix / ker.nr(), ix % ker.nr(), v[ix])).collect::<Vec<TI>>();
    let err = ker.kernel(&ops);
    assert_eq!(err, 0);
    display_error(&v, &expected, ker.mr(), ker.nr());
    assert_eq!(v, expected);
}

pub fn return_c<K, TI>(ker: &K, v: &[TI])
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    usize: AsPrimitive<TI>,
{
    fused_ops::<K, TI, _>(ker, v, &[], |_, _, c| c + 1.as_() - 1.as_())
}

pub fn return_c_plus_d<K, TI, TD>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    TD: LADatum + AsPrimitive<TI>,
    usize: AsPrimitive<TI> + AsPrimitive<TD>,
{
    let len = ker.mr() * ker.nr();
    let v: Vec<TI> = (0..len).map(|f| f.as_()).collect();
    let d: Vec<TD> = (0..len).map(|f| ((3 * f) % 7).as_()).collect();
    fused_ops::<K, TI, _>(
        ker,
        &v,
        &[FusedKerSpec::AddUnicast(mmm_stride_storage(&d, ker.nr()))],
        |row, col, c| c + d[row * ker.nr() + col].as_(),
    );
}

pub fn per_col<K, TI>(ker: &K, op: impl Fn(*const TI) -> FusedKerSpec<TI>, f: impl Fn(TI, TI) -> TI)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    usize: AsPrimitive<TI>,
{
    let len = ker.mr() * ker.nr();
    let v: Vec<TI> = (0..len).map(|f| f.as_()).collect();
    let bias: Vec<TI> = (0..ker.nr()).map(|f| (f + 1).as_()).collect();
    fused_ops::<K, TI, _>(ker, &v, &[op(bias.as_ptr())], |_, col, c| f(bias[col], c))
}

pub fn per_row<K, TI>(ker: &K, op: impl Fn(*const TI) -> FusedKerSpec<TI>, f: impl Fn(TI, TI) -> TI)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    usize: AsPrimitive<TI>,
{
    let len = ker.mr() * ker.nr();
    let v: Vec<TI> = (0..len).map(|f| f.as_()).collect();
    let bias: Vec<TI> = (0..ker.mr()).map(|f| (f + 1).as_()).collect();
    fused_ops::<K, TI, _>(ker, &v, &[op(bias.as_ptr())], |row, _, c| f(bias[row], c))
}

pub fn scalar<K, TI>(ker: &K, op: impl Fn(TI) -> FusedKerSpec<TI>, f: impl Fn(TI, TI) -> TI)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    isize: AsPrimitive<TI>,
{
    let len = ker.mr() * ker.nr();
    let v: Vec<TI> = (0..len as isize).map(|f| (f - len as isize / 2).as_()).collect();
    let five: TI = 5.as_();
    fused_ops::<K, TI, _>(ker, &v, &[op(five)], |_, _, c| f(five, c))
}

pub fn return_c_add_row_col_product<K, TI>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    usize: AsPrimitive<TI>,
{
    let len = ker.mr() * ker.nr();
    let v: Vec<TI> = (0..len).map(|f| (f + 1).as_()).collect();
    let rows: Vec<TI> = (0..ker.mr()).map(|f| (f + 3).as_()).collect();
    let cols: Vec<TI> = (0..ker.nr()).map(|f| (f + 2).as_()).collect();
    fused_ops::<K, TI, _>(
        ker,
        &v,
        &[FusedKerSpec::AddRowColProducts(rows.as_ptr(), cols.as_ptr())],
        |row, col, c| c + cols[col] * rows[row],
    )
}

pub fn return_c_clear<K, TI>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    usize: AsPrimitive<TI>,
{
    let len = ker.mr() * ker.nr();
    let v: Vec<TI> = (0..len).map(|f| f.as_()).collect();
    fused_ops::<K, TI, _>(ker, &v, &[FusedKerSpec::Clear], |_, _, _| 0.as_())
}

pub fn tile<K, TI>(ker: &K) -> BoxedStrategy<Vec<TI>>
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    i8: AsPrimitive<TI>,
{
    let len = ker.mr() * ker.nr();
    proptest::collection::vec(any::<i8>().prop_map(|c| c.as_()), len..=len).boxed()
}
