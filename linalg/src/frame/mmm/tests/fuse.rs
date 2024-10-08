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
            use $crate::frame::mmm::tests::fuse as test;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::fuse::*;
            use $crate::frame::mmm::MatMatMulKer;

            #[test]
            fn return_zeros() {
                test::return_zeros::<_, $tc, $ti>($ker)
            }

            #[test]
            fn store_non_contiguous() {
                test::store_non_contiguous::<_, $tc, $ti>($ker)
            }
            proptest::proptest! {
                #[test]
                fn return_c_prop(c in tile::<_, $ti>($ker)) {
                    test::return_c::<_, $ti>($ker, &c)
                }
            }

            fn fmin<T: PartialOrd>(a: T, b: T) -> T {
                if a < b {
                    a
                } else {
                    b
                }
            }

            fn fmax<T: PartialOrd>(a: T, b: T) -> T {
                if a > b {
                    a
                } else {
                    b
                }
            }

            macro_rules! bin {
                ($FKS:ident, $geo:expr, $f:expr, $extra_cond:expr) => {
                    paste! {
                        #[test]
                        fn [<$FKS:snake>]() {
                            if ($ker).is_supported_here() && $extra_cond {
                                test::$geo::<_, $ti>($ker, $crate::mmm::FusedKerSpec::$FKS, $f);
                            }
                        }
                    }
                };
            }

            bin!(PerColMin, per_col, fmin, true);
            bin!(PerColMax, per_col, fmax, true);
            bin!(PerColAdd, per_col, |a, b| a + b, true);
            bin!(PerColMul, per_col, |a, b| a * b, true);
            bin!(PerColSub, per_col, |a, b| a - b, true);
            bin!(PerColSubF, per_col, |a, b| b - a, true);

            bin!(PerRowMin, per_row, fmin, true);
            bin!(PerRowMax, per_row, fmax, true);
            bin!(PerRowAdd, per_row, |a, b| a + b, true);
            bin!(PerRowMul, per_row, |a, b| a * b, true);
            bin!(PerRowSub, per_row, |a, b| a - b, true);
            bin!(PerRowSubF, per_row, |a, b| b - a, true);

            bin!(ScalarMin, scalar, fmin, true);
            bin!(ScalarMax, scalar, fmax, true);
            bin!(ScalarAdd, scalar, |a, b| a + b, true);
            bin!(ScalarMul, scalar, |a, b| a * b, true);
            bin!(ScalarSub, scalar, |a, b| a - b, true);
            bin!(ScalarSubF, scalar, |a, b| b - a, true);

            bin!(
                LeakyRelu,
                scalar,
                |a, b| if b > <$ti>::zero() { b } else { a * b },
                ($ker).can_fuse(&$crate::mmm::FusedSpec::LeakyRelu(&tensor0(<$ti>::from(1_u8))))
            );

            #[test]
            fn return_c_add_row_col_product() {
                test::return_c_add_row_col_product::<_, $ti>($ker)
            }

            #[test]
            fn return_c_plus_d() {
                test::return_c_plus_d::<_, $ti, $ti>($ker)
            }

            #[test]
            fn return_c_clear() {
                test::return_c_clear::<_, $ti>($ker)
            }
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

pub fn store_non_contiguous<K, TC, TI>(ker: &K)
where
    K: MatMatMulKer<Acc = TI>,
    TC: LADatum,
    TI: LADatum + Bounded + PartialEq,
{
    if !ker.is_supported_here() {
        return;
    }
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

pub fn fused_ops<K, TI, E>(ker: &K, c: &[TI], ops: &[FusedKerSpec<TI>], expect: E)
where
    K: MatMatMulKer<Acc = TI>,
    TI: LADatum,
    E: Fn(usize, usize, TI) -> TI,
{
    if !ker.is_supported_here() {
        return;
    }
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
