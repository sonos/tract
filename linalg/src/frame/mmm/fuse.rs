use std::fmt::Debug;

use super::Tile;
use tract_data::internal::*;

#[repr(usize)]
#[derive(Copy, Clone, Debug, PartialEq, Hash)]
pub enum RoundingPolicy {
    Native,
    Zero,
    Away,
    MinusInf,
    PlusInf,
    Even,
    Odd,
}

#[derive(Clone, Debug)]
pub enum FusedSpec<'t> {
    Min(&'t Tensor),
    Max(&'t Tensor),
    PerRowMul(&'t Tensor),
    PerRowAdd(&'t Tensor),
    PerColMul(&'t Tensor),
    PerColAdd(&'t Tensor),
    AddRowColProducts(&'t Tensor, &'t Tensor),
    ScalarMul(&'t Tensor),
    ScalarAdd(&'t Tensor),
    QTowardsEven(&'t Tensor, usize),
    QTowardsPlusInf(&'t Tensor, usize),
    QAway(&'t Tensor, usize),
    AddUnicast(TensorView<'t>),
    QWrappingMulHighDoubling(i32),
    QShiftRightRounding(usize, RoundingPolicy),
}

// Scale(f32, rounding policy) // recalcul ?
// QWrapMulHigh(i32) + QShiftRightTiesEven + QShiftRightTiesAway + QShiftRightTiesPlus +
// QShiftRightTiesZero + QShiftRightTies
// Scale(i32, shift, policy)

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum FusedKerSpec<TI: Copy> {
    Done,
    Min(TI),
    Max(TI),
    AddUnicast(Tile),
    PerRowMul(*const TI),
    PerRowAdd(*const TI),
    PerColMul(*const TI),
    PerColAdd(*const TI),
    AddRowColProducts(*const TI, *const TI),
    ScalarMul(TI),
    ScalarAdd(TI),
    QTowardsEven(TI, usize),
    QTowardsPlusInf(TI, usize),
    QAway(TI, usize),
    QWrappingMulHighDoubling(i32),
    QShiftRightRounding(usize, RoundingPolicy),
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::frame::mmm::storage::*;
    use crate::frame::mmm::*;
    use crate::generic::PseudoRightShift;
    use num_traits::{AsPrimitive, Bounded, Zero};
    use proptest::prelude::*;
    use std::fmt;
    use std::ops::{Add, Mul, Sub};

    #[test]
    fn check_non_linear_enum_size() {
        assert_eq!(
            std::mem::size_of::<super::FusedKerSpec<f32>>(),
            std::mem::size_of::<usize>() + std::mem::size_of::<Tile>()
        );
        assert_eq!(
            std::mem::size_of::<super::FusedKerSpec<f32>>(),
            5 * std::mem::size_of::<usize>()
        );
    }

    #[macro_export]
    macro_rules! mmm_kernel_fuse_tests {
        ($cond:expr, $ker:ty, $tc:ty, $ti: ty) => {
            mod fuse {
                #[allow(unused_imports)]
                use crate::frame::mmm::fuse::test;
                use proptest::prelude::*;

                #[test]
                fn return_zeros() {
                    if $cond {
                        test::return_zeros::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c() {
                    if $cond {
                        test::return_c::<$ker, $tc, $ti>()
                    }
                }

                proptest::proptest! {
                    #[test]
                    fn return_c_prop(pb in any::<test::ReturnCProblem<$ker, $tc, $ti>>()) {
                        if $cond {
                            let got = pb.run();
                            prop_assert!(got.iter().zip(pb.c.iter()).all(|(g,e)| (*g as f32 - *e as f32).abs() < 1e-7),
                            "got: {:?}\nexpected: {:?}", pb.run(), pb.c)
                        }
                    }
                }

                #[test]
                fn return_c_mul_row() {
                    if $cond {
                        test::return_c_mul_row::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_add_row() {
                    if $cond {
                        test::return_c_add_row::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_mul_col() {
                    if $cond {
                        test::return_c_mul_col::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_add_col() {
                    if $cond {
                        test::return_c_add_col::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_add_row_col_product() {
                    if $cond {
                        test::return_c_add_row_col_product::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_max() {
                    if $cond {
                        test::return_c_max::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_min() {
                    if $cond {
                        test::return_c_min::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_mul() {
                    if $cond {
                        test::return_c_scalar_mul::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_add() {
                    if $cond {
                        test::return_c_scalar_add::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_plus_d() {
                    if $cond {
                        test::return_c_plus_d::<$ker, $tc, $ti>()
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! qmmm_kernel_fuse_tests {
        ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod fuseq {
                use crate::frame::mmm::fuse::RoundingPolicy;
                #[allow(unused_imports)]
                use crate::frame::mmm::fuse::test;
                use crate::frame::mmm::fuse::test::{QWrappingMulHighDoublingProblem, QAwayProblem, QTowardsPlusInfProblem, QRightShiftProblem};
                use crate::frame::mmm::kernel::MatMatMulKer;
                use num_traits::AsPrimitive;
                use proptest::prelude::*;

                #[test]
                #[ignore]
                fn return_q_towards_even() {
                    if $cond {
                        test::return_q_towards_even::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_q_towards_plusinf() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let half_len: $ti = (len / 2).as_();
                        let v: Vec<$tc> = (0..len)
                            .map(|f| {
                                (<usize as AsPrimitive<$ti>>::as_(f) - half_len)
                                    .min(<$tc>::max_value().as_())
                                    .max(<$tc>::min_value().as_())
                                    .as_()
                            })
                        .collect();
                        let pb = QTowardsPlusInfProblem::<$ker, $tc, $ti>::new(v);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_away() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let half_len: $ti = (len / 2).as_();
                        let v: Vec<$tc> = (0..len)
                            .map(|f| {
                                (<usize as AsPrimitive<$ti>>::as_(f) - half_len)
                                    .min(<$tc>::max_value().as_())
                                    .max(<$tc>::min_value().as_())
                                    .as_()
                            })
                        .collect();
                        let pb = QAwayProblem::<$ker, $tc, $ti>::new(v);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_towards_plusinf_1() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(2);
                        let pb = QTowardsPlusInfProblem::<$ker, $tc, $ti>::new(v);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_right_shift_0() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(1);
                        let pb = QRightShiftProblem::<$ker, $tc, $ti>::new(v, 1, RoundingPolicy::Zero);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_right_shift_1() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(1);
                        let pb = QRightShiftProblem::<$ker, $tc, $ti>::new(v, 1, RoundingPolicy::Away);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_right_shift_2() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(4);
                        let pb = QRightShiftProblem::<$ker, $tc, $ti>::new(v, 3, RoundingPolicy::Odd);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_wrapping_mul_high_doubling_0() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(6);
                        let pb = QWrappingMulHighDoublingProblem::<$ker, $tc, $ti>::new(v, 715827883);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                proptest::proptest! {
                    #[test]
                    fn return_q_towards_plusinf_prop(pb in any::<QTowardsPlusInfProblem<$ker, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }

                    #[test]
                    fn return_q_away_prop(pb in any::<QAwayProblem<$ker, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }

                    #[test]
                    fn return_q_right_shift_prop(pb in any::<QRightShiftProblem<$ker, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }

                    #[test]
                    fn return_q_wrapping_mul_high_doubling_prop(pb in any::<QWrappingMulHighDoublingProblem<$ker, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }
                }
            }
        };
    }

    pub fn null_packed_storage() -> PanelStore {
        PanelStore::Packed { ptr: std::ptr::null() }
    }

    pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize) -> Tile {
        Tile {
            ptr: v.as_mut_ptr() as _,
            row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
            col_byte_stride: std::mem::size_of::<T>() as isize,
            item_size: std::mem::size_of::<T>(),
        }
    }

    pub fn return_zeros<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + Bounded + Zero + PartialEq,
        TI: Copy + Debug,
    {
        let mut v = vec![TC::max_value(); K::mr() * K::nr()];
        let mut c = mmm_stride_storage(&mut v, K::nr());
        let err = K::kernel(&MatMatMulKerSpec {
            a: &null_packed_storage(),
            b: &null_packed_storage(),
            c: &mut c,
            linear: &LinearSpec::k(0),
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        let expected = vec![TC::zero(); v.len()];
        assert_eq!(v, expected);
    }

    pub fn fused_ops<K, TC, TI>(c: &[TC], ops: &[FusedKerSpec<TI>]) -> Vec<TC>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + 'static + PartialEq,
        TI: Copy + Debug,
        usize: AsPrimitive<TC>,
    {
        assert!(c.len() == K::mr() * K::nr());
        let mut v = c.to_vec();
        let mut c = mmm_stride_storage(&mut v, K::nr());
        let mut ops = ops.to_vec();
        ops.insert(0, FusedKerSpec::AddUnicast(c));
        ops.push(FusedKerSpec::Done);
        let err = K::kernel(&MatMatMulKerSpec {
            a: &null_packed_storage(),
            b: &null_packed_storage(),
            c: &mut c,
            linear: &LinearSpec::k(0),
            non_linear: ops.as_ptr(),
        });
        assert_eq!(err, 0);
        v
    }

    pub fn return_c<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + PartialEq,
        TI: Copy + Debug,
        usize: AsPrimitive<TC>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[]);
        assert_eq!(found, v);
    }

    pub fn return_c_plus_d<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + PartialEq,
        TI: Copy + Datum + Debug + 'static + Add<Output = TI> + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let d: Vec<TI> = (0..len).map(|f| ((3 * f) % 7).as_()).collect();
        let expected =
            (0..len).map(|ix| (AsPrimitive::<TI>::as_(ix) + d[ix]).as_()).collect::<Vec<TC>>();
        let found = fused_ops::<K, TC, TI>(
            &*v,
            &[FusedKerSpec::AddUnicast(Tile {
                ptr: d.as_ptr() as _,
                row_byte_stride: (K::nr() * std::mem::size_of::<TI>()) as isize,
                col_byte_stride: (std::mem::size_of::<TI>()) as isize,
                item_size: std::mem::size_of::<TI>(),
            })],
        );
        assert_eq!(found, expected);
    }

    pub fn return_c_mul_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + PartialEq,
        TI: Copy + Add + Mul<Output = TI> + Zero + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::PerRowMul(bias.as_ptr())]);
        let expected = (0..found.len())
            .map(|ix| {
                let row = ix / K::nr();
                let ix: TI = ix.as_();
                (ix * bias[row]).as_()
            })
            .collect::<Vec<TC>>();
        assert_eq!(found, expected);
    }

    pub fn return_c_add_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + PartialEq + 'static,
        TI: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::PerRowAdd(bias.as_ptr())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let row = ix / K::nr();
            let ix: TI = ix.as_();
            a == (ix + bias[row]).as_()
        }));
    }

    pub fn return_c_mul_col<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + 'static + PartialEq,
        TI: Copy + Add + Mul<Output = TI> + Zero + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::nr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::PerColMul(bias.as_ptr())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let col = ix % K::nr();
            let ix: TI = ix.as_();
            a == (ix * bias[col]).as_()
        }));
    }

    pub fn return_c_add_col<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: Copy
            + Add
            + Mul
            + Zero
            + Debug
            + fmt::Display
            + PartialEq
            + 'static
            + AsPrimitive<TC>
            + Debug,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::nr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::PerColAdd(bias.as_ptr())]);
        let expected = (0..found.len())
            .map(|ix| {
                let col = ix % K::nr();
                let ix: TI = ix.as_();
                (ix + bias[col]).as_()
            })
            .collect::<Vec<TC>>();
        assert_eq!(found, expected);
    }

    pub fn return_c_add_row_col_product<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + PartialEq + 'static,
        TI: Copy
            + Add
            + Mul<Output = TI>
            + Zero
            + Debug
            + fmt::Display
            + PartialEq
            + 'static
            + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let rows: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        let cols: Vec<TI> = (0..K::nr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(
            &*v,
            &[FusedKerSpec::AddRowColProducts(rows.as_ptr(), cols.as_ptr())],
        );
        let expected = (0..found.len())
            .map(|ix| {
                let row = ix / K::nr();
                let col = ix % K::nr();
                let ix: TI = ix.as_();
                (ix + cols[col] * rows[row]).as_()
            })
            .collect::<Vec<TC>>();
        assert_eq!(found, expected);
    }

    pub fn return_c_scalar_mul<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + PartialEq + 'static + Debug,
        TI: Copy
            + Add
            + Mul<Output = TI>
            + Zero
            + Debug
            + fmt::Display
            + PartialEq
            + 'static
            + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::ScalarMul(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == (ix * 5.as_()).as_()
        }));
    }

    pub fn return_c_max<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + PartialEq + 'static,
        TI: Copy
            + Add
            + Mul<Output = TI>
            + std::cmp::PartialOrd
            + Zero
            + Debug
            + fmt::Display
            + PartialEq
            + 'static
            + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::Max(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == if ix > 5.as_() { ix.as_() } else { 5.as_() }
        }));
    }

    pub fn return_c_min<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + PartialEq + 'static,
        TI: Copy
            + Add
            + Mul<Output = TI>
            + std::cmp::PartialOrd
            + Zero
            + Debug
            + fmt::Display
            + PartialEq
            + 'static
            + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::Min(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == if ix < 5.as_() { ix.as_() } else { 5.as_() }
        }));
    }

    pub fn return_c_scalar_add<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy + PartialEq + 'static,
        TI: Copy
            + Add
            + Mul<Output = TI>
            + Zero
            + Debug
            + fmt::Display
            + PartialEq
            + 'static
            + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TC, TI>(&*v, &[FusedKerSpec::ScalarAdd(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == (ix + 5.as_()).as_()
        }));
    }

    pub fn return_q_towards_even<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: Copy
            + PartialEq
            + 'static
            + Bounded
            + Debug
            + Sub<Output = TC>
            + AsPrimitive<TI>
            + Mul<Output = TC>,
        TI: Copy
            + Add
            + Sub<Output = TI>
            + Mul<Output = TI>
            + Debug
            + fmt::Display
            + Ord
            + PartialEq
            + 'static
            + AsPrimitive<TC>
            + AsPrimitive<i64>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        let len = K::mr() * K::nr();
        let half_len: TI = (len / 2).as_();
        let v: Vec<TC> = (0..len)
            .map(|f| {
                (<usize as AsPrimitive<TI>>::as_(f) - half_len)
                    .min(TC::max_value().as_())
                    .max(TC::min_value().as_())
                    .as_()
            })
            .collect();
        let found = fused_ops::<K, TC, TI>(
            &*v,
            &[FusedKerSpec::ScalarMul(2.as_()), FusedKerSpec::QTowardsEven((1 << 30).as_(), 2)],
        );
        assert!(found.iter().zip(v.iter()).all(|(&found, input)| {
            let input: TI = input.as_();
            let input: i64 = input.as_();
            let input = input >> 1;
            let trunc = input.abs();
            let nudge = (trunc & 0x3 == 0x3) as i64;
            let mut trunc = (trunc + nudge) >> 1;
            if input.is_negative() {
                trunc = -trunc;
            }
            trunc.as_() == found
        }));
    }

    #[derive(Debug, new)]
    pub struct QTowardsPlusInfProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        pub c: Vec<TC>,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for QTowardsPlusInfProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_p: ()) -> Self::Strategy {
            let len = K::mr() * K::nr();
            proptest::collection::vec((-20i64..20).prop_map(|i| i.as_()), len..=len)
                .prop_map(|c| QTowardsPlusInfProblem { c, boo: std::marker::PhantomData })
                .boxed()
        }
    }

    impl<K, TC, TI> QTowardsPlusInfProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + AsPrimitive<TI> + PartialEq,
        TI: Copy + Debug + 'static + AsPrimitive<i64>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        pub fn reference(&self) -> Vec<TC> {
            self.c
                .iter()
                .map(|input| {
                    let input: TI = input.as_();
                    let input: i64 = input.as_();
                    (((input >> 1) + 1) >> 1).as_()
                })
                .collect()
        }

        pub fn run(&self) -> Vec<TC> {
            fused_ops::<K, TC, TI>(
                &*self.c,
                &[
                    FusedKerSpec::ScalarMul(2.as_()),
                    FusedKerSpec::QTowardsPlusInf((1 << 30).as_(), 2),
                ],
            )
        }
    }

    #[derive(Debug, new)]
    pub struct QAwayProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        pub c: Vec<TC>,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for QAwayProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_p: ()) -> Self::Strategy {
            let len = K::mr() * K::nr();
            proptest::collection::vec((-20i64..20).prop_map(|i| i.as_()), len..=len)
                .prop_map(|c| QAwayProblem { c, boo: std::marker::PhantomData })
                .boxed()
        }
    }

    impl<K, TC, TI> QAwayProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + AsPrimitive<TI> + PartialEq,
        TI: Copy + Debug + 'static + AsPrimitive<i64>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        pub fn reference(&self) -> Vec<TC> {
            self.c
                .iter()
                .map(|input| {
                    let input: TI = input.as_();
                    let input: i64 = input.as_();
                    ((((input.abs() >> 1) + 1) >> 1) * input.signum()).as_()
                })
                .collect()
        }

        pub fn run(&self) -> Vec<TC> {
            fused_ops::<K, TC, TI>(
                &*self.c,
                &[FusedKerSpec::ScalarMul(2.as_()), FusedKerSpec::QAway((1 << 30).as_(), 2)],
            )
        }
    }

    #[derive(Debug, new)]
    pub struct QRightShiftProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        pub c: Vec<TC>,
        pub shift: usize,
        pub policy: RoundingPolicy,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for QRightShiftProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_p: ()) -> Self::Strategy {
            use RoundingPolicy::*;
            let len = K::mr() * K::nr();
            (
                proptest::collection::vec((-20i64..20).prop_map(|i| i.as_()), len..=len),
                0usize..8,
                proptest::prop_oneof![
                    Just(Zero),
                    Just(Away),
                    Just(PlusInf),
                    Just(MinusInf),
                    Just(Odd),
                    Just(Even)
                ],
            )
                .prop_map(|(c, shift, policy)| QRightShiftProblem {
                    c,
                    shift,
                    policy,
                    boo: std::marker::PhantomData,
                })
                .boxed()
        }
    }

    impl<K, TC, TI> QRightShiftProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + AsPrimitive<TI> + PartialEq,
        TI: Copy + Debug + 'static + AsPrimitive<i64> + PseudoRightShift + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        pub fn reference(&self) -> Vec<TC> {
            self.c.iter().map(|input| (input.as_()).shift(self.shift, self.policy).as_()).collect()
        }

        pub fn run(&self) -> Vec<TC> {
            fused_ops::<K, TC, TI>(
                &*self.c,
                &[FusedKerSpec::QShiftRightRounding(self.shift, self.policy)],
            )
        }
    }

    #[derive(Debug, new)]
    pub struct QWrappingMulHighDoublingProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        pub c: Vec<TC>,
        pub mult: i32,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for QWrappingMulHighDoublingProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_p: ()) -> Self::Strategy {
            let len = K::mr() * K::nr();
            (
                proptest::collection::vec((-20i64..20).prop_map(|i| (3 * i).as_()), len..=len),
                (1 << 29)..(1 << 30),
            )
                .prop_map(|(c, mult)| QWrappingMulHighDoublingProblem {
                    c,
                    mult,
                    boo: std::marker::PhantomData,
                })
                .boxed()
        }
    }

    impl<K, TC, TI> QWrappingMulHighDoublingProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + AsPrimitive<TI> + PartialEq,
        TI: Copy + Debug + 'static + AsPrimitive<i64> + PseudoRightShift + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        pub fn reference(&self) -> Vec<TC> {
            self.c.iter().map(|input| (input.as_()).doubling_mul(self.mult).as_()).collect()
        }

        pub fn run(&self) -> Vec<TC> {
            fused_ops::<K, TC, TI>(&*self.c, &[FusedKerSpec::QWrappingMulHighDoubling(self.mult)])
        }
    }

    #[derive(Debug, new)]
    pub struct ReturnCProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
    {
        pub c: Vec<TC>,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for ReturnCProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + Arbitrary,
        TI: Copy + Debug,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_p: ()) -> Self::Strategy {
            let len = K::mr() * K::nr();
            proptest::collection::vec(any::<TC>(), len..=len)
                .prop_map(|c| ReturnCProblem { c, boo: std::marker::PhantomData })
                .boxed()
        }
    }

    impl<K, TC, TI> ReturnCProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + AsPrimitive<TI> + PartialEq,
        TI: Copy + Debug + 'static + AsPrimitive<i64>,
        usize: AsPrimitive<TC>,
    {
        pub fn run(&self) -> Vec<TC> {
            fused_ops::<K, TC, TI>(&*self.c, &[])
        }
    }
}
