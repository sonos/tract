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
    AddUnicast(TensorView<'t>),
    QScale(usize, RoundingPolicy, i32),
}

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
    QScale(usize, RoundingPolicy, i32),
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::frame::mmm::storage::*;
    use crate::frame::mmm::*;
    use crate::generic::ScaleShiftAndRound;
    use num_traits::{AsPrimitive, Bounded, Zero};
    use proptest::prelude::*;
    use std::fmt;
    use std::ops::{Add, Mul};

    #[test]
    fn check_non_linear_enum_size() {
        assert_eq!(std::mem::size_of::<RoundingPolicy>(), std::mem::size_of::<usize>());
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
                use crate::frame::mmm::fuse::test::QScaleProblem;
                use crate::frame::mmm::kernel::MatMatMulKer;
                use proptest::prelude::*;

                #[test]
                fn return_q_scale_0() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(1);
                        let pb = QScaleProblem::<$ker, $tc, $ti>::new(v, 2^31, 1, RoundingPolicy::Zero);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_scale_1() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(1);
                        let pb = QScaleProblem::<$ker, $tc, $ti>::new(v, 2^31, 1, RoundingPolicy::Away);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_scale_2() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(4);
                        let pb = QScaleProblem::<$ker, $tc, $ti>::new(v, 2^31, 3, RoundingPolicy::Odd);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_scale_3() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(1);
                        let pb = QScaleProblem::<$ker, $tc, $ti>::new(v, 536870913, 0, RoundingPolicy::Zero);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }
                #[test]
                fn return_q_scale_4() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(2);
                        let pb = QScaleProblem::<$ker, $tc, $ti>::new(v, 536870913, 0, RoundingPolicy::Zero);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_scale_5() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(6);
                        let pb = QScaleProblem::<$ker, $tc, $ti>::new(v, 715827883, 0, RoundingPolicy::Even);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                macro_rules! test_q_scale {
                    ($policy: ident) => {
                        paste! {
                            #[test]
                            fn [<return_q_scale_ $policy:lower>]() {
                                if $cond {
                                    let len = (<$ker>::mr() * <$ker>::nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    let pb = QScaleProblem::<$ker, $tc, $ti>::new(v, 1<<30, 2, RoundingPolicy::$policy);
                                    assert_eq!(pb.run(), pb.reference())
                              //      }
                                }
                            }
                        }
                    }
                }

                test_q_scale!(Zero);
                test_q_scale!(Away);
                test_q_scale!(MinusInf);
                test_q_scale!(PlusInf);
                test_q_scale!(Even);
                test_q_scale!(Odd);

                proptest::proptest! {
                    #[test]
                    fn return_q_scale_prop(pb in any::<QScaleProblem<$ker, $tc, $ti>>()) {
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

    #[derive(Debug, new)]
    pub struct QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        pub c: Vec<TC>,
        pub mult: i32,
        pub shift: usize,
        pub policy: RoundingPolicy,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + Arbitrary + 'static,
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
                (1 << 29)..(1 << 30),
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
                .prop_map(|(c, mult, shift, policy)| QScaleProblem {
                    c,
                    shift,
                    policy,
                    mult,
                    boo: std::marker::PhantomData,
                })
                .boxed()
        }
    }

    impl<K, TC, TI> QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: Copy + Debug + 'static + AsPrimitive<TI> + PartialEq,
        TI: Copy + Debug + 'static + AsPrimitive<i64> + ScaleShiftAndRound + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        pub fn reference(&self) -> Vec<TC> {
            self.c
                .iter()
                .map(|input| (input.as_()).q_scale(self.mult, self.shift, self.policy).as_())
                .collect()
        }

        pub fn run(&self) -> Vec<TC> {
            fused_ops::<K, TC, TI>(
                &*self.c,
                &[FusedKerSpec::QScale(self.shift, self.policy, self.mult)],
            )
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
