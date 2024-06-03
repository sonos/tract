use std::fmt::Debug;

use super::{MMMInput, OutputStore, OutputStoreKer};
use tract_data::internal::*;

#[repr(usize)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RoundingPolicy {
    Native,
    Zero,
    Away,
    MinusInf,
    PlusInf,
    Even,
    Odd,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Min,
    Max,
    Add,
    Mul,
    Sub,
    SubF,
}

impl BinOp {
    pub fn flip(&self) -> BinOp {
        use BinOp::*;
        match self {
            Sub => SubF,
            SubF => Sub,
            sym => *sym,
        }
    }
}

#[derive(Clone, Debug)]
pub enum FusedSpec<'t> {
    BinScalar(&'t Tensor, BinOp),
    BinPerRow(TensorView<'t>, BinOp),
    BinPerCol(TensorView<'t>, BinOp),
    AddRowColProducts(&'t Tensor, &'t Tensor),
    AddUnicast(OutputStore),
    LeakyRelu(&'t Tensor),
    QScale(isize, RoundingPolicy, i32),
    RoundingShiftRight(usize, RoundingPolicy),
    ShiftLeft(usize),
    Store(OutputStore),
    AddMatMul { a: &'t dyn MMMInput, b: &'t dyn MMMInput },
}

impl<'t> FusedSpec<'t> {
    pub fn prefer_col_outer(&self) -> bool {
        false
        /*
        if let FusedSpec::AddMatMul { b, .. } = self {
        match b {
        InputStore::Packed { .. } => false,
        InputStore::VirtualPacking { .. } => true,
        }
        } else {
        false
        }
        */
    }
}

// Careful here, the jump_to comments are used by the build script.
#[repr(C, usize)]
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[rustfmt::skip]
pub enum FusedKerSpec<TI: Copy> {
    Done,                                       // jump_to:done
    Clear,                                      // jump_to:clear

    ScalarMin(TI),                              // jump_to:scalar_min
    ScalarMax(TI),                              // jump_to:scalar_max
    ScalarAdd(TI),                              // jump_to:scalar_add
    ScalarMul(TI),                              // jump_to:scalar_mul
    ScalarSub(TI),                              // jump_to:scalar_sub
    ScalarSubF(TI),                             // jump_to:scalar_sub_flipped

    LeakyRelu(TI),                              // jump_to:leaky_relu

    PerRowMin(*const TI),                       // jump_to:per_row_min
    PerRowMax(*const TI),                       // jump_to:per_row_max
    PerRowAdd(*const TI),                       // jump_to:per_row_add
    PerRowMul(*const TI),                       // jump_to:per_row_mul
    PerRowSub(*const TI),                       // jump_to:per_row_sub
    PerRowSubF(*const TI),                      // jump_to:per_row_sub_flipped

    PerColMin(*const TI),                       // jump_to:per_col_min
    PerColMax(*const TI),                       // jump_to:per_col_max
    PerColAdd(*const TI),                       // jump_to:per_col_add
    PerColMul(*const TI),                       // jump_to:per_col_mul
    PerColSub(*const TI),                       // jump_to:per_col_sub
    PerColSubF(*const TI),                      // jump_to:per_col_sub_flipped

    QScale(isize, RoundingPolicy, i32),         // jump_to:q_scale
    RoundingShiftRight(usize, RoundingPolicy),  // jump_to:q_shr
    ShiftLeft(usize),                           // jump_to:q_shl
    AddUnicast(OutputStoreKer),                 // jump_to:add_unicast
    AddRowColProducts(*const TI, *const TI),    // jump_to:add_row_col_products
    Store(OutputStoreKer),                      // jump_to:store

    // jump_to:add_mat_mul
    AddMatMul { k: usize, pa: *const u8, pb: *const u8, cpu_variant: usize },
}

unsafe impl<TI: Copy> Send for FusedKerSpec<TI> {}
unsafe impl<TI: Copy> Sync for FusedKerSpec<TI> {}

#[cfg(test)]
#[macro_use]
pub mod test {
    use crate::frame::mmm::storage::*;
    use crate::frame::mmm::*;
    use crate::generic::{ScaleShiftAndRound, Scaler};
    use num_traits::{AsPrimitive, Bounded};
    use proptest::prelude::*;
    use tract_data::internal::*;

    #[test]
    fn check_non_linear_enum_size() {
        assert_eq!(std::mem::size_of::<RoundingPolicy>(), std::mem::size_of::<usize>());
        assert_eq!(
            std::mem::size_of::<super::FusedKerSpec<f32>>(),
            std::mem::size_of::<usize>() + std::mem::size_of::<OutputStoreKer>()
        );
        assert_eq!(
            std::mem::size_of::<super::FusedKerSpec<f32>>(),
            5 * std::mem::size_of::<usize>()
        );
    }

    #[macro_export]
    macro_rules! mmm_kernel_fuse_tests {
        ($cond:expr, $ker:ident, $tc:ty, $ti: ty) => {
            mod fuse {
                use super::$ker;
                use num_traits::Zero;
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                use tract_data::prelude::tensor0;
                #[allow(unused_imports)]
                use $crate::frame::mmm::fuse::test;
                use $crate::frame::mmm::fuse::test::tile;
                use $crate::frame::mmm::fuse::FusedKerSpec;
                use $crate::frame::mmm::{FusedSpec, MatMatMulKer};

                #[test]
                fn return_zeros() {
                    if $cond {
                        test::return_zeros::<_, $tc, $ti>($ker)
                    }
                }

                #[test]
                fn store_non_contiguous() {
                    if $cond {
                        test::store_non_contiguous::<_, $tc, $ti>($ker)
                    }
                }
                proptest::proptest! {
                #[test]
                fn return_c_prop(c in tile::<_, $tc, $ti>($ker)) {
                if $cond {
                test::return_c::<_, $tc, $ti>($ker, &c)
                }
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
                                if $cond && $extra_cond {
                                    test::$geo::<_, $tc, $ti>($ker, FusedKerSpec::$FKS, $f);
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
                    $ker.can_fuse(&FusedSpec::LeakyRelu(&tensor0(<$ti>::from(1_u8))))
                );

                #[test]
                fn return_c_add_row_col_product() {
                    if $cond {
                        test::return_c_add_row_col_product::<_, $tc, $ti>($ker)
                    }
                }

                #[test]
                fn return_c_plus_d() {
                    if $cond {
                        test::return_c_plus_d::<_, $tc, $ti>($ker)
                    }
                }

                #[test]
                fn return_c_clear() {
                    if $cond {
                        test::return_c_clear::<_, $tc, $ti>($ker)
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! qmmm_kernel_fuse_tests {
        ($cond:expr, $ker:ident, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod fuseq {
                use $crate::frame::mmm::fuse::RoundingPolicy;
                #[allow(unused_imports)]
                use $crate::frame::mmm::fuse::test;
                use $crate::frame::mmm::fuse::test::QScaleProblem;
                use $crate::frame::mmm::kernel::MatMatMulKer;
                use $crate::generic::Scaler;
                use proptest::prelude::*;
                use super::$ker;

                // FIXME: Scaler should be arbitrary
                macro_rules! test_q_scale {
                    ($policy: ident) => {
                        paste! {
                            #[test]
                            fn [<return_q_scale_halfpos_ $policy:lower>]() {
                                if $cond {
                                    let len = ($ker.mr() * $ker.nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<_, $tc, $ti>::new($ker, v, Scaler::new(0.5f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_halfneg_ $policy:lower>]() {
                                if $cond {
                                    let len = ($ker.mr() * $ker.nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<_, $tc, $ti>::new($ker, v, Scaler::new(-0.5f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_pot_ $policy:lower>]() {
                                if $cond {
                                    let len = ($ker.mr() * $ker.nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<_, $tc, $ti>::new($ker, v, Scaler::new(0.25f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_nonpot_ $policy:lower>]() {
                                if $cond {
                                    let len = ($ker.mr() * $ker.nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<_, $tc, $ti>::new($ker, v, Scaler::new(1f32 / 5., RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_bigpot_ $policy:lower>]() {
                                if $cond {
                                    let len = ($ker.mr() * $ker.nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<_, $tc, $ti>::new($ker, v, Scaler::new(4f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_bignonpot_ $policy:lower>]() {
                                if $cond {
                                    let len = ($ker.mr() * $ker.nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<_, $tc, $ti>::new($ker, v, Scaler::new(14., RoundingPolicy::$policy)).run()
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
                    fn return_q_scale_prop(pb in any_with::<QScaleProblem<_, $tc, $ti>>($ker)) {
                        if $cond {
                            pb.run()
                        }
                    }
                }

                #[test]
                fn return_c_scale_bigpot() {
                    if $cond {
                        test::return_c_scale_bigpot::<_, $tc, $ti>($ker)
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

    use crate::LADatum;
    pub fn return_zeros<K, TC, TI>(ker: K)
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
        assert_eq!(v, expected);
    }

    pub fn store_non_contiguous<K, TC, TI>(ker: K)
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

    pub fn fused_ops<K, TC, TI, E>(ker: K, c: &[TC], ops: &[FusedKerSpec<TI>], expect: E)
    where
        K: MatMatMulKer<Acc = TI>,
        TC: Datum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
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
        let expected = (0..v.len())
            .map(|ix| expect(ix / ker.nr(), ix % ker.nr(), v[ix].as_()).as_())
            .collect::<Vec<TC>>();
        let err = ker.kernel(&ops);
        assert_eq!(err, 0);
        if v != expected {
            println!("found, expected:");
            for m in 0..ker.mr() {
                for n in 0..ker.nr() {
                    use nu_ansi_term::Color::*;
                    let f = v[m * ker.nr() + n];
                    let e = expected[m * ker.nr() + n];
                    let color = if f != e { Red } else { Green };
                    print!("{} ", color.paint(format!("{:4}", f)));
                }
                print!("      ");
                for n in 0..ker.nr() {
                    print!("{:4} ", expected[m * ker.nr() + n]);
                }
                println!();
            }
        }
        assert_eq!(v, expected);
    }

    pub fn return_c<K, TC, TI>(ker: K, v: &[TC])
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        fused_ops::<K, TC, TI, _>(ker, v, &[], |_, _, c| c + 1.as_() - 1.as_())
    }

    pub fn return_c_plus_d<K, TC, TI>(ker: K)
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = ker.mr() * ker.nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let d: Vec<TI> = (0..len).map(|f| ((3 * f) % 7).as_()).collect();
        fused_ops::<K, TC, TI, _>(
            ker,
            &v,
            &[FusedKerSpec::AddUnicast(mmm_stride_storage(&d, ker.nr()))],
            |row, col, c| c + d[row * ker.nr() + col],
        );
    }

    pub fn per_col<K, TC, TI>(
        ker: K,
        op: impl Fn(*const TI) -> FusedKerSpec<TI>,
        f: impl Fn(TI, TI) -> TI,
    ) where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = ker.mr() * ker.nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..ker.nr()).map(|f| (f + 1).as_()).collect();
        fused_ops::<K, TC, TI, _>(ker, &v, &[op(bias.as_ptr())], |_, col, c| f(bias[col], c))
    }

    pub fn per_row<K, TC, TI>(
        ker: K,
        op: impl Fn(*const TI) -> FusedKerSpec<TI>,
        f: impl Fn(TI, TI) -> TI,
    ) where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = ker.mr() * ker.nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..ker.mr()).map(|f| (f + 1).as_()).collect();
        fused_ops::<K, TC, TI, _>(ker, &v, &[op(bias.as_ptr())], |row, _, c| f(bias[row], c))
    }

    pub fn scalar<K, TC, TI>(ker: K, op: impl Fn(TI) -> FusedKerSpec<TI>, f: impl Fn(TI, TI) -> TI)
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        isize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = ker.mr() * ker.nr();
        let v: Vec<TC> = (0..len as isize).map(|f| (f - len as isize / 2).as_()).collect();
        let five: TI = 5.as_();
        fused_ops::<K, TC, TI, _>(ker, &v, &[op(five)], |_, _, c| f(five, c))
    }

    pub fn return_c_add_row_col_product<K, TC, TI>(ker: K)
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = ker.mr() * ker.nr();
        let v: Vec<TC> = (0..len).map(|f| (f + 1).as_()).collect();
        let rows: Vec<TI> = (0..ker.mr()).map(|f| (f + 3).as_()).collect();
        let cols: Vec<TI> = (0..ker.nr()).map(|f| (f + 2).as_()).collect();
        fused_ops::<K, TC, TI, _>(
            ker,
            &v,
            &[FusedKerSpec::AddRowColProducts(rows.as_ptr(), cols.as_ptr())],
            |row, col, c| c + cols[col] * rows[row],
        )
    }

    pub fn return_c_clear<K, TC, TI>(ker: K)
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = ker.mr() * ker.nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(ker, &v, &[FusedKerSpec::Clear], |_, _, _| 0.as_())
    }

    pub fn return_c_scale_bigpot<K, TC, TI>(ker: K)
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC> + ScaleShiftAndRound,
        isize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = ker.mr() * ker.nr();
        let v: Vec<TC> = (-(len as isize) / 2..).take(len).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(ker, &v, &[FusedKerSpec::ShiftLeft(1)], |_, _, c| c.q_shl(1))
    }

    #[derive(Debug, new)]
    pub struct QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum,
        TI: LADatum + AsPrimitive<TC>,
        i64: AsPrimitive<TC>,
    {
        pub ker: K,
        pub c: Vec<TC>,
        pub scaler: Scaler,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<Acc = TI> + 'static + Copy + Default,
        TC: LADatum + Arbitrary,
        TI: LADatum + AsPrimitive<TC>,
        i64: AsPrimitive<TC>,
    {
        type Parameters = K;
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(ker: K) -> Self::Strategy {
            use RoundingPolicy::*;
            let len = ker.mr() * ker.nr();
            (
                proptest::collection::vec((-20i64..20).prop_map(|i| i.as_()), len..=len),
                -5i32..5,
                prop_oneof!(Just(1f32), 0f32..1f32),
                proptest::prop_oneof![
                    Just(Zero),
                    Just(Away),
                    Just(PlusInf),
                    Just(MinusInf),
                    Just(Odd),
                    Just(Even)
                ],
            )
                .prop_map(|(c, scale_pot, scale_mult, policy)| QScaleProblem {
                    ker: K::default(),
                    c,
                    scaler: Scaler::new(scale_mult * 2f32.powi(scale_pot), policy),
                    boo: std::marker::PhantomData,
                })
                .boxed()
        }
    }

    impl<K, TC, TI> QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC> + ScaleShiftAndRound + AsPrimitive<i64>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        pub fn run(&self) {
            if let FusedSpec::QScale(shift, policy, mult) = self.scaler.as_fused_spec() {
                fused_ops::<K, TC, TI, _>(
                    self.ker,
                    &self.c,
                    &[FusedKerSpec::QScale(shift, policy, mult)],
                    |_, _, c| c.q_scale(self.scaler),
                )
            } else if let FusedSpec::RoundingShiftRight(shift, policy) = self.scaler.as_fused_spec()
            {
                fused_ops::<K, TC, TI, _>(
                    self.ker,
                    &self.c,
                    &[FusedKerSpec::RoundingShiftRight(shift, policy)],
                    |_, _, c| c.q_shr(shift, policy),
                )
            } else if let FusedSpec::ShiftLeft(shift) = self.scaler.as_fused_spec() {
                fused_ops::<K, TC, TI, _>(
                    self.ker,
                    &self.c,
                    &[FusedKerSpec::ShiftLeft(shift)],
                    |_, _, c| c.q_shl(shift),
                )
            } else {
                unreachable!()
            }
        }
    }

    pub fn tile<K, TC, TI>(ker: K) -> BoxedStrategy<Vec<TC>>
    where
        K: MatMatMulKer<Acc = TI>,
        TC: LADatum,
        TI: LADatum + AsPrimitive<TC>,
        i8: AsPrimitive<TC>,
    {
        let len = ker.mr() * ker.nr();
        proptest::collection::vec(any::<i8>().prop_map(|c| c.as_()), len..=len).boxed()
    }
}
