use std::fmt::Debug;

use super::{InputStore, OutputStore, OutputStoreKer, PackedStore};
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

#[derive(Copy, Clone, Debug, PartialEq, Hash)]
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
    BinPerRow(&'t Tensor, BinOp),
    BinPerCol(&'t Tensor, BinOp),
    AddRowColProducts(&'t Tensor, &'t Tensor),
    AddUnicast(OutputStore),
    QScale(isize, RoundingPolicy, i32),
    RoundingShiftRight(usize, RoundingPolicy),
    ShiftLeft(usize),
    Store(OutputStore),
    AddMatMul { k: usize, a: PackedStore, b: InputStore },
}

impl<'t> FusedSpec<'t> {
    pub fn prefer_col_outer(&self) -> bool {
        if let FusedSpec::AddMatMul { b, .. } = self {
            match &b {
                &InputStore::Packed { .. } => false,
                &InputStore::VirtualPacking { .. } => true,
                &InputStore::LatePacking { .. } => true,
            }
        } else {
            false
        }
    }
}

// Careful here, the jump_to comments are used by the build script.
#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
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
                use super::super::$ker;
                #[allow(unused_imports)]
                use crate::frame::mmm::fuse::test;
                use crate::frame::mmm::fuse::test::tile;
                #[allow(unused_imports)]
                use tract_data::prelude::f16;

                #[test]
                fn return_zeros() {
                    if $cond {
                        test::return_zeros::<$ker, $tc, $ti>()
                    }
                }

                proptest::proptest! {
                    #[test]
                    fn return_c_prop(c in tile::<$ker, $tc, $ti>()) {
                        if $cond {
                            test::return_c::<$ker, $tc, $ti>(&c)
                        }
                    }
                }

                #[test]
                fn return_c_min_row() {
                    if $cond {
                        test::return_c_min_row::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_max_row() {
                    if $cond {
                        test::return_c_max_row::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_add_row() {
                    if $cond {
                        test::return_c_add_row::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_mul_row() {
                    if $cond {
                        test::return_c_mul_row::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_sub_row() {
                    if $cond {
                        test::return_c_sub_row::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_subf_row() {
                    if $cond {
                        test::return_c_subf_row::<$ker, $tc, $ti>()
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
                fn return_c_scalar_max() {
                    if $cond {
                        test::return_c_scalar_max::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_min() {
                    if $cond {
                        test::return_c_scalar_min::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_add() {
                    if $cond {
                        test::return_c_scalar_add::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_mul() {
                    if $cond {
                        test::return_c_scalar_mul::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_sub() {
                    if $cond {
                        test::return_c_scalar_sub::<$ker, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_subf() {
                    if $cond {
                        test::return_c_scalar_subf::<$ker, $tc, $ti>()
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
        ($cond:expr, $ker:ident, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod fuseq {
                use crate::frame::mmm::fuse::RoundingPolicy;
                #[allow(unused_imports)]
                use crate::frame::mmm::fuse::test;
                use crate::frame::mmm::fuse::test::QScaleProblem;
                use crate::frame::mmm::kernel::MatMatMulKer;
                use crate::generic::Scaler;
                use proptest::prelude::*;
                use super::super::$ker;

                // FIXME: Scaler should be arbitrary
                macro_rules! test_q_scale {
                    ($policy: ident) => {
                        paste! {
                            #[test]
                            fn [<return_q_scale_halfpos_ $policy:lower>]() {
                                if $cond {
                                    let len = (<$ker>::mr() * <$ker>::nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<$ker, $tc, $ti>::new(v, Scaler::new(0.5f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_halfneg_ $policy:lower>]() {
                                if $cond {
                                    let len = (<$ker>::mr() * <$ker>::nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<$ker, $tc, $ti>::new(v, Scaler::new(-0.5f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_pot_ $policy:lower>]() {
                                if $cond {
                                    let len = (<$ker>::mr() * <$ker>::nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<$ker, $tc, $ti>::new(v, Scaler::new(0.25f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_nonpot_ $policy:lower>]() {
                                if $cond {
                                    let len = (<$ker>::mr() * <$ker>::nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<$ker, $tc, $ti>::new(v, Scaler::new(1f32 / 5., RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_bigpot_ $policy:lower>]() {
                                if $cond {
                                    let len = (<$ker>::mr() * <$ker>::nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<$ker, $tc, $ti>::new(v, Scaler::new(4f32, RoundingPolicy::$policy)).run()
                                }
                            }

                            #[test]
                            fn [<return_q_scale_bignonpot_ $policy:lower>]() {
                                if $cond {
                                    let len = (<$ker>::mr() * <$ker>::nr()) as i64;
                                    let v = (0..len).map(|i| (i - len / 2) as $tc).collect();
                                    QScaleProblem::<$ker, $tc, $ti>::new(v, Scaler::new(14., RoundingPolicy::$policy)).run()
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
                            pb.run()
                        }
                    }
                }

                #[test]
                fn return_c_scale_bigpot() {
                    if $cond {
                        test::return_c_scale_bigpot::<$ker, $tc, $ti>()
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
    pub fn return_zeros<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum,
        TI: LADatum + Bounded + PartialEq,
    {
        let mut v = vec![TC::max_value(); K::mr() * K::nr()];
        let c = mmm_stride_storage(&mut v, K::nr());
        let non_linear = tvec![FusedKerSpec::Clear, FusedKerSpec::Store(c), FusedKerSpec::Done];
        let err = K::kernel(&non_linear);
        assert_eq!(err, 0);
        let expected = vec![TC::zero(); v.len()];
        assert_eq!(v, expected);
    }

    pub fn fused_ops<K, TC, TI, E>(c: &[TC], ops: &[FusedKerSpec<TI>], expect: E)
    where
        K: MatMatMulKer<TI>,
        TC: Datum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        E: Fn(usize, usize, TI) -> TI,
    {
        assert!(c.len() == K::mr() * K::nr());
        let mut v = c.to_vec();
        let c = mmm_stride_storage(&mut v, K::nr());
        let mut ops = ops.to_vec();
        ops.insert(0, FusedKerSpec::AddUnicast(c));
        ops.insert(0, FusedKerSpec::Clear);
        ops.push(FusedKerSpec::Store(c));
        ops.push(FusedKerSpec::Done);
        let expected = (0..v.len())
            .map(|ix| expect(ix / K::nr(), ix % K::nr(), v[ix].as_()).as_())
            .collect::<Vec<TC>>();
        let err = K::kernel(&ops);
        assert_eq!(err, 0);
        if v != expected {
            println!("found, expected:");
            for m in 0..K::mr() {
                for n in 0..K::nr() {
                    print!("{:4} ", v[m * K::nr() + n]);
                }
                print!("      ");
                for n in 0..K::nr() {
                    print!("{:4} ", expected[m * K::nr() + n]);
                }
                println!();
            }
        }
        assert_eq!(v, expected);
    }

    pub fn return_c<K, TC, TI>(v: &[TC])
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        fused_ops::<K, TC, TI, _>(&*v, &[], |_, _, c| c + 1.as_() - 1.as_())
    }

    pub fn return_c_plus_d<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let d: Vec<TI> = (0..len).map(|f| ((3 * f) % 7).as_()).collect();
        fused_ops::<K, TC, TI, _>(
            &*v,
            &[FusedKerSpec::AddUnicast(mmm_stride_storage(&d, K::nr()))],
            |row, col, c| c + d[row * K::nr() + col],
        );
    }

    pub fn return_c_min_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerRowMin(bias.as_ptr())], |row, _, c| {
            if c < bias[row] {
                c
            } else {
                bias[row]
            }
        })
    }

    pub fn return_c_max_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerRowMax(bias.as_ptr())], |row, _, c| {
            if c > bias[row] {
                c
            } else {
                bias[row]
            }
        })
    }

    pub fn return_c_add_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerRowAdd(bias.as_ptr())], |row, _, c| {
            c + bias[row]
        })
    }

    pub fn return_c_mul_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerRowMul(bias.as_ptr())], |row, _, c| {
            c * bias[row]
        })
    }

    pub fn return_c_sub_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerRowSub(bias.as_ptr())], |row, _, c| {
            bias[row] - c
        })
    }

    pub fn return_c_subf_row<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerRowSubF(bias.as_ptr())], |row, _, c| {
            c - bias[row]
        })
    }

    pub fn return_c_add_col<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::nr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerColAdd(bias.as_ptr())], |_, col, c| {
            c + bias[col]
        })
    }

    pub fn return_c_mul_col<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::nr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::PerColMul(bias.as_ptr())], |_, col, c| {
            c * bias[col]
        })
    }

    pub fn return_c_add_row_col_product<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let rows: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        let cols: Vec<TI> = (0..K::nr()).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(
            &*v,
            &[FusedKerSpec::AddRowColProducts(rows.as_ptr(), cols.as_ptr())],
            |row, col, c| c + cols[col] * rows[row],
        )
    }

    pub fn return_c_scalar_min<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::ScalarMin(5.as_())], |_, _, c| {
            if c > 5.as_() {
                5.as_()
            } else {
                c
            }
        })
    }

    pub fn return_c_scalar_max<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::ScalarMax(5.as_())], |_, _, c| {
            if c < 5.as_() {
                5.as_()
            } else {
                c
            }
        })
    }

    pub fn return_c_scalar_add<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::ScalarAdd(5.as_())], |_, _, c| c + 5.as_())
    }

    pub fn return_c_scalar_mul<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::ScalarMul(5.as_())], |_, _, c| c * 5.as_())
    }

    pub fn return_c_scalar_sub<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let five: TI = 5.as_();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::ScalarSub(5.as_())], |_, _, c| five - c)
    }

    pub fn return_c_scalar_subf<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let five: TI = 5.as_();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::ScalarSubF(5.as_())], |_, _, c| c - five)
    }

    pub fn return_c_scale_bigpot<K, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC> + ScaleShiftAndRound,
        isize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (-(len as isize) / 2..).take(len).map(|f| f.as_()).collect();
        fused_ops::<K, TC, TI, _>(&*v, &[FusedKerSpec::ShiftLeft(1)], |_, _, c| c.q_shl(1))
    }

    #[derive(Debug, new)]
    pub struct QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: LADatum,
        TI: LADatum + AsPrimitive<TC>,
        i64: AsPrimitive<TC>,
    {
        pub c: Vec<TC>,
        pub scaler: Scaler,
        pub boo: std::marker::PhantomData<(K, TC, TI)>,
    }

    impl<K, TC, TI> Arbitrary for QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + Arbitrary,
        TI: LADatum + AsPrimitive<TC>,
        i64: AsPrimitive<TC>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_p: ()) -> Self::Strategy {
            use RoundingPolicy::*;
            let len = K::mr() * K::nr();
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
                    c,
                    scaler: Scaler::new(scale_mult * 2f32.powi(scale_pot), policy),
                    boo: std::marker::PhantomData,
                })
                .boxed()
        }
    }

    impl<K, TC, TI> QScaleProblem<K, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TC: LADatum + AsPrimitive<TI>,
        TI: LADatum + AsPrimitive<TC> + ScaleShiftAndRound + AsPrimitive<i64>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
        i64: AsPrimitive<TC>,
    {
        pub fn run(&self) {
            if let FusedSpec::QScale(shift, policy, mult) = self.scaler.as_fused_spec() {
                fused_ops::<K, TC, TI, _>(
                    &*self.c,
                    &[FusedKerSpec::QScale(shift, policy, mult)],
                    |_, _, c| c.q_scale(self.scaler),
                )
            } else if let FusedSpec::RoundingShiftRight(shift, policy) = self.scaler.as_fused_spec()
            {
                fused_ops::<K, TC, TI, _>(
                    &*self.c,
                    &[FusedKerSpec::RoundingShiftRight(shift, policy)],
                    |_, _, c| c.q_shr(shift, policy),
                )
            } else if let FusedSpec::ShiftLeft(shift) = self.scaler.as_fused_spec() {
                fused_ops::<K, TC, TI, _>(
                    &*self.c,
                    &[FusedKerSpec::ShiftLeft(shift)],
                    |_, _, c| c.q_shl(shift),
                )
            } else {
                unreachable!()
            }
        }
    }

    pub fn tile<K, TC, TI>() -> BoxedStrategy<Vec<TC>>
    where
        K: MatMatMulKer<TI>,
        TC: LADatum,
        TI: LADatum + AsPrimitive<TC>,
        i8: AsPrimitive<TC>,
    {
        let len = K::mr() * K::nr();
        proptest::collection::vec(any::<i8>().prop_map(|c| c.as_()), len..=len).boxed()
    }
}
