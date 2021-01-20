use std::fmt::Debug;

use num_traits::Zero;

use super::MatMatMulKer;
use tract_data::prelude::*;

#[derive(PartialEq, Clone, Hash, Debug)]
pub enum FusedSpec {
    Min(Tensor),
    Max(Tensor),
    AddC,
    PerRowMul(Tensor),
    PerRowAdd(Tensor),
    PerColMul(Tensor),
    PerColAdd(Tensor),
    AddRowColProducts(Tensor, Tensor),
    ScalarMul(Tensor),
    ScalarAdd(Tensor),
    QTowardsEven(Tensor, usize),
    QTowardsPlusInf(Tensor, usize),
}

/*
   impl Debug for FusedSpec {
   fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
   match self {
   FusedSpec::Min(t) => write!(fmt, "Min({:?})", t),
   FusedSpec::Max(t) => write!(fmt, "Max({:?})", t),
   FusedSpec::AddC => write!(fmt, "AddC"),
   FusedSpec::PerRowMul(_) => write!(fmt, "PerRowMul"),
   FusedSpec::PerRowAdd(_) => write!(fmt, "PerRowAdd"),
   FusedSpec::PerColMul(_) => write!(fmt, "PerColMul"),
   FusedSpec::PerColAdd(_) => write!(fmt, "PerColAdd"),
   FusedSpec::AddRowColProducts(_, _) => write!(fmt, "AddRowColProducts"),
   FusedSpec::ScalarMul(_) => write!(fmt, "ScalarMul"),
   FusedSpec::ScalarAdd(_) => write!(fmt, "ScalarAdd"),
   FusedSpec::QTowardsEven(_, _) => write!(fmt, "QTowardsEven"),
   FusedSpec::QTowardsPlusInf(_, _) => write!(fmt, "QTowardsPlusInf"),
   }
   }
   }

   impl std::hash::Hash for FusedSpec {
   fn hash<H>(&self, state: &mut H)
   where
   H: std::hash::Hasher,
   {
/*
use FusedSpec::*;
fn h<TI: Copy, H: std::hash::Hasher>(it: &[TI], state: &mut H) {
unsafe {
it.len().hash(state);
let bytes: &[u8] = std::slice::from_raw_parts(
it.as_ptr() as _,
it.len() * std::mem::size_of::<TI>(),
);
bytes.hash(state)
}
}
*/
std::mem::discriminant(self).hash(state);
match self {
    AddC => (),
    Min(a) | Max(a) | ScalarMul(a) | ScalarAdd(a) => h(&[*a], state),
    PerRowMul(a) | PerRowAdd(a) | PerColMul(a) | PerColAdd(a) => h(&*a, state),
    AddRowColProducts(a, b) => {
        h(&*a, state);
        h(&*b, state);
    }
    QTowardsEven(a, b) | QTowardsPlusInf(a, b) => {
        h(&[*a], state);
        h(&[*b], state);
    }
}
}
}
*/

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum FusedKerSpec<TI: Copy> {
    Done,
    Min(TI),
    Max(TI),
    AddC,
    PerRowMul(*const TI),
    PerRowAdd(*const TI),
    PerColMul(*const TI),
    PerColAdd(*const TI),
    AddRowColProducts(*const TI, *const TI),
    ScalarMul(TI),
    ScalarAdd(TI),
    QTowardsEven(TI, usize),
    QTowardsPlusInf(TI, usize),
}

pub struct ScratchSpaceFusedNonLinear<TI: Copy> {
    uspecs: Vec<FusedKerSpec<TI>>,
    non_linear_buffers: Vec<Vec<TI>>,
}

impl<TI: Copy> Default for ScratchSpaceFusedNonLinear<TI> {
    fn default() -> ScratchSpaceFusedNonLinear<TI> {
        ScratchSpaceFusedNonLinear { uspecs: vec![], non_linear_buffers: vec![] }
    }
}

impl<TI: Copy> ScratchSpaceFusedNonLinear<TI> {
    pub unsafe fn for_tile<TA, TB, TC, K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) -> *const FusedKerSpec<TI>
    where
        TA: Datum + Copy,
        TB: Datum + Copy,
        TC: Datum + Copy,
        TI: Datum + Copy + Debug + Zero,
    {
        self.uspecs.clear();
        for spec in specs {
            let s = match spec {
                FusedSpec::Min(m) => FusedKerSpec::Min(*m.to_scalar_unchecked()),
                FusedSpec::Max(m) => FusedKerSpec::Max(*m.to_scalar_unchecked()),
                FusedSpec::AddC => FusedKerSpec::AddC,
                FusedSpec::PerRowAdd(v)
                | FusedSpec::PerRowMul(v)
                | FusedSpec::PerColMul(v)
                | FusedSpec::PerColAdd(v) => {
                    let (dir, r) = if matches!(spec, FusedSpec::PerColAdd(_) | FusedSpec::PerColMul(_)) {
                        (right, K::nr())
                    } else {
                        (down, K::mr())
                    };
                    dbg!(v);
                    dbg!(dir);
                    dbg!(r);
                    let have = v.len().saturating_sub(dir * r);
                    let ptr = if have < K::mr() {
                        let mut buf = vec![TI::zero(); r];
                        if have > 0 {
                            buf[..have]
                                .copy_from_slice(&v.as_slice_unchecked()[dir * r..][..have]);
                        }
                        dbg!(&buf);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr_unchecked::<TI>().add(dir * r)
                    };
                    match spec {
                        FusedSpec::PerRowAdd(_) => FusedKerSpec::PerRowAdd(ptr),
                        FusedSpec::PerRowMul(_) => FusedKerSpec::PerRowMul(ptr),
                        FusedSpec::PerColAdd(_) => FusedKerSpec::PerColAdd(ptr),
                        FusedSpec::PerColMul(_) => FusedKerSpec::PerColMul(ptr),
                        _ => unreachable!(),
                    }
                }
                FusedSpec::AddRowColProducts(rows, cols) => {
                    let have = rows.len() - down * K::mr();
                    let row_ptr = if have < K::mr() {
                        let mut buf = vec![TI::zero(); K::mr()];
                        buf[..have]
                            .copy_from_slice(&rows.as_slice_unchecked()[down * K::mr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        rows.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    let have = cols.len() - right * K::nr();
                    let col_ptr = if have < K::nr() {
                        let mut buf = vec![TI::zero(); K::nr()];
                        buf[..have]
                            .copy_from_slice(&cols.as_slice_unchecked()[right * K::nr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        cols.as_ptr_unchecked::<TI>().add(right * K::nr())
                    };
                    FusedKerSpec::AddRowColProducts(row_ptr, col_ptr)
                }
                FusedSpec::ScalarMul(t) => FusedKerSpec::ScalarMul(*t.to_scalar_unchecked()),
                FusedSpec::ScalarAdd(t) => FusedKerSpec::ScalarAdd(*t.to_scalar_unchecked()),
                FusedSpec::QTowardsEven(m, s) => {
                    FusedKerSpec::QTowardsEven(*m.to_scalar_unchecked(), *s)
                }
                FusedSpec::QTowardsPlusInf(m, s) => {
                    FusedKerSpec::QTowardsPlusInf(*m.to_scalar_unchecked(), *s)
                }
            };
            self.uspecs.push(s);
        }
        dbg!(&self.uspecs);
        self.uspecs.push(FusedKerSpec::Done);
        self.uspecs.as_ptr()
    }
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::frame::mmm::storage::*;
    use crate::frame::mmm::*;
    use num_traits::{AsPrimitive, Bounded, Zero};
    use proptest::prelude::*;
    use std::fmt;
    use std::ops::{Add, Mul, Sub};

    #[test]
    fn check_non_linear_enum_size() {
        assert_eq!(
            std::mem::size_of::<super::FusedKerSpec<f32>>(),
            3 * std::mem::size_of::<usize>()
        )
    }

    #[macro_export]
    macro_rules! mmm_kernel_fuse_tests {
        ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod fuse {
                #[allow(unused_imports)]
                use crate::frame::mmm::fuse::test;
                use proptest::prelude::*;

                #[test]
                fn return_zeros() {
                    if $cond {
                        test::return_zeros::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c() {
                    if $cond {
                        test::return_c::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                proptest::proptest! {
                    #[test]
                    fn return_c_prop(pb in any::<test::ReturnCProblem<$ker, $ta, $tb, $tc, $ti>>()) {
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
                        test::return_c_mul_row::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_add_row() {
                    if $cond {
                        test::return_c_add_row::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_mul_col() {
                    if $cond {
                        test::return_c_mul_col::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_add_col() {
                    if $cond {
                        test::return_c_add_col::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_add_row_col_product() {
                    if $cond {
                        test::return_c_add_row_col_product::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_max() {
                    if $cond {
                        test::return_c_max::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_min() {
                    if $cond {
                        test::return_c_min::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_mul() {
                    if $cond {
                        test::return_c_scalar_mul::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }

                #[test]
                fn return_c_scalar_add() {
                    if $cond {
                        test::return_c_scalar_add::<$ker, $ta, $tb, $tc, $ti>()
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! qmmm_kernel_fuse_tests {
        ($cond:expr, $ker:ty, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
            mod fuseq {
                #[allow(unused_imports)]
                use crate::frame::mmm::fuse::test;
                use crate::frame::mmm::fuse::test::QTowardsPlusInfProblem;
                use crate::frame::mmm::kernel::MatMatMulKer;
                use num_traits::AsPrimitive;
                use proptest::prelude::*;

                #[test]
                #[ignore]
                fn return_q_towards_even() {
                    if $cond {
                        test::return_q_towards_even::<$ker, $ta, $tb, $tc, $ti>()
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
                        let pb = QTowardsPlusInfProblem::<$ker, $ta, $tb, $tc, $ti>::new(v);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                #[test]
                fn return_q_towards_plusinf_1() {
                    if $cond {
                        let len = <$ker>::mr() * <$ker>::nr();
                        let mut v = vec!(0; len - 1);
                        v.push(2);
                        let pb = QTowardsPlusInfProblem::<$ker, $ta, $tb, $tc, $ti>::new(v);
                        assert_eq!(pb.run(), pb.reference())
                    }
                }

                proptest::proptest! {
                    #[test]
                    fn return_q_towards_plusinf_prop(pb in any::<QTowardsPlusInfProblem<$ker, $ta, $tb, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run(), pb.reference())
                        }
                    }
                }
            }
        };
    }

    pub fn null_packed_storage<T: Copy>() -> PanelStore {
        PanelStore::Packed { ptr: std::ptr::null::<T>() as _ }
    }

    pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize) -> PanelStore {
        PanelStore::Strides {
            ptr: v.as_mut_ptr() as _,
            row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
            col_byte_stride: std::mem::size_of::<T>() as isize,
            item_size: std::mem::size_of::<T>(),
        }
    }

    pub fn return_zeros<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
        TC: Copy + Debug + Bounded + Zero + PartialEq,
        TI: Copy + Debug,
    {
        let mut v = vec![TC::max_value(); K::mr() * K::nr()];
        let mut c = mmm_stride_storage(&mut v, K::nr());
        let err = K::kernel(&MatMatMulKerSpec {
            a: &null_packed_storage::<TA>(),
            b: &null_packed_storage::<TB>(),
            c: &mut c,
            linear: &LinearSpec::k(0),
            non_linear: std::ptr::null(),
        });
        assert_eq!(err, 0);
        let expected = vec![TC::zero(); v.len()];
        assert_eq!(v, expected);
    }

    pub fn fused_ops<K, TA, TB, TC, TI>(c: &[TC], ops: &[FusedKerSpec<TI>]) -> Vec<TC>
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
        TC: Copy + 'static + PartialEq,
        TI: Copy + Debug,
        usize: AsPrimitive<TC>,
    {
        assert!(c.len() == K::mr() * K::nr());
        let mut v = c.to_vec();
        let mut c = mmm_stride_storage(&mut v, K::nr());
        let mut ops = ops.to_vec();
        ops.insert(0, FusedKerSpec::AddC);
        ops.push(FusedKerSpec::Done);
        let err = K::kernel(&MatMatMulKerSpec {
            a: &null_packed_storage::<TA>(),
            b: &null_packed_storage::<TB>(),
            c: &mut c,
            linear: &LinearSpec::k(0),
            non_linear: ops.as_ptr(),
        });
        assert_eq!(err, 0);
        v
    }

    pub fn return_c<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
        TC: Copy + Debug + 'static + PartialEq,
        TI: Copy + Debug,
        usize: AsPrimitive<TC>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[]);
        assert_eq!(found, v);
    }

    pub fn return_c_mul_row<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
        TC: Copy + Debug + 'static + PartialEq,
        TI: Copy + Add + Mul<Output = TI> + Zero + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::PerRowMul(bias.as_ptr())]);
        let expected = (0..found.len())
            .map(|ix| {
                let row = ix / K::nr();
                let ix: TI = ix.as_();
                (ix * bias[row]).as_()
            })
            .collect::<Vec<TC>>();
        assert_eq!(found, expected);
    }

    pub fn return_c_add_row<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
        TC: Copy + PartialEq + 'static,
        TI: Copy + Add + Mul + Zero + Debug + fmt::Display + PartialEq + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::mr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::PerRowAdd(bias.as_ptr())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let row = ix / K::nr();
            let ix: TI = ix.as_();
            a == (ix + bias[row]).as_()
        }));
    }

    pub fn return_c_mul_col<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
        TC: Copy + 'static + PartialEq,
        TI: Copy + Add + Mul<Output = TI> + Zero + Debug + fmt::Display + 'static + AsPrimitive<TC>,
        usize: AsPrimitive<TC> + AsPrimitive<TI>,
    {
        let len = K::mr() * K::nr();
        let v: Vec<TC> = (0..len).map(|f| f.as_()).collect();
        let bias: Vec<TI> = (0..K::nr()).map(|f| f.as_()).collect();
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::PerColMul(bias.as_ptr())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let col = ix % K::nr();
            let ix: TI = ix.as_();
            a == (ix * bias[col]).as_()
        }));
    }

    pub fn return_c_add_col<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
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
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::PerColAdd(bias.as_ptr())]);
        let expected = (0..found.len())
            .map(|ix| {
                let col = ix % K::nr();
                let ix: TI = ix.as_();
                (ix + bias[col]).as_()
            })
            .collect::<Vec<TC>>();
        assert_eq!(found, expected);
    }

    pub fn return_c_add_row_col_product<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
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
        let found = fused_ops::<K, TA, TB, TC, TI>(
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

    pub fn return_c_scalar_mul<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
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
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::ScalarMul(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == (ix * 5.as_()).as_()
        }));
    }

    pub fn return_c_max<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
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
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::Max(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == if ix > 5.as_() { ix.as_() } else { 5.as_() }
        }));
    }

    pub fn return_c_min<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
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
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::Min(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == if ix < 5.as_() { ix.as_() } else { 5.as_() }
        }));
    }

    pub fn return_c_scalar_add<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
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
        let found = fused_ops::<K, TA, TB, TC, TI>(&*v, &[FusedKerSpec::ScalarAdd(5.as_())]);
        assert!(found.iter().enumerate().all(|(ix, &a)| {
            let ix: TI = ix.as_();
            a == (ix + 5.as_()).as_()
        }));
    }

    pub fn return_q_towards_even<K, TA, TB, TC, TI>()
    where
        K: MatMatMulKer<TI>,
        TA: Copy,
        TB: Copy,
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
        let found = fused_ops::<K, TA, TB, TC, TI>(
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
    pub struct QTowardsPlusInfProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: Copy + Debug,
        TB: Copy + Debug,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
        i64: AsPrimitive<TC>,
    {
        pub c: Vec<TC>,
        pub boo: std::marker::PhantomData<(K, TA, TB, TC, TI)>,
    }

    impl<K, TA, TB, TC, TI> Arbitrary for QTowardsPlusInfProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: Copy + Debug,
        TB: Copy + Debug,
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

    impl<K, TA, TB, TC, TI> QTowardsPlusInfProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: Copy + Debug,
        TB: Copy + Debug,
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
            fused_ops::<K, TA, TB, TC, TI>(
                &*self.c,
                &[
                    FusedKerSpec::ScalarMul(2.as_()),
                    FusedKerSpec::QTowardsPlusInf((1 << 30).as_(), 2),
                ],
            )
        }
    }

    #[derive(Debug, new)]
    pub struct ReturnCProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: Copy + Debug,
        TB: Copy + Debug,
        TC: Copy + Debug + 'static,
        TI: Copy + Debug,
    {
        pub c: Vec<TC>,
        pub boo: std::marker::PhantomData<(K, TA, TB, TC, TI)>,
    }

    impl<K, TA, TB, TC, TI> Arbitrary for ReturnCProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: Copy + Debug,
        TB: Copy + Debug,
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

    impl<K, TA, TB, TC, TI> ReturnCProblem<K, TA, TB, TC, TI>
    where
        K: MatMatMulKer<TI>,
        TA: Copy + Debug,
        TB: Copy + Debug,
        TC: Copy + Debug + 'static + AsPrimitive<TI> + PartialEq,
        TI: Copy + Debug + 'static + AsPrimitive<i64>,
        usize: AsPrimitive<TC>,
    {
        pub fn run(&self) -> Vec<TC> {
            fused_ops::<K, TA, TB, TC, TI>(&*self.c, &[])
        }
    }
}
