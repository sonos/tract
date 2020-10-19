use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Deref, Mul, Neg};
use tract_data::internal::*;

use num_traits::{AsPrimitive, Bounded, Zero};

use super::MatMatMul;
use super::*;

pub trait QMatMatMul<TA, TB, TC, TI>:
    Debug + fmt::Display + dyn_clone::DynClone + Send + Sync + DynHash
where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
{
    fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI>;
    fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI>;

    unsafe fn set_zero_point_a_scalar(&mut self, value: TA);
    unsafe fn set_zero_point_a_vector(&mut self, values: Vec<TA>);
    unsafe fn set_zero_point_b_scalar(&mut self, value: TB);
    unsafe fn set_zero_point_b_vector(&mut self, values: Vec<TB>);

    unsafe fn set_zero_point_c_scalar(&mut self, value: TC);
    unsafe fn set_scale_factor(&mut self, factor: f32);

    unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC, non_linear: &[FusedSpec<TI>]);
}

dyn_clone::clone_trait_object!(<TA, TB, TC, TI> QMatMatMul<TA, TB, TC, TI> where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
);

impl<TA, TB, TC, TI> Hash for Box<dyn QMatMatMul<TA, TB, TC, TI>>
where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dyn_hash(state)
    }
}

#[derive(Debug, Clone)]
pub enum QuantizedParam<T: SloppyHash> {
    Scalar(T),
    Vector(Vec<T>),
}

impl<T: SloppyHash> std::hash::Hash for QuantizedParam<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Scalar(t) => t.sloppy_hash(state),
            Self::Vector(t) => {
                t.len().hash(state);
                t.iter().for_each(|v| v.sloppy_hash(state))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + SloppyHash + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
{
    pub mmm: MatMatMulImpl<K, TA, TB, TC, TI>,
    pub zero_point_a: Option<QuantizedParam<TA>>,
    pub zero_point_b: Option<QuantizedParam<TB>>,

    pub zero_point_c: Option<TC>,
    pub scale_factor: Option<(TI, usize)>,
}

impl<K, TA, TB, TC, TI> QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + AsPrimitive<TI>,
    TB: Copy + Zero + SloppyHash + AsPrimitive<TI> + Debug + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
{
    fn sum_a_over_k(&self, mut a: *const TA) -> Vec<TI> {
        match &self.mmm.a_storage {
            MatrixStoreSpec::Packed { .. } => {
                let mr = K::mr();
                let mut result = vec![TI::zero(); self.m];
                unsafe {
                    for p in 0..(self.m / mr) {
                        for _k in 0..self.k {
                            for row in 0..mr {
                                result[p * mr + row] = result[p * mr + row] + (*a).as_();
                                a = a.offset(1);
                            }
                        }
                    }
                    if self.m % mr != 0 {
                        let p = self.m / mr;
                        for _k in 0..self.k {
                            for row in 0..mr {
                                if row < self.m % mr {
                                    result[p * mr + row] = result[p * mr + row] + (*a).as_();
                                }
                                a = a.offset(1);
                            }
                        }
                    }
                }
                result
            }
            a => panic!("Storage for A {:?} not supported for quantized ops", a),
        }
    }

    fn sum_b_over_k(&self, mut b: *const TB) -> Vec<TI> {
        let mut result = vec![TI::zero(); self.n];
        match &self.mmm.b_storage {
            MatrixStoreSpec::Packed { .. } => unsafe {
                let nr = K::nr();
                for p in 0..(self.n / nr) {
                    for _k in 0..self.k {
                        for col in 0..nr {
                            result[p * nr + col] = result[p * nr + col] + (*b).as_();
                            b = b.offset(1);
                        }
                    }
                }
                if self.n % nr != 0 {
                    let p = self.n / nr;
                    for _k in 0..self.k {
                        for col in 0..nr {
                            if col < self.n % nr {
                                result[p * nr + col] = result[p * nr + col] + (*b).as_();
                            }
                            b = b.offset(1);
                        }
                    }
                }
            },
            MatrixStoreSpec::OffsetsAndPtrs { row_byte_offsets, col_byte_offsets, .. } => unsafe {
                for n in 0..self.n {
                    for k in 0..self.k {
                        let offset = (row_byte_offsets[k] + col_byte_offsets[n])
                            / std::mem::size_of::<TB>() as isize;
                        result[n] = result[n] + (*b.offset(offset)).as_();
                    }
                }
            },
            b => panic!("Storage {:?} for B not supported for quantized ops", b),
        }
        result
    }
}

impl<K, TA, TB, TC, TI> From<MatMatMulImpl<K, TA, TB, TC, TI>> for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
{
    fn from(mmm: MatMatMulImpl<K, TA, TB, TC, TI>) -> QMatMatMulImpl<K, TA, TB, TC, TI> {
        QMatMatMulImpl {
            mmm,
            zero_point_a: None,
            zero_point_b: None,
            zero_point_c: None,
            scale_factor: None,
        }
    }
}

impl<K, TA, TB, TC, TI> Deref for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
{
    type Target = MatMatMulImpl<K, TA, TB, TC, TI>;
    fn deref(&self) -> &Self::Target {
        &self.mmm
    }
}

unsafe impl<K, TA, TB, TC, TI> Send for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
{
}

unsafe impl<K, TA, TB, TC, TI> Sync for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
{
}

impl<K, TA, TB, TC, TI> QMatMatMul<TA, TB, TC, TI> for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + Debug + SloppyHash + AsPrimitive<TI> + 'static,
    TB: Copy + Zero + Debug + SloppyHash + AsPrimitive<TI> + 'static,
    TC: Copy + Debug + Bounded + AsPrimitive<TI> + SloppyHash + 'static,
    TI: Copy + Add + Mul<Output = TI> + Zero + Neg<Output = TI> + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI> + 'static,
    usize: AsPrimitive<TI>,
    i32: AsPrimitive<TI>,
{
    fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI> {
        &self.mmm
    }

    fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI> {
        &mut self.mmm
    }

    unsafe fn set_zero_point_a_scalar(&mut self, value: TA) {
        self.zero_point_a = Some(QuantizedParam::Scalar(value))
    }

    unsafe fn set_zero_point_b_scalar(&mut self, value: TB) {
        self.zero_point_b = Some(QuantizedParam::Scalar(value))
    }

    unsafe fn set_zero_point_c_scalar(&mut self, value: TC) {
        self.zero_point_c = Some(value)
    }

    unsafe fn set_zero_point_a_vector(&mut self, mut values: Vec<TA>) {
        let wanted = self.m() + K::mr() - 1 / K::mr() * K::mr();
        while values.len() < wanted {
            values.push(values[values.len() - 1])
        }
        self.zero_point_a = Some(QuantizedParam::Vector(values))
    }

    unsafe fn set_zero_point_b_vector(&mut self, mut values: Vec<TB>) {
        let wanted = self.n() + K::nr() - 1 / K::nr() * K::nr();
        while values.len() < wanted {
            values.push(values[values.len() - 1])
        }
        self.zero_point_b = Some(QuantizedParam::Vector(values))
    }

    unsafe fn set_scale_factor(&mut self, factor: f32) {
        // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/util/gemmlowp_common.h#L16
        let factor_bits = factor.to_bits();
        let current_exponent = factor_bits >> 23;
        let bumped_multi = f32::from_bits(factor_bits & 0x007fffff | 0x3f000000);
        let int_multi = (bumped_multi * (1i64 << 31) as f32).round() as i32;
        let shift = 126 - current_exponent;
        self.scale_factor = Some((int_multi.as_(), shift as usize));
    }

    unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC, non_linear: &[FusedSpec<TI>]) {
        /* SUM_k( A[m,k] * B[k,n] )
            = SUM_k( A'[m,k] * B'[k,n] )
            - A0[m] * SUM_k(B'[k,n])
            + (A0[m].K - SUM_k(A'[m,k])) * B0[n]
        */
        let mut non_linear = non_linear.to_vec();
        if let Some(ref a0) = self.zero_point_a {
            let mut sum_b_over_k = self.sum_b_over_k(b);
            for n in 0..self.n {
                sum_b_over_k[n] = sum_b_over_k[n].neg();
            }
            let term = match a0 {
                QuantizedParam::Scalar(a0) => {
                    for n in 0..self.n {
                        sum_b_over_k[n] = sum_b_over_k[n] * a0.as_();
                    }
                    FusedSpec::PerColAdd(sum_b_over_k)
                }
                QuantizedParam::Vector(a0) => {
                    let a0 = a0.iter().map(|a| a.as_()).collect();
                    FusedSpec::AddRowColProducts(a0, sum_b_over_k)
                }
            };
            non_linear.insert(0, term);
        }
        if let Some(ref b0) = self.zero_point_b {
            let mut sum_a_over_k = self.sum_a_over_k(a);
            for m in 0..self.m {
                sum_a_over_k[m] = sum_a_over_k[m].neg();
                if let Some(ref a0) = self.zero_point_a {
                    match a0 {
                        QuantizedParam::Scalar(a0) => {
                            sum_a_over_k[m] = a0.as_() * self.k.as_() + sum_a_over_k[m];
                        }
                        QuantizedParam::Vector(a0) => {
                            sum_a_over_k[m] = a0[m].as_() * self.k.as_() + sum_a_over_k[m];
                        }
                    }
                }
            }
            let term = match b0 {
                QuantizedParam::Scalar(b0) => {
                    for m in 0..self.m {
                        sum_a_over_k[m] = sum_a_over_k[m] * b0.as_();
                    }
                    FusedSpec::PerRowAdd(sum_a_over_k)
                }
                QuantizedParam::Vector(b0) => {
                    let b0 = b0.iter().map(|b| b.as_()).collect();
                    FusedSpec::AddRowColProducts(sum_a_over_k, b0)
                }
            };
            non_linear.insert(0, term);
        }
        if let Some(scale) = self.scale_factor {
            non_linear.push(FusedSpec::QTowardsPlusInf(scale.0, scale.1));
        }
        if let Some(c0) = self.zero_point_c {
            non_linear.push(FusedSpec::ScalarAdd(c0.as_()));
        }
        non_linear.push(FusedSpec::Min(TC::max_value().as_()));
        non_linear.push(FusedSpec::Max(TC::min_value().as_()));
        self.mmm.run(a, b, c, &non_linear);
    }
}

impl<K, TA, TB, TC, TI> fmt::Display for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.mmm)
    }
}

impl<TA, TB, TC, TI, K> std::hash::Hash for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn hash<S: std::hash::Hasher>(&self, state: &mut S) {
        /*
        pub mmm: MatMatMulImpl<K, TA, TB, TC, TI>,
        pub zero_point_a: Option<QuantizedParam<TA>>,
        pub zero_point_b: Option<QuantizedParam<TB>>,

        pub zero_point_c: Option<TC>,
        pub scale_factor: Option<(TI, usize)>,
        */
        self.mmm.hash(state);
        if let Some(a) = &self.zero_point_a {
            a.hash(state);
        }
        if let Some(b) = &self.zero_point_b {
            b.hash(state);
        }
    }
}

impl<TA, TB, TC, TI, K> DynHash for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + SloppyHash + 'static,
    TB: Copy + Zero + SloppyHash + 'static,
    TC: Copy + Debug + SloppyHash + 'static,
    TI: Copy + Add + Mul + Zero + Debug + SloppyHash + 'static,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(self, hasher)
    }
}

#[cfg(test)]
#[allow(dead_code)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::align::Buffer;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use std::marker::PhantomData;
    use std::ops::{AddAssign, Sub};

    #[derive(Debug)]
    pub struct QMatMulProblem<TA: SloppyHash, TB: SloppyHash, TC, TI: SloppyHash> {
        pub m: usize,
        pub k: usize,
        pub n: usize,
        pub a: Vec<TA>,
        pub a0: QuantizedParam<TA>,
        pub b: Vec<TB>,
        pub b0: QuantizedParam<TB>,
        pub boo: PhantomData<(TC, TI)>,
    }

    impl<TI: Arbitrary + 'static + SloppyHash> Arbitrary for QuantizedParam<TI> {
        type Parameters = usize;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(n: usize) -> Self::Strategy {
            prop_oneof![
                any::<TI>().prop_map(QuantizedParam::Scalar),
                vec(any::<TI>(), n..=n).prop_map(QuantizedParam::Vector),
            ]
            .boxed()
        }
    }

    impl<TA, TB, TC, TI> Arbitrary for QMatMulProblem<TA, TB, TC, TI>
    where
        TA: Arbitrary + 'static + Debug + 'static + SloppyHash,
        TB: Arbitrary + 'static + Debug + 'static + SloppyHash,
        TC: Arbitrary + 'static + Debug + 'static,
        TI: Arbitrary + 'static + Debug + 'static + SloppyHash,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (1usize..10, 1usize..10, 1usize..10)
                .prop_flat_map(|(m, k, n)| {
                    (
                        Just(m),
                        Just(k),
                        Just(n),
                        vec(any::<TA>(), m * k..=m * k),
                        any_with::<QuantizedParam<TA>>(m),
                        vec(any::<TB>(), k * n..=k * n),
                        any_with::<QuantizedParam<TB>>(n),
                    )
                })
                .prop_map(|(m, k, n, a, a0, b, b0)| QMatMulProblem {
                    m,
                    k,
                    n,
                    a,
                    a0,
                    b,
                    b0,
                    boo: PhantomData,
                })
                .boxed()
        }
    }

    impl<TA, TB, TC, TI> QMatMulProblem<TA, TB, TC, TI>
    where
        TA: Arbitrary + SloppyHash + 'static + Debug + AsPrimitive<TI> + Zero + Copy,
        TB: Arbitrary + SloppyHash + 'static + Debug + AsPrimitive<TI> + Zero + Copy,
        TC: Arbitrary
            + SloppyHash
            + 'static
            + Debug
            + Copy
            + Bounded
            + AsPrimitive<TI>
            + Zero
            + 'static,
        TI: Arbitrary
            + 'static
            + Debug
            + Copy
            + AsPrimitive<TC>
            + Add<Output = TI>
            + Mul<Output = TI>
            + Sub<Output = TI>
            + AddAssign
            + Neg<Output = TI>
            + Zero
            + SloppyHash
            + Ord,
        usize: AsPrimitive<TI>,
        i32: AsPrimitive<TI>,
    {
        pub fn reference(&self) -> Vec<TC> {
            let mut i = vec![TI::zero(); self.m * self.n];
            for m in 0..self.m {
                for n in 0..self.n {
                    for k in 0..self.k {
                        let a: TI = self.a[k + self.k * m].as_();
                        let b: TI = self.b[n + self.n * k].as_();
                        let a0 = match &self.a0 {
                            QuantizedParam::Scalar(a0) => a0.as_(),
                            QuantizedParam::Vector(a0) => a0[m].as_(),
                        };
                        let b0 = match &self.b0 {
                            QuantizedParam::Scalar(b0) => b0.as_(),
                            QuantizedParam::Vector(b0) => b0[n].as_(),
                        };
                        i[n + self.n * m] += (a - a0) * (b - b0);
                    }
                }
            }
            i.iter()
                .map(|i| i.max(&TC::min_value().as_()).min(&TC::max_value().as_()).as_())
                .collect()
        }

        pub fn run<K: MatMatMulKer<TA, TB, TC, TI>>(&self) -> Vec<TC> {
            unsafe {
                let mut c = vec![TC::zero(); self.m * self.n];
                let mut mmm = QMatMatMulImpl::from(MatMatMulImpl::<K, TA, TB, TC, TI>::new(
                    self.m, self.k, self.n,
                ));
                let mut packed_a =
                    Buffer::uninitialized(mmm.a_pack().len(), mmm.a_pack().alignment());
                mmm.a_pack().pack(packed_a.as_mut_ptr(), self.a.as_ptr(), self.k as isize, 1);
                let mut packed_b =
                    Buffer::uninitialized(mmm.b_pack().len(), mmm.b_pack().alignment());
                mmm.b_pack().pack(packed_b.as_mut_ptr(), self.b.as_ptr(), self.n as isize, 1);
                match &self.a0 {
                    QuantizedParam::Scalar(a0) => mmm.set_zero_point_a_scalar(*a0),
                    QuantizedParam::Vector(a0) => mmm.set_zero_point_a_vector(a0.clone()),
                }
                match &self.b0 {
                    QuantizedParam::Scalar(b0) => mmm.set_zero_point_b_scalar(*b0),
                    QuantizedParam::Vector(b0) => mmm.set_zero_point_b_vector(b0.clone()),
                }
                mmm.run(packed_a.as_ptr(), packed_b.as_ptr(), c.as_mut_ptr(), &[]);
                c
            }
        }
    }

    #[macro_export]
    macro_rules! qmmm_frame_tests {
        ($cond:expr, $ker:ty, $ta: ty, $tb: ty, $tc: ty, $ti: ty) => {
            mod qframe {
                use proptest::prelude::*;
                use std::marker::PhantomData;
                #[allow(unused_imports)]
                use $crate::frame::mmm::qmmm::test::*;
                use $crate::frame::mmm::qmmm::QuantizedParam;

                proptest::proptest! {
                    #[test]
                    fn q_mat_mul_prop(pb in any::<QMatMulProblem<$ta, $tb, $tc, $ti>>()) {
                        if $cond {
                            prop_assert_eq!(pb.run::<$ker>(), pb.reference())
                        }
                    }
                }

                #[test]
                fn q_mat_mul_1() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 1,
                            n: 1,
                            a0: QuantizedParam::Vector(vec![1]),
                            a: vec![0],
                            b0: QuantizedParam::Vector(vec![1]),
                            b: vec![0],
                            boo: PhantomData,
                        };
                        assert_eq!(pb.run::<$ker>(), pb.reference());
                    }
                }

                #[test]
                fn q_mat_mul_sat_1() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 1,
                            n: 1,
                            a0: QuantizedParam::Vector(vec![0]),
                            a: vec![3],
                            b0: QuantizedParam::Vector(vec![43]),
                            b: vec![0],
                            boo: PhantomData,
                        };
                        assert_eq!(pb.run::<$ker>(), pb.reference());
                    }
                }
                #[test]

                fn q_mat_mul_sat_2() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 1,
                            n: 1,
                            a0: QuantizedParam::Vector(vec![0]),
                            a: vec![<$ta>::min_value()],
                            b0: QuantizedParam::Vector(vec![0]),
                            b: vec![1],
                            boo: PhantomData,
                        };
                        assert_eq!(pb.run::<$ker>(), pb.reference());
                    }
                }

                #[test]
                fn q_mat_mul_n2() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 1,
                            n: 2,
                            a: vec![0],
                            a0: QuantizedParam::Vector(vec![1]),
                            b: vec![0, 0],
                            b0: QuantizedParam::Vector(vec![0, 1]),
                            boo: PhantomData,
                        };
                        assert_eq!(pb.run::<$ker>(), pb.reference());
                    }
                }

                #[test]
                fn q_mat_mul_k2() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 2,
                            n: 1,
                            a: vec![0, 1],
                            a0: QuantizedParam::Vector(vec![0]),
                            b: vec![0, 1],
                            b0: QuantizedParam::Vector(vec![0]),
                            boo: PhantomData,
                        };
                        assert_eq!(pb.run::<$ker>(), pb.reference());
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! qmmm_s_frame_tests {
        ($cond:expr, $ker:ty, $ta: ty, $tb: ty, $tc: ty, $ti: ty) => {
            mod qframe_s {
                use std::marker::PhantomData;
                #[allow(unused_imports)]
                use $crate::frame::mmm::qmmm::test::*;
                use $crate::frame::mmm::qmmm::QuantizedParam;

                #[test]
                fn q_mat_mul_1_1_5() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 1,
                            n: 5,
                            a: vec![-1],
                            a0: QuantizedParam::Scalar(0),
                            b: vec![0, 0, 0, 0, -2],
                            b0: QuantizedParam::Scalar(0),
                            boo: PhantomData,
                        };
                        assert_eq!(pb.run::<$ker>(), pb.reference());
                    }
                }

                #[test]
                fn q_mat_mul_1_1_1() {
                    if $cond {
                        let pb = QMatMulProblem {
                            m: 1,
                            k: 1,
                            n: 1,
                            a: vec![11],
                            a0: QuantizedParam::Scalar(10),
                            b: vec![-1],
                            b0: QuantizedParam::Scalar(0),
                            boo: PhantomData,
                        };
                        assert_eq!(pb.run::<$ker>(), pb.reference());
                    }
                }
            }
        };
    }
}
