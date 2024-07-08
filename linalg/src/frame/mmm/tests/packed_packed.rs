use crate::frame::block_quant::PackedBlockQuantFormat;
use crate::frame::mmm::*;
use crate::LADatum;
use num_traits::{AsPrimitive, Zero};
use proptest::collection::vec;
use proptest::prelude::*;
use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use tests::display_error;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_packed_packed_tests {
    ($cond:expr, $ker:ident, $packing_id:ident : $packing: expr, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
        mod $packing_id {
            use tract_data::prelude::*;
            #[allow(unused_imports)]
            use super::$ker;
            use num_traits::{Zero, One};
            use proptest::prelude::*;
            #[allow(unused_imports)]
            use tract_data::prelude::f16;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::packed_packed::*;
            use $crate::frame::mmm::MatMatMulKer;

            proptest::proptest! {
                #[test]
                fn packed_packed_prop(pb in any_with::<PackedPackedProblem<_, $ta, $tb, $tc, $ti>>(($ker, $packing))) {
                    if $cond {
                        pb.check().unwrap()
                    }
                }
            }

            #[test]
            fn packed_packed_1()  -> TractResult<()> {
                if $cond {
                    let a = vec![<$ta>::one(); $ker.mr()];
                    let b = vec![<$tb>::one(); $ker.nr()];
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_2()  -> TractResult<()> {
                if $cond {
                    let a = vec![<$ta>::one(); $ker.mr() * 2];
                    let b = vec![<$tb>::one(); $ker.nr() * 2];
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_13()  -> TractResult<()> {
                if $cond {
                    let a = vec![<$ta>::one(); $ker.mr() * 13];
                    let b = vec![<$tb>::one(); $ker.nr() * 13];
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_a_scale()  -> TractResult<()> {
                if $cond {
                    let a = tensor1(&(1..=$ker.mr() as i64).collect::<Vec<_>>()).cast_to::<$ta>()?.as_slice::<$ta>()?.to_vec();
                    let b = vec![<$tb>::one(); $ker.nr()];
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }
            
            #[test]
            fn packed_packed_a_scale_times_2()  -> TractResult<()> {
                if $cond {
                    let a = tensor1(&(1..=2*$ker.mr() as i64).collect::<Vec<_>>()).cast_to::<$ta>()?.as_slice::<$ta>()?.to_vec();
                    let b = vec![<$tb>::one(); $ker.nr() * 2];
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_empty() -> TractResult<()> {
                if $cond {
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                        $ker,
                        $packing,
                        vec![<$ta>::zero(); 0],
                        vec![<$tb>::zero(); 0],
                    ).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_bug_1() -> TractResult<()> {
                if $cond {
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                        $ker,
                        $packing,
                        vec![<$ta>::zero(); $ker.mr()],
                        vec![<$tb>::zero(); $ker.nr()],
                    ).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_bug_2() -> TractResult<()> {
                if $cond {
                    let mut a = vec![<$ta>::zero(); $ker.mr()];
                    a[0] = <$ta>::one();
                    let mut b = vec![<$tb>::zero(); $ker.nr()];
                    b[0] = <$tb>::one();
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                        $ker,
                        $packing,
                        a, b
                    ).check()?;
                }
                Ok(())
            }

            macro_rules! set {
                ($lhs: expr, $rhs: expr) => {
                    if let Ok(x) = tensor0($rhs).cast_to_scalar() {
                        $lhs = x;
                    }

                }
            }

            #[test]
            fn packed_packed_bug_3() -> TractResult<()> {
                if $cond && $ker.mr() >= 4 {
                    let mut a = vec![<$ta>::zero(); 2 * $ker.mr()];
                    let mut b = vec![<$tb>::zero(); 2 * $ker.nr()];
                    set!(a[2], -0.7548828f32);
                    set!(a[3], 0.23547363f32);
                    set!(b[2*$ker.nr() - 1], 0.93603516);
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                        $ker,
                        $packing,
                        a, b
                    ).check()?;
                }
                Ok(())
            }

        }
    };
}

#[derive(Debug, new)]
pub struct PackedPackedProblem<K, TA, TB, TC, TI>
where
    K: MatMatMulKer<Acc = TI>,
    TA: LADatum + AsPrimitive<TI>,
    TB: LADatum + AsPrimitive<TI>,
    TC: LADatum + Copy + PartialEq + 'static + Debug,
    TI: LADatum + fmt::Display + AsPrimitive<TC>,
    usize: AsPrimitive<TA> + AsPrimitive<TB>,
{
    pub ker: K,
    pub packing: usize,
    pub a: Vec<TA>,
    pub b: Vec<TB>,
    pub _phantom: PhantomData<(K, TC, TI)>,
}

fn data<T: LADatum>() -> BoxedStrategy<T>
where
    f32: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    match T::datum_type() {
        DatumType::F64 => (-1f32..1f32).prop_map(|t| t.as_()).boxed(),
        DatumType::F32 => (-1f32..1f32).prop_map(|t| t.as_()).boxed(),
        DatumType::F16 => (-1f32..1f32).prop_map(|t| t.as_()).boxed(),
        DatumType::I8 => (-5i8..5).prop_map(|t| t.as_()).boxed(),
        DatumType::I32 => (-5i8..5).prop_map(|t| t.as_()).boxed(),
        _ => todo!(),
    }
}

impl<K, TA, TB, TC, TI> Arbitrary for PackedPackedProblem<K, TA, TB, TC, TI>
where
    K: MatMatMulKer<Acc = TI> + Default + Copy,
    TA: LADatum + AsPrimitive<TI>,
    TB: LADatum + AsPrimitive<TI>,
    TC: LADatum + Copy + PartialEq + 'static + Debug,
    TI: LADatum + fmt::Display + AsPrimitive<TC>,
    usize: AsPrimitive<TA> + AsPrimitive<TB>,
    f32: AsPrimitive<TA> + AsPrimitive<TB>,
    i8: AsPrimitive<TA> + AsPrimitive<TB>,
{
    type Parameters = (K, usize);
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((ker, packing): Self::Parameters) -> Self::Strategy {
        (0usize..100)
            .prop_flat_map(|k| {
                let ker = K::default();
                let m = k * ker.mr();
                let n = k * ker.nr();
                (vec(data::<TA>(), m..=m), vec(data::<TB>(), n..=n))
            })
            .prop_map(move |(a, b)| Self { ker, packing, a, b, _phantom: PhantomData })
            .boxed()
    }
}

impl<K, TA, TB, TC, TI> PackedPackedProblem<K, TA, TB, TC, TI>
where
    K: MatMatMulKer<Acc = TI>,
    TA: LADatum + AsPrimitive<TI>,
    TB: LADatum + AsPrimitive<TI>,
    TC: LADatum + Copy + Zero + PartialEq + 'static + Debug,
    TI: LADatum + fmt::Display + AsPrimitive<TC>,
    usize: AsPrimitive<TA> + AsPrimitive<TB>,
{
    pub fn padded_inputs(&self) -> TractResult<(Tensor, Tensor)> {
        let pack_a = self.ker.packings()[self.packing].0;
        let pack_b = self.ker.packings()[self.packing].1;
        assert!(pack_b.k_alignment() == 1);
        let k = self.a.len() / self.ker.mr();
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());

        let mut a = Tensor::zero::<TA>(&[pack_a.r(), k_aligned])?;
        for row in 0..pack_a.r() {
            for col in 0..k {
                a.to_array_view_mut()?[[row, col]] = self.a[col + k * row];
            }
        }
        let mut b = Tensor::zero::<TB>(&[k_aligned, pack_b.r()])?;
        for row in 0..k {
            for col in 0..pack_b.r() {
                b.to_array_view_mut()?[[row, col]] = self.b[col + pack_b.r() * row];
            }
        }

        Ok((a, b))
    }

    pub fn reference(&self) -> TractResult<Tensor> {
        let mr = self.ker.mr();
        let nr = self.ker.nr();
        ensure!(self.a.len() / mr == self.b.len() / nr);
        let k = self.a.len() / mr;
        let pack_a = self.ker.packings()[self.packing].0;
        let (mut a, _b) = self.padded_inputs()?;
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());
        if let Some(pbqf) = pack_a.downcast_ref::<PackedBlockQuantFormat>() {
            a = pbqf.simulate_precision_loss(a, 1)?;
        };
        let mut vi = Tensor::zero::<TI>(&[mr, nr])?;
        let mut view = vi.to_array_view_mut::<TI>()?.into_dimensionality()?;
        for m in 0..mr {
            for n in 0..nr {
                for ik in 0..k {
                    let a: TI = a.as_slice::<TA>()?[ik + k_aligned * m].as_();
                    let b: TI = self.b[n + nr * ik].as_();
                    view[(m, n)] += a * b;
                }
            }
        }
        Ok(vi.cast_to::<TC>()?.into_owned())
    }

    pub fn run(&self) -> TractResult<Tensor> {
        let pack_a = self.ker.packings()[self.packing].0;
        let pack_b = self.ker.packings()[self.packing].1;
        assert!(pack_b.k_alignment() == 1);
        let k = self.a.len() / self.ker.mr();
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());

        let (a, b) = self.padded_inputs()?;
        let pa = pack_a.prepare_tensor(&a, 1, 0)?;
        let pb = pack_b.prepare_tensor(&b, 0, 1)?;

        let mut v = vec![TC::zero(); self.ker.mr() * self.ker.nr()];

        let c = OutputStoreKer {
            ptr: v.as_mut_ptr() as _,
            row_byte_stride: (std::mem::size_of::<TC>() * self.ker.nr()) as isize,
            col_byte_stride: (std::mem::size_of::<TC>()) as isize,
            item_size: std::mem::size_of::<TC>(),
        };

        let non_linear_ops = tvec!(
            FusedKerSpec::Clear,
            FusedKerSpec::AddMatMul {
                k: k_aligned,
                pa: pa.panel_bytes(0, None)?,
                pb: pb.panel_bytes(0, None)?,
                packing: self.packing
            },
            FusedKerSpec::Store(c),
            FusedKerSpec::Done
        );
        let err = self.ker.kernel(&non_linear_ops);
        assert_eq!(err, 0);
        tensor1(&v).into_shape(&[self.ker.mr(), self.ker.nr()])
    }

    pub fn check(&self) -> TractResult<()> {
        let expected = self.reference()?;
        let found = self.run()?;
        let app = if TI::datum_type() == f16::datum_type() {
            Approximation::SuperApproximate
        } else {
            Approximation::Approximate
        };
        let result = found.close_enough(&expected, app);
        if result.is_err() {
            let exp = expected.as_slice::<TC>()?;
            let found = found.as_slice::<TC>()?;
            display_error(found, exp, self.ker.mr(), self.ker.nr());
        }
        result
    }
}
