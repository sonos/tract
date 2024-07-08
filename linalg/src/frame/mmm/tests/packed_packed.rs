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
            use tract_data::TractResult;
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
                    packed_packed::<_, $ta, $tb, $tc, $ti>($ker, $packing, 1)?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_2()  -> TractResult<()> {
                if $cond {
                    packed_packed::<_, $ta, $tb, $tc, $ti>($ker, $packing, 2)?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_13()  -> TractResult<()> {
                if $cond {
                    packed_packed::<_, $ta, $tb, $tc, $ti>($ker, $packing, 13)?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_empty() -> TractResult<()> {
                if $cond {
                    PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                        $ker,
                        $packing,
                        0,
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
                        1,
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
                        1, a, b
                    ).check()?;
                }
                Ok(())
            }

            /*
               #[test]
               fn packed_packed_bug_3() -> TractResult<()> {
               if $cond && $ker.mr() >= 4 {
               let mut a = vec![<$ta>::zero(); $ker.mr()];
               a[1] = 0.26635742f32.as_();
               a[2] = -0.4741211;
               let mut b = vec![<$tb>::zero(); $ker.nr()];
               b[0] = <$tb>::one();
               PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
               $ker,
               $packing,
               1, a, b
               ).check()?;
               }
               Ok(())
               }
               */
        }
    };
}

#[test]
fn generic_f16_q40f16() {
    PackedPackedProblem {
        ker: generic_f16_q40f16(),
        packing: 1,
        k: 2,
        a: vec![[0.0, 0.0, 0.26635742, -0.4741211, 0.0, 0.0, 0.0, 0.0]],
        b: vec![-0.25195313, 0.0],
        _phantom: PhantomData,
    }.check()
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
    pub k: usize,
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
                (Just(k), vec(data::<TA>(), m..=m), vec(data::<TB>(), n..=n))
            })
            .prop_map(move |(k, a, b)| Self { ker, packing, k, a, b, _phantom: PhantomData })
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
        let k_aligned = self.k.next_multiple_of(pack_a.k_alignment());

        let mut a = Tensor::zero::<TA>(&[pack_a.r(), k_aligned])?;
        for row in 0..pack_a.r() {
            for col in 0..self.k {
                a.to_array_view_mut()?[[row, col]] = self.a[col + self.k * row];
            }
        }
        let mut b = Tensor::zero::<TB>(&[k_aligned, pack_b.r()])?;
        for row in 0..self.k {
            for col in 0..pack_b.r() {
                b.to_array_view_mut()?[[row, col]] = self.b[col + pack_b.r() * row];
            }
        }

        Ok((a, b))
    }

    pub fn reference(&self) -> TractResult<Tensor> {
        let mr = self.ker.mr();
        let nr = self.ker.nr();
        let pack_a = self.ker.packings()[self.packing].0;
        let (mut a, _b) = self.padded_inputs()?;
        let k_aligned = self.k.next_multiple_of(pack_a.k_alignment());
        if let Some(pbqf) = pack_a.downcast_ref::<PackedBlockQuantFormat>() {
            a = pbqf.simulate_precision_loss(a, 1)?;
        };
        let mut vi = Tensor::zero::<TI>(&[mr, nr])?;
        let mut view = vi.to_array_view_mut::<TI>()?.into_dimensionality()?;
        for m in 0..mr {
            for n in 0..nr {
                for k in 0..self.k {
                    let a: TI = a.as_slice::<TA>()?[k + k_aligned * m].as_();
                    let b: TI = self.b[n + nr * k].as_();
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
        let k_aligned = self.k.next_multiple_of(pack_a.k_alignment());

        let (a, b) = self.padded_inputs()?;
        dbg!(a.to_array_view::<TA>()?);
        let pa = pack_a.prepare_tensor(&a, 1, 0)?;
        let pb = pack_b.prepare_tensor(&b, 0, 1)?;

        let mut v = vec![TC::zero(); self.ker.mr() * self.ker.nr()];
        let c = mmm_stride_storage(&mut v, self.ker.nr(), 1);

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

pub fn packed_packed<K, TA, TB, TC, TI>(ker: K, packing: usize, k: usize) -> TractResult<()>
where
    K: MatMatMulKer<Acc = TI>,
    TA: LADatum + AsPrimitive<TI>,
    TB: LADatum + AsPrimitive<TI>,
    TC: LADatum + Copy + PartialEq + Zero + 'static + Debug,
    TI: LADatum + AsPrimitive<TC>,
    usize: AsPrimitive<TC> + AsPrimitive<TA> + AsPrimitive<TB>,
{
    let a = vec![TA::one(); ker.mr() * k];
    let b = vec![TB::one(); ker.nr() * k];
    PackedPackedProblem::<K, TA, TB, TC, TI>::new(ker, packing, k, a, b).check()
}

pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize, csc: usize) -> OutputStoreKer {
    OutputStoreKer {
        ptr: v.as_mut_ptr() as _,
        row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
        col_byte_stride: (std::mem::size_of::<T>() * csc) as isize,
        item_size: std::mem::size_of::<T>(),
    }
}
