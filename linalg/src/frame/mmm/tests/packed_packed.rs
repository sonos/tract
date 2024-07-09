use crate::frame::block_quant::PackedBlockQuantFormat;
use crate::frame::mmm::*;
use pack::PackedFormat;
use proptest::collection::vec;
use proptest::prelude::*;
use std::fmt::Debug;
use tests::display_error;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_packed_packed_tests {
    ($cond:expr, $ker:ident, $packing_id:ident : $packing: expr) => {
        mod $packing_id {
            #[allow(unused_imports)]
            use super::$ker;
            use proptest::prelude::*;
            #[allow(unused_imports)]
            use tract_data::prelude::f16;
            use tract_data::prelude::*;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::packed_packed::*;
            use $crate::frame::mmm::MatMatMulKer;

            proptest::proptest! {
                #[test]
                fn packed_packed_prop(pb in any_with::<PackedPackedProblem<_>>(($ker, $packing))) {
                    if $cond {
                        pb.check().unwrap()
                    }
                }
            }

            #[test]
            fn packed_packed_1() -> TractResult<()> {
                if $cond {
                    let a = vec![1f32; $ker.mr()];
                    let b = vec![1f32; $ker.nr()];
                    PackedPackedProblem::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_2() -> TractResult<()> {
                if $cond {
                    let a = vec![1f32; $ker.mr() * 2];
                    let b = vec![1f32; $ker.nr() * 2];
                    PackedPackedProblem::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_13() -> TractResult<()> {
                if $cond {
                    let a = vec![1f32; $ker.mr() * 13];
                    let b = vec![1f32; $ker.nr() * 13];
                    PackedPackedProblem::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_a_scale() -> TractResult<()> {
                if $cond {
                    let a = (1..=$ker.mr() as i64).map(|x| x as f32).collect();
                    let b = vec![1f32; $ker.nr()];
                    PackedPackedProblem::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_a_scale_times_2() -> TractResult<()> {
                if $cond {
                    let a = (1..=2 * $ker.mr() as i64).map(|x| x as f32).collect();
                    let b = vec![1f32; $ker.nr() * 2];
                    PackedPackedProblem::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_empty() -> TractResult<()> {
                if $cond {
                    PackedPackedProblem::new($ker, $packing, vec![0f32; 0], vec![0f32; 0])
                        .check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_bug_1() -> TractResult<()> {
                if $cond {
                    PackedPackedProblem::new(
                        $ker,
                        $packing,
                        vec![0f32; $ker.mr()],
                        vec![0f32; $ker.nr()],
                    )
                    .check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_bug_2() -> TractResult<()> {
                if $cond {
                    let mut a = vec![0f32; $ker.mr()];
                    a[0] = 1.;
                    let mut b = vec![0f32; $ker.nr()];
                    b[0] = 1.;
                    PackedPackedProblem::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }

            #[test]
            fn packed_packed_bug_3() -> TractResult<()> {
                if $cond && $ker.mr() >= 4 {
                    let mut a = vec![0f32; 2 * $ker.mr()];
                    let mut b = vec![0f32; 2 * $ker.nr()];
                    a[2] = -0.7548828f32;
                    a[3] = 0.23547363f32;
                    b[2 * $ker.nr() - 1] = 0.93603516;
                    PackedPackedProblem::new($ker, $packing, a, b).check()?;
                }
                Ok(())
            }
        }
    };
}

#[derive(Debug, new)]
pub struct PackedPackedProblem<K>
where
    K: MatMatMulKer,
{
    pub ker: K,
    pub packing: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

/*
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

pub fn input_dts(ker: &impl MatMatMulKer, packing: usize) -> (DatumType, DatumType) {
    let (a_pack, b_pack) = ker.packings()[0];
    let dta = if let Some(pf) = a_pack.downcast_ref::<PackedFormat>() {
        pf.dt
    } else {
        f32::datum_type()
    };
    let dtb = if let Some(pf) = b_pack.downcast_ref::<PackedFormat>() {
        pf.dt
    } else {
        f32::datum_type()
    };
    (dta, dtb)
}
*/

impl<K> Arbitrary for PackedPackedProblem<K>
where
    K: MatMatMulKer + Default,
{
    type Parameters = (K, usize);
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((ker, packing): Self::Parameters) -> Self::Strategy {
        (0usize..100)
            .prop_flat_map(|k| {
                let ker = K::default();
                let m = k * ker.mr();
                let n = k * ker.nr();
                let range =
                    if ker.internal_type().is_integer() { (-5f32)..5f32 } else { (-1f32)..1f32 };
                (vec(range.clone(), m..=m), vec(range, n..=n))
            })
            .prop_map(move |(a, b)| Self { ker, packing, a, b })
            .boxed()
    }
}

impl<K: MatMatMulKer + Default> PackedPackedProblem<K> {
    pub fn padded_inputs(&self) -> TractResult<(Tensor, Tensor)> {
        let (pack_a, pack_b) = self.ker.packings()[self.packing];
        assert!(pack_b.k_alignment() == 1);
        let k = self.a.len() / self.ker.mr();
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());

        let mut a = Tensor::zero::<f32>(&[pack_a.r(), k_aligned])?;
        for row in 0..pack_a.r() {
            for col in 0..k {
                a.to_array_view_mut()?[[row, col]] = self.a[col + k * row];
            }
        }
        if let Some(pf) = pack_a.downcast_ref::<PackedFormat>() {
            a = a.cast_to_dt(pf.dt)?.into_owned();
        }
        let mut b = Tensor::zero::<f32>(&[k_aligned, pack_b.r()])?;
        for row in 0..k {
            for col in 0..pack_b.r() {
                b.to_array_view_mut()?[[row, col]] = self.b[col + pack_b.r() * row];
            }
        }
        if let Some(pf) = pack_b.downcast_ref::<PackedFormat>() {
            b = b.cast_to_dt(pf.dt)?.into_owned();
        }

        Ok((a, b))
    }

    pub fn reference(&self) -> TractResult<Tensor> {
        let mr = self.ker.mr();
        let nr = self.ker.nr();
        ensure!(self.a.len() / mr == self.b.len() / nr);
        let k = self.a.len() / mr;
        let pack_a = self.ker.packings()[self.packing].0;
        let (mut a, b) = self.padded_inputs()?;
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());
        if let Some(pbqf) = pack_a.downcast_ref::<PackedBlockQuantFormat>() {
            a = pbqf.simulate_precision_loss(a, 1)?;
        };
        let mut c = Tensor::zero::<K::Acc>(&[mr, nr])?;

        let a = a.cast_to::<K::Acc>()?;
        let a = a.as_slice::<K::Acc>()?;
        let b = b.cast_to::<K::Acc>()?;
        let b = b.as_slice::<K::Acc>()?;
        let mut view = c.to_array_view_mut::<K::Acc>()?.into_dimensionality()?;
        for m in 0..mr {
            for n in 0..nr {
                for ik in 0..k {
                    let a = a[ik + k_aligned * m];
                    let b = b[n + nr * ik];
                    view[(m, n)] += a * b;
                }
            }
        }
        Ok(c)
    }

    pub fn run(&self) -> TractResult<Tensor> {
        let (pack_a, pack_b) = self.ker.packings()[self.packing];
        assert!(pack_b.k_alignment() == 1);
        let k = self.a.len() / self.ker.mr();
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());

        let (a, b) = self.padded_inputs()?;
        let pa = pack_a.prepare_tensor(&a, 1, 0)?;
        let pb = pack_b.prepare_tensor(&b, 0, 1)?;

        let mut v = Tensor::zero_dt(self.ker.internal_type(), &[self.ker.mr(), self.ker.nr()])?;
        let item_size = self.ker.internal_type().size_of();

        let c = OutputStoreKer {
            ptr: v.as_bytes_mut().as_mut_ptr(),
            row_byte_stride: (item_size * self.ker.nr()) as isize,
            col_byte_stride: item_size as isize,
            item_size,
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
        Ok(v)
    }

    pub fn check(&self) -> TractResult<()> {
        dbg!(1);
        let expected = self.reference()?;
        dbg!(2);
        let found = self.run()?;
        dbg!(3);
        let app = if K::Acc::datum_type() == f16::datum_type() {
            Approximation::SuperApproximate
        } else {
            Approximation::Approximate
        };
        let result = found.close_enough(&expected, app);
        if result.is_err() {
            let exp = expected.as_slice::<K::Acc>()?;
            let found = found.as_slice::<K::Acc>()?;
            display_error(found, exp, self.ker.mr(), self.ker.nr());
        }
        result
    }
}
