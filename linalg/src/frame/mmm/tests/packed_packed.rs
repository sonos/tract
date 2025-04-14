use crate::block_quant::PackedBlockQuantFormat;
use crate::mmm::tests::display_error;
use crate::mmm::{AsInputValue, FusedKerSpec, FusedSpec, MatMatMul, MatMatMulKer, OutputStoreKer};
use crate::pack::PackedFormat;
use proptest::collection::vec;
use proptest::prelude::*;
use std::fmt::Debug;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_packed_packed_tests {
    ($ker:expr, $packing_id:ident : $packing: expr) => {
        mod $packing_id {
            use super::*;
            #[allow(unused_imports)]
            use proptest::prelude::*;
            #[allow(unused_imports)]
            use tract_data::prelude::f16;
            use tract_data::prelude::*;
            use tract_itertools::Itertools;
            use $crate::frame::mmm::kernel::MatMatMulKer;
            #[allow(unused_imports)]
            use $crate::frame::mmm::tests::packed_packed::*;

            mod fuse {
                use super::*;

                proptest::proptest! {
                    #[test]
                    fn prop(pb in arbitrary_problem(false, $ker, $packing)) {
                        pb.check().unwrap()
                    }
                }

                fn t(a: impl Into<Vec<f32>>, b: impl Into<Vec<f32>>) -> TractResult<()> {
                    PackedPackedProblem::kernel($ker, $packing, a, b).check()
                }

                #[test]
                fn packed_packed_1() -> TractResult<()> {
                    t(vec![1f32; $ker.mr()], vec![1f32; $ker.nr()])
                }

                #[test]
                fn packed_packed_2() -> TractResult<()> {
                    t(vec![1f32; $ker.mr() * 2], vec![1f32; $ker.nr() * 2])
                }

                #[test]
                fn packed_packed_13() -> TractResult<()> {
                    t(vec![1f32; $ker.mr() * 13], vec![1f32; $ker.nr() * 13])
                }

                #[test]
                fn packed_packed_a_scale() -> TractResult<()> {
                    t((1..=$ker.mr() as i64).map(|x| x as f32).collect_vec(), vec![1f32; $ker.nr()])
                }

                #[test]
                fn packed_packed_a_scale_times_2() -> TractResult<()> {
                    t(
                        (1..=2 * $ker.mr() as i64).map(|x| x as f32).collect_vec(),
                        vec![1f32; $ker.nr() * 2],
                    )
                }

                #[test]
                fn packed_packed_empty() -> TractResult<()> {
                    t(vec![0f32; 0], vec![0f32; 0])
                }

                #[test]
                fn packed_packed_bug_1() -> TractResult<()> {
                    t(vec![0f32; $ker.mr()], vec![0f32; $ker.nr()])
                }

                #[test]
                fn packed_packed_bug_2() -> TractResult<()> {
                    let mut a = vec![0f32; $ker.mr()];
                    a[0] = 1.;
                    let mut b = vec![0f32; $ker.nr()];
                    b[0] = 1.;
                    t(a, b)
                }

                #[test]
                fn packed_packed_bug_3() -> TractResult<()> {
                    if $ker.mr() >= 4 {
                        let mut a = vec![0f32; 2 * $ker.mr()];
                        let mut b = vec![0f32; 2 * $ker.nr()];
                        a[2] = -0.7548828f32;
                        a[3] = 0.23547363f32;
                        b[2 * $ker.nr() - 1] = 0.93603516;
                        t(a, b)?;
                    }
                    Ok(())
                }

                #[test]
                fn packed_packed_bug_4() -> TractResult<()> {
                    if $ker.mr() > 16 {
                        let mut a = vec![0f32; $ker.mr()];
                        let mut b = vec![0f32; $ker.nr()];
                        a[16] = 1.;
                        b[0] = 1.;
                        t(a, b)?;
                    }
                    Ok(())
                }
            }

            mod frame {
                use super::*;

                proptest::proptest! {
                    #[test]
                    fn prop(pb in arbitrary_problem(true, $ker, $packing)) {
                        pb.check().unwrap()
                    }
                }

                fn t(
                    m: usize,
                    n: usize,
                    a: impl Into<Vec<f32>>,
                    b: impl Into<Vec<f32>>,
                ) -> TractResult<()> {
                    PackedPackedProblem::frame($ker, $packing, m, n, a, b).check()
                }

                fn ti(
                    m: usize,
                    n: usize,
                    a: impl Into<Vec<i32>>,
                    b: impl Into<Vec<i32>>,
                ) -> TractResult<()> {
                    let a = a.into().into_iter().map(|i| i as f32).collect_vec();
                    let b = b.into().into_iter().map(|i| i as f32).collect_vec();
                    t(m, n, a, b)
                }

                #[test]
                fn trivial_1x2() -> TractResult<()> {
                    ti(1, 2, [0], [0, 0])
                }

                #[test]
                fn packed_packed_empty() -> TractResult<()> {
                    t($ker.mr(), $ker.nr(), [], [])
                }

                #[test]
                fn packed_packed_empty_2() -> TractResult<()> {
                    t(2 * $ker.mr(), 2 * $ker.nr(), [], [])
                }

                #[test]
                fn mat_mul_1() -> TractResult<()> {
                    ti(3, 2, [-3, 3, 5, -5, 6, 0, -6, -5, 0, 0, 9, 7], [-8, 5, 5, -3, 5, 7, -8, -1])
                }

                #[test]
                fn mat_mul_2() -> TractResult<()> {
                    ti(1, 3, [122, 82], [0, 0, 37, 0, 0, 57])
                }
            }
        }
    };
}

#[derive(Debug, new)]
pub struct PackedPackedProblem<K>
where
    K: MatMatMulKer,
{
    pub frame_test: Option<(usize, usize)>,
    pub ker: K,
    pub packing: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

pub fn arbitrary_problem<K: MatMatMulKer>(
    frame_test: bool,
    ker: &K,
    packing: usize,
) -> BoxedStrategy<PackedPackedProblem<K>> {
    let (mr, nr) = (ker.mr(), ker.nr());
    let item_range = if ker.internal_type().is_integer() { (-5f32)..5f32 } else { (-1f32)..1f32 };
    let (m_range, n_range) =
        if frame_test { (1usize..3 * mr, 1usize..3 * nr) } else { (mr..mr + 1, nr..nr + 1) };
    let ker = ker.clone();
    (m_range, 0usize..40, n_range)
        .prop_flat_map(move |(m, k, n)| {
            (
                vec(item_range.clone(), k * m..=k * m),
                vec(item_range.clone(), k * n..=k * n),
                Just((m, n)),
            )
        })
        .prop_map(move |(mut a, mut b, mn)| {
            a.reverse();
            b.reverse();
            PackedPackedProblem {
                frame_test: Some(mn).filter(|_| frame_test),
                ker: ker.clone(),
                packing,
                a,
                b,
            }
        })
        .boxed()
}

impl<K: MatMatMulKer> PackedPackedProblem<K> {
    pub fn kernel(
        ker: &K,
        packing: usize,
        a: impl Into<Vec<f32>>,
        b: impl Into<Vec<f32>>,
    ) -> PackedPackedProblem<K> {
        PackedPackedProblem {
            frame_test: None,
            ker: ker.clone(),
            packing,
            a: a.into(),
            b: b.into(),
        }
    }

    pub fn frame(
        ker: &K,
        packing: usize,
        m: usize,
        n: usize,
        a: impl Into<Vec<f32>>,
        b: impl Into<Vec<f32>>,
    ) -> PackedPackedProblem<K> {
        PackedPackedProblem {
            frame_test: Some((m, n)),
            ker: ker.clone(),
            packing,
            a: a.into(),
            b: b.into(),
        }
    }

    pub fn mkn(&self) -> (usize, usize, usize) {
        let (m, n) = self.frame_test.unwrap_or((self.ker.mr(), self.ker.nr()));
        assert!(m != 0 && n != 0);
        let k = self.a.len() / m;
        assert_eq!(self.b.len() / n, k);
        (m, k, n)
    }

    pub fn padded_inputs(&self) -> TractResult<(Tensor, Tensor)> {
        let (pack_a, pack_b) = &self.ker.packings()[self.packing];
        assert!(pack_b.k_alignment() == 1);
        let (m, k, n) = self.mkn();
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());

        let mut a = Tensor::zero::<f32>(&[m, k_aligned])?;
        for row in 0..m {
            for col in 0..k {
                a.to_array_view_mut()?[[row, col]] = self.a[col + k * row];
            }
        }
        if let Some(pf) = pack_a.downcast_ref::<PackedFormat>() {
            a = a.cast_to_dt(pf.dt)?.into_owned();
        }
        let mut b = Tensor::zero::<f32>(&[k_aligned, n])?;
        for row in 0..k {
            for col in 0..n {
                b.to_array_view_mut()?[[row, col]] = self.b[col + n * row];
            }
        }
        if let Some(pf) = pack_b.downcast_ref::<PackedFormat>() {
            b = b.cast_to_dt(pf.dt)?.into_owned();
        }

        Ok((a, b))
    }

    pub fn reference(&self) -> TractResult<Tensor> {
        let (m, k, n) = self.mkn();
        let pack_a = &self.ker.packings()[self.packing].0;
        let (mut a, b) = self.padded_inputs()?;
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());
        if let Some(pbqf) = pack_a.downcast_ref::<PackedBlockQuantFormat>() {
            a = pbqf.simulate_precision_loss(a, 1)?;
        };
        let mut c = Tensor::zero::<K::Acc>(&[m, n])?;

        let a = a.cast_to::<K::Acc>()?;
        let a = a.as_slice::<K::Acc>()?;
        let b = b.cast_to::<K::Acc>()?;
        let b = b.as_slice::<K::Acc>()?;
        let mut view = c.to_array_view_mut::<K::Acc>()?.into_dimensionality()?;
        for ix_m in 0..m {
            for ix_n in 0..n {
                for ix_k in 0..k {
                    let a = a[ix_k + k_aligned * ix_m];
                    let b = b[ix_n + n * ix_k];
                    view[(ix_m, ix_n)] += a * b;
                }
            }
        }
        Ok(c)
    }

    pub fn run(&self) -> TractResult<Tensor> {
        let (m, k, n) = self.mkn();
        let (pack_a, pack_b) = &self.ker.packings()[self.packing];
        assert!(pack_b.k_alignment() == 1);
        let k_aligned = k.next_multiple_of(pack_a.k_alignment());

        let (a, b) = self.padded_inputs()?;
        let pa = pack_a.prepare_one(&a, 1, 0)?;
        let pb = pack_b.prepare_one(&b, 0, 1)?;

        let mut v = unsafe { Tensor::uninitialized_dt(self.ker.internal_type(), &[m, n])? };
        let item_size = self.ker.internal_type().size_of();

        if self.frame_test.is_some() {
            unsafe {
                let c = self.ker.c_view(Some(0), Some(1)).wrap(&v.view_mut());
                let ops = tvec!(
                    FusedSpec::AddMatMul {
                        a: AsInputValue::Borrowed(&*pa),
                        b: AsInputValue::Borrowed(&*pb),
                        packing: self.packing
                    },
                    FusedSpec::Store(c)
                );
                self.ker.run(m, n, &ops)?;
            }
        } else {
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
        }
        Ok(v)
    }

    pub fn check(&self) -> TractResult<()> {
        if !self.ker.is_supported_here() {
            return Ok(());
        }
        let expected = self.reference()?;
        let found = self.run()?;
        let app = if K::Acc::datum_type() == f16::datum_type() {
            Approximation::SuperApproximate
        } else {
            Approximation::Approximate
        };
        let result = found.close_enough(&expected, app);
        if result.is_err() {
            let exp = expected.as_slice::<K::Acc>()?;
            let found = found.as_slice::<K::Acc>()?;
            let (m, _, n) = self.mkn();
            display_error(found, exp, m, n);
        }
        result
    }
}
