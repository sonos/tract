use crate::frame::block_quant::PackedBlockQuantFormat;
use crate::frame::mmm::*;
use crate::LADatum;
use num_traits::{AsPrimitive, Zero};
use proptest::collection::vec;
use proptest::prelude::*;
use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use tract_data::internal::*;

#[macro_export]
macro_rules! mmm_packed_packed_tests {
    ($cond:expr, $ker:ident, $packing_id:ident : $packing: expr, $ta:ty, $tb:ty, $tc:ty, $ti: ty) => {
        mod $packing_id {
            #[allow(unused_imports)]
            use super::$ker;
            use num_traits::Zero;
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
                        prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
                    }
                }
            }

            #[test]
            fn packed_packed_1() {
                if $cond {
                    packed_packed::<_, $ta, $tb, $tc, $ti>($ker, $packing, 1)
                }
            }

            #[test]
            fn packed_packed_2() {
                if $cond {
                    packed_packed::<_, $ta, $tb, $tc, $ti>($ker, $packing, 2)
                }
            }

            #[test]
            fn packed_packed_13() {
                if $cond {
                    packed_packed::<_, $ta, $tb, $tc, $ti>($ker, $packing, 13)
                }
            }

            #[test]
            fn packed_packed_empty() {
                if $cond {
                    let pb = PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                        $ker,
                        $packing,
                        0,
                        vec![<$ta>::zero(); 0],
                        vec![<$tb>::zero(); 0],
                        false,
                        );
                    assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
                }
            }

            #[test]
            fn packed_packed_bug_1() {
                if $cond {
                    let pb = PackedPackedProblem::<_, $ta, $tb, $tc, $ti>::new(
                        $ker,
                        $packing,
                        1,
                        vec![<$ta>::zero(); $ker.mr()],
                        vec![<$tb>::zero(); $ker.nr()],
                        true,
                        );
                    assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
                }
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
    pub k: usize,
    pub a: Vec<TA>,
    pub b: Vec<TB>,
    pub trans_c: bool,
    pub _phantom: PhantomData<(K, TC, TI)>,
}

impl<K, TA, TB, TC, TI> Arbitrary for PackedPackedProblem<K, TA, TB, TC, TI>
where
    K: MatMatMulKer<Acc = TI> + Default + Copy,
    TA: LADatum + AsPrimitive<TI>,
    TB: LADatum + AsPrimitive<TI>,
    TC: LADatum + Copy + PartialEq + 'static + Debug,
    TI: LADatum + fmt::Display + AsPrimitive<TC>,
    usize: AsPrimitive<TA> + AsPrimitive<TB>,
{
    type Parameters = (K, usize);
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((ker, packing): Self::Parameters) -> Self::Strategy {
        (0usize..20, any::<bool>())
            .prop_flat_map(|(k, trans_c)| {
                let ker = K::default();
                let m = k * ker.mr();
                let n = k * ker.nr();
                let a = (0usize..10).prop_map(|x| x.as_());
                let b = (0usize..10).prop_map(|x| x.as_());
                (Just(k), Just(trans_c), vec(a, m..=m), vec(b, n..=n))
            })
            .prop_map(move |(k, trans_c, a, b)| Self {
                ker,
                packing,
                k,
                a,
                b,
                trans_c,
                _phantom: PhantomData,
            })
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

    pub fn reference(&self) -> TractResult<Vec<TC>> {
        let mr = self.ker.mr();
        let nr = self.ker.nr();
        let pack_a = self.ker.packings()[self.packing].0;
        let (mut a, _b) = self.padded_inputs()?;
        let k_aligned = self.k.next_multiple_of(pack_a.k_alignment());
        if let Some(pbqf) = pack_a.downcast_ref::<PackedBlockQuantFormat>() {
            pbqf.simulate_precision_loss(&mut a, 1)?
        };
        let mut vi = vec![TI::zero(); mr * nr];
        for m in 0..mr {
            for n in 0..nr {
                for k in 0..self.k {
                    let a: TI = a.as_slice::<TA>()?[k + k_aligned * m].as_();
                    let b: TI = self.b[n + nr * k].as_();
                    let offset = if self.trans_c { m + n * mr } else { n + m * nr };
                    vi[offset] += a * b;
                }
            }
        }
        Ok(vi.into_iter().map(|ti| ti.as_()).collect())
    }

    pub fn run(&self) -> TractResult<Vec<TC>> {
        let pack_a = self.ker.packings()[self.packing].0;
        let pack_b = self.ker.packings()[self.packing].1;
        assert!(pack_b.k_alignment() == 1);
        let k_aligned = self.k.next_multiple_of(pack_a.k_alignment());

        let (a, b) = self.padded_inputs()?;
        let pa = pack_a.prepare_tensor(&a, 1, 0)?;
        let pb = pack_b.prepare_tensor(&b, 0, 1)?;

        let mut v = vec![TC::zero(); self.ker.mr() * self.ker.nr()];
        let c = if self.trans_c {
            mmm_stride_storage(&mut v, 1, self.ker.mr())
        } else {
            mmm_stride_storage(&mut v, self.ker.nr(), 1)
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
}

pub fn packed_packed<K, TA, TB, TC, TI>(ker: K, packing: usize, k: usize)
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
    let pb = PackedPackedProblem::<K, TA, TB, TC, TI>::new(ker, packing, k, a, b, false);
    assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
}

pub fn mmm_stride_storage<T: Copy>(v: &mut [T], rsc: usize, csc: usize) -> OutputStoreKer {
    OutputStoreKer {
        ptr: v.as_mut_ptr() as _,
        row_byte_stride: (std::mem::size_of::<T>() * rsc) as isize,
        col_byte_stride: (std::mem::size_of::<T>() * csc) as isize,
        item_size: std::mem::size_of::<T>(),
    }
}
