use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use tract_data::internal::*;

pub trait Lut: fmt::Debug + dyn_clone::DynClone + Send + Sync {
    fn table(&self) -> &[u8];
    fn run(&self, buf: &mut [u8]);
}

dyn_clone::clone_trait_object!(Lut);

#[derive(Debug, Clone, Hash)]
pub struct LutImpl<K: LutKer> {
    table: Tensor,
    _boo: PhantomData<K>,
}

impl<K: LutKer> LutImpl<K> {
    pub fn new(table: &[u8]) -> LutImpl<K> {
        unsafe {
            LutImpl {
                table: Tensor::from_raw_aligned::<u8>(
                    &[table.len()],
                    table,
                    K::table_alignment_bytes(),
                )
                .unwrap(),
                _boo: PhantomData,
            }
        }
    }
}

impl<K: LutKer> Lut for LutImpl<K> {
    fn table(&self) -> &[u8] {
        self.table.as_slice().unwrap()
    }

    fn run(&self, buf: &mut [u8]) {
        unsafe {
            let table: *const u8 = self.table.as_ptr_unchecked();
            let align = K::input_alignment_bytes();
            let aligned_start = (buf.as_ptr() as usize).next_multiple_of(align);
            let prefix = (aligned_start - buf.as_ptr() as usize).min(buf.len());
            for i in 0..(prefix as isize) {
                let ptr = buf.as_mut_ptr().offset(i);
                *ptr = *table.offset(*ptr as isize);
            }
            let remaining = buf.len() - prefix;
            if remaining == 0 {
                return;
            }
            let n = K::n();
            let aligned_len = remaining / n * n;
            if aligned_len > 0 {
                K::run(buf.as_mut_ptr().add(prefix), aligned_len, table);
            }
            let remaining = buf.len() - aligned_len - prefix;
            for i in 0..remaining {
                let ptr = buf.as_mut_ptr().add(i + prefix + aligned_len);
                *ptr = *table.offset(*ptr as isize);
            }
        }
    }
}

pub trait LutKer: Clone + fmt::Debug + Send + Sync + Hash {
    fn name() -> &'static str;
    fn n() -> usize;
    fn input_alignment_bytes() -> usize;
    fn table_alignment_bytes() -> usize;
    unsafe fn run(buf: *mut u8, len: usize, table: *const u8);
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug)]
    pub struct LutProblem {
        pub table: Vec<u8>,
        pub data: Vec<u8>,
    }

    impl Arbitrary for LutProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_p: ()) -> Self::Strategy {
            proptest::collection::vec(any::<u8>(), 1..256)
                .prop_flat_map(|table| {
                    let data = proptest::collection::vec(0..table.len() as u8, 0..100);
                    (Just(table), data)
                })
                .prop_map(|(table, data)| LutProblem { table, data })
                .boxed()
        }
    }

    impl LutProblem {
        pub fn reference(&self) -> Vec<u8> {
            self.data.iter().map(|x| self.table[*x as usize]).collect()
        }

        pub fn test<K: LutKer>(&self) -> Vec<u8> {
            let lut = LutImpl::<K>::new(&self.table);
            let mut data = self.data.clone();
            lut.run(&mut data);
            data
        }
    }

    #[macro_export]
    macro_rules! lut_frame_tests {
        ($cond:expr, $ker:ty) => {
            mod lut {
                use proptest::prelude::*;
                #[allow(unused_imports)]
                use $crate::frame::lut::test::*;

                proptest::proptest! {
                    #[test]
                    fn lut_prop(pb in any::<LutProblem>()) {
                        if $cond {
                            prop_assert_eq!(pb.test::<$ker>(), pb.reference())
                        }
                    }
                }

                #[test]
                fn test_empty() {
                    let pb = LutProblem { table: vec![0], data: vec![] };
                    assert_eq!(pb.test::<$ker>(), pb.reference())
                }
            }
        };
    }
}
