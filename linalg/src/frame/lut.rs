use crate::align::Buffer;
use std::marker::PhantomData;

pub trait Lut {
    fn run(&self, buf: &mut [u8]);
}

#[derive(Debug, Clone)]
pub struct LutImpl<K>
where
    K: LutKer,
{
    table: Buffer<u8>,
    _boo: PhantomData<K>,
}

impl<K> LutImpl<K>
where
    K: LutKer,
{
    pub fn new(table: &[u8]) -> LutImpl<K> {
        LutImpl {
            table: Buffer::realign_data(table, K::table_alignment_bytes()),
            _boo: PhantomData,
        }
    }
}

impl<K> Lut for LutImpl<K>
where
    K: LutKer,
{
    fn run(&self, buf: &mut [u8]) {
        let align = K::input_alignment_bytes();
        let aligned_start = (buf.as_ptr() as usize + align - 1) / align * align;
        let prefix = (aligned_start - buf.as_ptr() as usize).min(buf.len());
        for i in 0..(prefix as isize) {
            unsafe {
                let ptr = buf.as_mut_ptr().offset(i);
                *ptr = self.table[*ptr as usize];
            }
        }
        let remaining = buf.len() - prefix;
        if remaining == 0 {
            return
        }
        let n = K::n();
        let aligned_len = remaining / n * n;
        if aligned_len > 0 {
            unsafe {
                K::run(
                    buf.as_mut_ptr().offset(prefix as isize),
                    aligned_len,
                    self.table.as_ptr(),
                );
            }
        }
        let remaining = buf.len() - aligned_len - prefix;
        for i in 0..remaining {
            unsafe {
                let ptr = buf.as_mut_ptr().offset((i + prefix + aligned_len) as isize);
                *ptr = self.table[*ptr as usize];
            }
        }
    }
}

pub trait LutKer {
    fn name() -> &'static str;
    fn n() -> usize;
    fn input_alignment_bytes() -> usize;
    fn table_alignment_bytes() -> usize;
    fn run(buf: *mut u8, len: usize, table: *const u8);
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
                    let pb = LutProblem { table: vec!(0), data: vec!() };
                    assert_eq!(pb.test::<$ker>(), pb.reference())
                }
            }
        }
    }
}
