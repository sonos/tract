use std::fmt::Debug;
use std::marker::PhantomData;

pub trait TanhFunc {
    fn tanh(self) -> Self;
}

impl TanhFunc for f32 {
    fn tanh(self) -> f32 {
        crate::generic::tanh::stanh(self)
    }
}

pub trait Tanh<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync + TanhFunc,
{
    fn run(&self, vec: &mut [T]);
}

dyn_clone::clone_trait_object!(<T> Tanh<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct TanhImpl<K, T>
where
    T: Copy + Debug + PartialEq + Send + Sync + TanhFunc,
    K: TanhKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> Tanh<T> for TanhImpl<K, T>
where
    T: Copy + Debug + PartialEq + Send + Sync + TanhFunc,
    K: TanhKer<T> + Clone,
{
    fn run(&self, vec: &mut [T]) {
        if vec.len() == 0 {
            return;
        }
        let alignment = K::alignment_bytes();
        let mut offset = 0;
        unsafe {
            while offset < vec.len() && &vec[offset] as *const T as usize % alignment != 0 {
                *vec.get_unchecked_mut(offset) = vec.get_unchecked(offset).tanh();
                offset += 1;
            }
            let len = (vec.len() - offset) / K::nr() * K::nr();
            if len > 0 {
                K::run(&mut vec[offset..][..len]);
            }
            for i in (len + offset)..vec.len() {
                *vec.get_unchecked_mut(i) = vec.get_unchecked(i).tanh();
            }
        }
    }
}

pub trait TanhKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
    fn nr() -> usize;
    fn run(vec: &mut [T]);
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::TanhKer;
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! tanh_frame_tests {
        ($cond:expr, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn tanh(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        crate::frame::tanh::test::test_tanh::<$ker>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn tanh_4_magic() {
                if $cond {
                    crate::frame::tanh::test::test_tanh::<$ker>(&[0f32, -20.0, 20.0, 0.0]).unwrap()
                }
            }

            #[test]
            fn tanh_4zeros() {
                if $cond {
                    crate::frame::tanh::test::test_tanh::<$ker>(&[0.0; 4]).unwrap();
                }
            }

            #[test]
            fn tanh_20_ones() {
                crate::frame::tanh::test::test_tanh::<$ker>(&[1.0; 20]).unwrap();
            }

            #[test]
            fn tanh_18_zeros() {
                if $cond {
                    crate::frame::tanh::test::test_tanh::<$ker>(&[0.0; 18]).unwrap();
                }
            }
        };
    }

    pub fn test_tanh<K: TanhKer<f32>>(values: &[f32]) -> TestCaseResult {
        use crate::frame::tanh::Tanh;
        let op = crate::frame::tanh::TanhImpl::<K, f32>::new();
        let mut found = values.to_vec();
        op.run(&mut found);
        let expected = values.iter().map(|x| x.tanh()).collect::<Vec<_>>();
        crate::check_close(&*found, &*expected)
    }
}
