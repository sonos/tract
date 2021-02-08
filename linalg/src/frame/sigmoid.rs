use std::fmt::Debug;
use std::marker::PhantomData;

pub trait SigmoidFunc {
    fn sigmoid(self) -> Self;
}

impl SigmoidFunc for f32 {
    fn sigmoid(self) -> f32 {
        crate::generic::sigmoid::ssigmoid(self)
    }
}

pub trait Sigmoid<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync + SigmoidFunc,
{
    fn run(&self, vec: &mut [T]);
}

dyn_clone::clone_trait_object!(<T> Sigmoid<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct SigmoidImpl<K, T>
where
    T: Copy + Debug + PartialEq + Send + Sync + SigmoidFunc,
    K: SigmoidKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> Sigmoid<T> for SigmoidImpl<K, T>
where
    T: Copy + Debug + PartialEq + Send + Sync + SigmoidFunc,
    K: SigmoidKer<T> + Clone,
{
    fn run(&self, vec: &mut [T]) {
        if vec.len() == 0 {
            return;
        }
        let alignment = K::alignment_bytes();
        let mut offset = 0;
        unsafe {
            while offset < vec.len() && &vec[offset] as *const T as usize % alignment != 0 {
                *vec.get_unchecked_mut(offset) = vec.get_unchecked(offset).sigmoid();
                offset += 1;
            }
            let len = (vec.len() - offset) / K::nr() * K::nr();
            if len > 0 {
                K::run(&mut vec[offset..][..len]);
            }
            for i in (len + offset)..vec.len() {
                *vec.get_unchecked_mut(i) = vec.get_unchecked(i).sigmoid();
            }
        }
    }
}

pub trait SigmoidKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone
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
    use super::SigmoidKer;
    use proptest::test_runner::TestCaseResult;

    #[macro_export]
    macro_rules! sigmoid_frame_tests {
        ($cond:expr, $ker:ty) => {
            proptest::proptest! {
                #[test]
                fn sigmoid(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                    if $cond {
                        crate::frame::sigmoid::test::test_sigmoid::<$ker>(&*xs).unwrap()
                    }
                }
            }

            #[test]
            fn sigmoid_4_magic() {
                if $cond {
                    crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[0f32, -20.0, 20.0, 0.0])
                        .unwrap()
                }
            }

            #[test]
            fn sigmoid_4zeros() {
                if $cond {
                    crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[0.0; 4]).unwrap();
                }
            }

            #[test]
            fn sigmoid_20_ones() {
                crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[1.0; 20]).unwrap();
            }

            #[test]
            fn sigmoid_18_zeros() {
                if $cond {
                    crate::frame::sigmoid::test::test_sigmoid::<$ker>(&[0.0; 18]).unwrap();
                }
            }
        };
    }

    pub fn test_sigmoid<K: SigmoidKer<f32>>(values: &[f32]) -> TestCaseResult {
        use crate::frame::sigmoid::Sigmoid;
        let op = crate::frame::sigmoid::SigmoidImpl::<K, f32>::new();
        let mut found = values.to_vec();
        while found.len() < K::nr() {
            found.push(0f32);
        }
        op.run(&mut found);
        let expected = values.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect::<Vec<_>>();
        crate::test::check_close(&found[..values.len()], &*expected)
    }
}
