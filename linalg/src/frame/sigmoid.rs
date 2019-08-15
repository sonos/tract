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

pub trait Sigmoid<T>: Send + Sync + Debug + objekt::Clone
where
    T: Copy + Debug + PartialEq + Send + Sync + SigmoidFunc,
{
    fn run(&self, vec: &mut [T]);
}

clone_trait_object!(<T> Sigmoid<T> where T: Copy);

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

pub trait SigmoidKer<T>: Send + Sync + Debug + objekt::Clone
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
    /*
    use super::*;
    use crate::align;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;
    */

    #[macro_export]
    macro_rules! sigmoid_frame_tests {
        ($cond:expr, $ker:ty) => {
            mod frame {
                proptest::proptest!{
                    #[test]
                    fn sigmoid(xs in proptest::collection::vec(-25f32..25.0, 0..100)) {
                        use crate::frame::sigmoid;
                        use sigmoid::Sigmoid;
                        let op = sigmoid::SigmoidImpl::<$ker, f32>::new();
                        if $cond {
                            let mut found = xs.clone();
                            op.run(&mut *found);
                            let expected = xs.iter().map(|x| 1.0/(1.0 + (-x).exp())).collect::<Vec<_>>();
                            crate::check_close(&*found, &*expected).unwrap()
                        }
                    }
                }

                #[test]
                fn test_4_magic() {
                    use crate::frame::sigmoid::Sigmoid;
                    let op = crate::frame::sigmoid::SigmoidImpl::<$ker, f32>::new();
                    if $cond {
                        let mut values = vec!(0f32, -20.0, 20.0, 0.0);
                        op.run(&mut *values);
                        crate::check_close(&*values, &[0.5f32, 0.0, 1.0, 0.5]).unwrap();
                    }
                }

                #[test]
                fn test_4zeros() {
                    use crate::frame::sigmoid::Sigmoid;
                    let op = crate::frame::sigmoid::SigmoidImpl::<$ker, f32>::new();
                    if $cond {
                        let mut zeroes = vec!(0f32; 4);
                        op.run(&mut zeroes);
                        crate::check_close(&*zeroes, &[0.5f32; 4]).unwrap();
                    }
                }
            }
        }
    }

}
