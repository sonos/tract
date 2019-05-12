//! N-way tensor broadcast
use crate::model::TVec;
use num_traits::One;

/// Computes a shape, if any, to which all shapes can be broadcasted.
pub fn multi_broadcast<T>(shapes: &[impl AsRef<[T]>]) -> Option<TVec<T>>
where
    T: One + PartialEq + Copy,
{
    let len = shapes.iter().map(|shape| shape.as_ref().len()).max()?;
    let mut shape = tvec!();
    for i in 0..len {
        let mut wanted_size = T::one();
        for shape in shapes {
            let len = shape.as_ref().len();
            let dim = if i < len { shape.as_ref()[len - i - 1] } else { T::one() };
            if dim != T::one() {
                if wanted_size != T::one() && dim != wanted_size {
                    return None;
                }
                wanted_size = dim;
            }
        }
        shape.push(wanted_size)
    }
    shape.reverse();
    Some(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_1() {
        assert_eq!(multi_broadcast(&tvec![tvec![2, 3, 4, 5], tvec![]]), Some(tvec![2, 3, 4, 5]))
    }

    #[test]
    fn onnx_2() {
        assert_eq!(multi_broadcast(&tvec![tvec![2, 3, 4, 5], tvec![5]]), Some(tvec![2, 3, 4, 5]))
    }

    #[test]
    fn onnx_3() {
        assert_eq!(multi_broadcast(&tvec![tvec![4, 5], tvec![2, 3, 4, 5]]), Some(tvec![2, 3, 4, 5]))
    }

    #[test]
    fn onnx_4() {
        assert_eq!(
            multi_broadcast(&tvec![tvec![1, 4, 5], tvec![2, 3, 4, 1]]),
            Some(tvec![2, 3, 4, 5])
        )
    }

    #[test]
    fn onnx_5() {
        assert_eq!(
            multi_broadcast(&tvec![tvec![3, 4, 5], tvec![2, 1, 1, 1]]),
            Some(tvec![2, 3, 4, 5])
        )
    }
}
