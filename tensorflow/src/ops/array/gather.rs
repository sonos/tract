use ndarray::*;

use tract_core::internal::*;

#[derive(Debug, Clone, new)]
pub struct GatherNd {}

pub fn gather_nd(_pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    Ok(Box::new(GatherNd::new()))
}

impl GatherNd {
    fn eval_t<T: Datum + Copy>(
        &self,
        data: &SharedTensor,
        indices: &ArrayViewD<i32>,
    ) -> TractResult<TVec<SharedTensor>> {
        let data = data.to_array_view::<T>()?;
        let mut shape: TVec<usize> = indices.shape().into();
        let n = shape.pop().unwrap();
        shape.extend(data.shape()[n..].iter().cloned());
        let mut array = unsafe { ArrayD::<T>::uninitialized(&*shape) };
        for prefix in ndarray::indices(&indices.shape()[0..indices.ndim() - 1]) {
            let mut dst = array.view_mut();
            let mut coords = indices.view();
            for &x in prefix.slice().iter() {
                dst.index_axis_inplace(Axis(0), x);
                coords.index_axis_inplace(Axis(0), x);
            }
            let mut src = data.view();
            for &x in coords.iter() {
                src.index_axis_inplace(Axis(0), x as _);
            }
            dst.assign(&src);
        }
        Ok(tvec![array.into_arc_tensor()])
    }
}

impl Op for GatherNd {
    fn name(&self) -> Cow<str> {
        "tf.GatherNd".into()
    }
}

impl StatelessOp for GatherNd {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (data, indices) = args_2!(inputs);
        let indices = indices.cast_to::<i32>()?;
        let indices = indices.to_array_view::<i32>()?;
        dispatch_copy!(Self::eval_t(data.datum_type())(self, &data, &indices))
    }
}

impl InferenceRulesOp for GatherNd {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[1].rank, move |s, indices_rank| {
            let indices_rank = indices_rank as usize;
            for i in 0..(indices_rank - 1) {
                s.equals(&outputs[0].shape[i], &inputs[1].shape[i])?;
            }
            s.given_2(
                &inputs[1].shape[indices_rank - 1],
                &inputs[1].rank,
                move |s, n, input_rank| {
                    if let Ok(n) = n.to_integer() {
                        for i in 0..(input_rank - n) as usize {
                            s.equals(&outputs[0].shape[indices_rank - 1 + i], &inputs[1].shape[i])?;
                        }
                    }
                    Ok(())
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // https://www.tensorflow.org/api_docs/python/tf/gather_nd
    #[test]
    fn simple_indexing() {
        let g = GatherNd::new();
        assert_eq!(
            g.eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[0, 0], [1, 1]]))).unwrap(),
            tvec!(rctensor1(&[1, 4]))
        );
    }

    #[test]
    fn slice_indexing() {
        let g = GatherNd::new();
        assert_eq!(
            g.eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[1], [0]]))).unwrap(),
            tvec!(rctensor2(&[[3, 4], [1, 2]]))
        );
    }

    #[test]
    fn tensor_3d_1() {
        let g = GatherNd::new();
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[1]]))).unwrap(),
            tvec!(rctensor3(&[[[11, 21], [31, 41]]]))
        );
    }

    #[test]
    fn tensor_3d_2() {
        let g = GatherNd::new();
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[0, 1], [1, 0]]))).unwrap(),
            tvec!(rctensor2(&[[30, 40], [11, 21]]))
        );
    }

    #[test]
    fn tensor_3d_3() {
        let g = GatherNd::new();
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[0, 0, 1], [1, 0, 1]]))).unwrap(),
            tvec!(rctensor1(&[20, 21]))
        );
    }
}
