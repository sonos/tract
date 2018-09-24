use tfdeploy::analyser::rules::prelude::*;
use tfdeploy::ops::prelude::*;

pub fn squeeze(pb: &::tfpb::node_def::NodeDef) -> TfdResult<Box<Op>> {
    let mut squeeze_dims = pb.get_attr_opt_list_int("squeeze_dims")?;
    if let Some(ref mut squeeze_dims) = squeeze_dims {
        squeeze_dims.sort();
        squeeze_dims.reverse();
    }
    let t = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Squeeze(t)(squeeze_dims)))
}

#[derive(Debug, Clone, new)]
pub struct Squeeze<T: Datum> {
    squeeze_dims: Option<Vec<isize>>,
    t: PhantomData<T>,
}

impl<T: Datum> Squeeze<T> {
    fn squeezable(&self, ix: usize, d: usize, stream_dim: Option<usize>) -> bool {
        stream_dim != Some(ix) && d == 1
            && self
                .squeeze_dims
                .as_ref()
                .filter(|v| v.len() > 0)
                .map(|squeeze_dims| squeeze_dims.contains(&(ix as _)))
                .unwrap_or(true)
    }

    /// Removes the dimensions of size 1 from the given shape vector.
    fn squeeze_shape(&self, shape: &[usize], stream_dim: Option<usize>) -> Vec<usize> {
        shape
            .iter()
            .enumerate()
            .filter_map(|(ix, d)| {
                if self.squeezable(ix, *d, stream_dim) {
                    None
                } else {
                    Some(*d)
                }
            })
            .collect()
    }
}

impl<T: Datum> Op for Squeeze<T> {
    fn name(&self) -> &str {
        "tf.Squeeze"
    }
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let data = input.into_array::<T>()?;
        let shape = self.squeeze_shape(data.shape(), None);
        Ok(tvec![data.into_shape(shape)?.into()])
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        _buffer: &mut Box<OpBuffer>,
    ) -> TfdResult<Option<TVec<Value>>> {
        let input = args_1!(inputs);
        if let Some(Stream {
            info,
            chunk: Some(chunk),
            ..
        }) = input.into_stream()
        {
            let chunk = chunk.into_tensor().into_array::<T>()?;
            let shape = self.squeeze_shape(chunk.shape(), Some(info.axis));
            Ok(Some(tvec![Tensor::from(chunk.into_shape(shape)?).into(),]))
        } else {
            Ok(None)
        }
    }
}

impl<T: Datum> InferenceRulesOp for Squeeze<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datum_type, &outputs[0].datum_type)
            .equals(&inputs[0].datum_type, T::datum_type())
            .given(&inputs[0].shape, move |solver, shape: Vec<TDim>| {
                let stream_dim = shape.iter().position(|d| d.is_stream());
                let shape: Vec<TDim> = shape
                    .into_iter()
                    .enumerate()
                    .filter(|(ix, d)| {
                        !self.squeezable(*ix, d.to_integer().unwrap_or(1) as usize, stream_dim)
                    })
                    .map(|(_, d)| d)
                    .collect();
                solver.equals(&outputs[0].shape, shape);
            });
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use tfdeploy::dim::TDim;
    use ndarray::*;
    use tfdeploy::ops::InferenceOp;
    use tfdeploy::Tensor;

    fn run<I>(op: Squeeze<i32>, input: I) -> Tensor
    where
        I: Into<Tensor>,
    {
        op.eval(tvec![input.into().into()])
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
    }

    #[test]
    fn squeeze_1() {
        assert_eq!(
            run(Squeeze::new(None), Array::from_elem([1, 2, 1, 3, 1, 1], 0)).shape(),
            &[2, 3]
        );
    }

    #[test]
    fn squeeze_2() {
        assert_eq!(
            run(
                Squeeze::new(Some(vec![2, 4])),
                Array::from_elem([1, 2, 1, 3, 1, 1], 0)
            ).shape(),
            &[1, 2, 3, 1]
        );
    }

    #[test]
    fn squeeze_3() {
        // Tf considers empty list as absent list.
        assert_eq!(
            run(
                Squeeze::new(Some(vec![])),
                Array::from_elem([1, 2, 1, 3, 1, 1], 0)
            ).shape(),
            &[2, 3]
        );
    }

    #[test]
    fn squeeze_inference_1() {
        let input = TensorFact::default()
            .with_datum_type(DatumType::TDim)
            .with_shape(shapefact![1, 1, (TDim::stream() - 2), 16]);

        let op = Squeeze::<TDim>::new(Some(vec![1]));
        let inferred = op
            .infer_facts(tvec!(input), tvec!(TensorFact::default()))
            .unwrap();

        let expect: TVec<_> = tvec!(
            TensorFact::default()
                .with_datum_type(DatumType::TDim)
                .with_shape(shapefact![1, (TDim::stream() - 2), 16])
        );

        assert_eq!(inferred.1, expect);
    }
}
