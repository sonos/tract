use tfdeploy::analyser::rules::prelude::*;
use tfdeploy::ops::prelude::*;
use ndarray::prelude::*;

pub fn space_to_batch_nd(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let datum_type = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(SpaceToBatch(datum_type)()))
}
pub fn batch_to_space_nd(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let datum_type = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(BatchToSpace(datum_type)()))
}

#[derive(Debug, Clone)]
struct SpaceToBatchBuffer<T: Datum> {
    inited: bool,
    buffer: Vec<ArrayD<T>>,
}
impl<T: Datum> OpBuffer for SpaceToBatchBuffer<T> {}

#[derive(Debug, Clone, new)]
pub struct SpaceToBatch<T: Datum>(PhantomData<T>);

impl<T: Datum> Op for SpaceToBatch<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> Result<TVec<Value>> {
        let (input, block_shape, paddings) = args_3!(inputs);
        let block_shape = block_shape.as_i32s().ok_or("block shape expected as I32")?;
        let paddings = paddings.cast_to_array::<i32>()?;
        let mut data = input.into_array::<T>()?;

        for (ix, pad) in paddings.view().outer_iter().enumerate() {
            if pad[0] != 0 {
                let mut pad_shape = data.shape().to_vec();
                pad_shape[ix + 1] = pad[0] as usize;
                let tmp = ::ndarray::stack(
                    ::ndarray::Axis(ix + 1),
                    &[::ndarray::ArrayD::zeros(pad_shape).view(), data.view()],
                )?;
                data = tmp;
            }
            if pad[1] != 0 {
                let mut pad_shape = data.shape().to_vec();
                pad_shape[ix + 1] = pad[1] as usize;
                let tmp = ::ndarray::stack(
                    ::ndarray::Axis(ix + 1),
                    &[data.view(), ::ndarray::ArrayD::zeros(pad_shape).view()],
                )?;
                data = tmp;
            }
        }
        let mut reshaped = vec![data.shape()[0]];
        let block_size = block_shape.iter().map(|a| *a as usize).product::<usize>();
        let mut final_shape = vec![block_size * data.shape()[0]];
        for (m, &block_shape_dim) in block_shape.iter().enumerate() {
            reshaped.push(data.shape()[m + 1] / block_shape_dim as usize);
            reshaped.push(block_shape_dim as usize);
            final_shape.push(data.shape()[m + 1] / block_shape_dim as usize);
        }
        reshaped.extend(&data.shape()[block_shape.len() + 1..]);
        final_shape.extend(&data.shape()[block_shape.len() + 1..]);
        let data = data.into_shape(reshaped)?;

        let mut permuted_axes: Vec<_> = (0..block_shape.len()).map(|x| 2 * x + 2).collect();
        permuted_axes.push(0);
        permuted_axes.extend((0..block_shape.len()).map(|x| 2 * x + 1));
        permuted_axes.extend((block_shape.len() * 2 + 1)..data.ndim());
        let data = data.permuted_axes(permuted_axes);
        let data: Vec<T> = data.into_iter().map(|x| *x).collect();
        let data = ::ndarray::ArrayD::from_shape_vec(final_shape, data)?;

        Ok(tvec![data.into()])
    }

    /// Returns a new streaming buffer for the operation.
    fn new_buffer(&self) -> Box<OpBuffer> {
        Box::new(SpaceToBatchBuffer::<T> {
            inited: false,
            buffer: vec![],
        })
    }

    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<TVec<Value>>> {
        let (input, block_shape, paddings) = args_3!(inputs);
        let block_shape = block_shape
            .into_const()
            .ok_or("Expected block_shape to be const")?;
        let block_shape = block_shape
            .into_tensor()
            .take_i32s()
            .ok_or("Expected block_shape to be i32s")?;
        let block_shape: Array1<i32> = block_shape.into_dimensionality()?;

        let paddings = paddings
            .into_const()
            .ok_or("Expected paddings to be const")?;
        let casted_paddings = TDim::tensor_cast_to_array(&paddings)?;
        let mut paddings = casted_paddings.view().into_dimensionality()?.to_owned();

        let Stream { info, chunk, .. } =
            input.into_stream().ok_or("Expected input to be a stream")?;
        let data = if let Some(data) = chunk {
            data
        } else {
            return Ok(None);
        };
        if data.shape()[info.axis] != 1 {
            bail!("Expected streaming dim to be 1")
        }
        let buffer = buffer
            .downcast_mut::<SpaceToBatchBuffer<T>>()
            .ok_or("The buffer can't be downcasted to Buffer<T>.")?;
        if !buffer.inited {
            for _ in 0..paddings[(info.axis - 1, 0)].to_integer()? {
                buffer.buffer.push(Array::zeros(data.shape()));
            }
            buffer.inited = true;
        };
        buffer.buffer.push(data.into_array()?);
        if buffer.buffer.len() < block_shape[info.axis - 1] as usize {
            return Ok(None);
        }
        paddings[(info.axis - 1, 0)] = 0.to_dim();
        paddings[(info.axis - 1, 1)] = 0.to_dim();
        let mut shape = buffer.buffer[0].shape().to_vec();
        shape[info.axis] = block_shape[info.axis - 1] as usize;
        let data = Array::from_shape_fn(shape, |mut coords| -> T {
            let buf = &buffer.buffer[coords[info.axis]];
            coords[info.axis] = 0;
            buf[coords]
        });
        buffer.buffer.clear();
        Ok(Some(self.eval(tvec!(
            data.into(),
            block_shape.into(),
            paddings.into()
        ))?))
    }
}

impl<T: Datum> InferenceRulesOp for SpaceToBatch<T> {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver.equals(&inputs.len, 3).equals(&outputs.len, 1);
        rules(solver, &outputs[0], &inputs[0], &inputs[1], &inputs[2]);
    }
}

fn rules<'r, 'p: 'r>(
    solver: &mut Solver<'r>,
    batch: &'p TensorProxy,
    space: &'p TensorProxy,
    block_shape: &'p TensorProxy,
    paddings: &'p TensorProxy,
) {
    solver
        .equals(&batch.datum_type, &space.datum_type)
        .equals(&block_shape.datum_type, DatumType::I32)
        .equals(&batch.rank, &space.rank)
        .equals(&block_shape.rank, 1)
        .equals(&paddings.rank, 2)
        .equals(&block_shape.shape[0], &paddings.shape[0])
        .given(&block_shape.value, move |solver, block_shape: Tensor| {
            let block_shape: ArrayD<i32> = block_shape.take_i32s().unwrap();
            let block_shape_prod = block_shape.iter().map(|s| *s as usize).product::<usize>();
            solver.equals(
                &batch.shape[0],
                (block_shape_prod as isize) * space.shape[0].bex(),
            );
            solver.given(&paddings.value, move |solver, paddings: Tensor| {
                let paddings = TDim::tensor_cast_to_array(&paddings).unwrap(); // FIXMEa
                let paddings = paddings.view().into_dimensionality().unwrap();
                for d in 0..block_shape.len() {
                    solver.equals(
                        space.shape[1 + d].bex() + paddings[(d, 0)] + paddings[(d, 1)],
                        (block_shape[d] as isize) * batch.shape[1 + d].bex(),
                    );
                }
            });
        })
        .given(&block_shape.value, move |solver, block_shape: Tensor| {
            let block_shape: ArrayD<i32> = block_shape.take_i32s().unwrap();
            solver.given(&space.rank, move |solver, rank: isize| {
                for d in block_shape.len() + 1..(rank as usize) {
                    solver.equals(&space.shape[d], &batch.shape[d]);
                }
            });
        });
}

#[derive(Debug, Clone)]
struct BatchToSpaceBuffer {
    cropped: usize,
}
impl OpBuffer for BatchToSpaceBuffer {}

#[derive(Debug, Clone, new)]
pub struct BatchToSpace<T: Datum>(PhantomData<T>);

impl<T: Datum> Op for BatchToSpace<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> Result<TVec<Value>> {
        use ndarray::*;
        let (input, block_shape, crops) = args_3!(inputs);
        let block_shape = block_shape.as_i32s().ok_or("block shape expected as I32")?;
        let crops = crops.cast_to_array::<i32>()?;
        let data = input.into_array()?;
        let input_shape = data.shape().to_vec();
        let crops: ArrayView2<i32> = crops.view().into_dimensionality()?;

        let block_size = block_shape.iter().map(|a| *a as usize).product::<usize>();

        // block_dim_1 .. block_dim_n, batches/bloc_size, dim_1, .. dim_n, chan_1, .., chan_n
        let mut unflatten_blocked_shape = vec![];
        unflatten_blocked_shape.extend(block_shape.iter().map(|a| *a as usize));
        let batches = data.shape()[0] / block_size;
        unflatten_blocked_shape.push(batches);
        unflatten_blocked_shape.extend(&data.shape()[1..]);
        let data = data.into_shape(&*unflatten_blocked_shape)?;
        let mut permuted_axes = vec![block_shape.len()];
        let mut padded_shape = vec![batches];
        for i in 0..block_shape.shape()[0] {
            permuted_axes.push(block_shape.len() + 1 + i);
            permuted_axes.push(i);
            padded_shape.push(block_shape[i] as usize * input_shape[i + 1]);
        }
        permuted_axes.extend((1 + block_shape.len() * 2)..data.ndim());
        padded_shape.extend(&input_shape[1 + block_shape.len()..]);
        let data = data.permuted_axes(permuted_axes);
        let data: Vec<T> = data.into_iter().map(|x| *x).collect();
        let data = ::ndarray::ArrayD::from_shape_vec(padded_shape, data)?;
        let mut data = data;
        for (i, crop) in crops.outer_iter().enumerate() {
            if crop[0] != 0 || crop[1] != 0 {
                let end = data.shape()[1 + i] as usize;
                let range = (crop[0] as usize)..(end - crop[1] as usize);
                data = data
                    .slice_axis(Axis(i + 1), range.into())
                    .map(|x| *x)
                    .to_owned();
            }
        }
        Ok(tvec![data.into()])
    }

    /// Returns a new streaming buffer for the operation.
    fn new_buffer(&self) -> Box<OpBuffer> {
        Box::new(BatchToSpaceBuffer { cropped: 0 })
    }

    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<TVec<Value>>> {
        let (input, block_shape, crops) = args_3!(inputs);
        let block_shape = block_shape
            .into_const()
            .ok_or("Expected block_shape to be const")?;
        let block_shape = block_shape
            .into_tensor()
            .take_i32s()
            .ok_or("Expected block_shape to be i32s")?;
        let block_shape: Array1<i32> = block_shape.into_dimensionality()?;

        let crops = crops.into_const().ok_or("Expected crops to be const")?;
        let casted_crops = TDim::tensor_cast_to_array(&crops)?;
        let mut crops = casted_crops.view().into_dimensionality()?.to_owned();

        let Stream { info, chunk, .. } =
            input.into_stream().ok_or("Expected input to be a stream")?;
        let data = if let Some(data) = chunk {
            data
        } else {
            return Ok(None);
        };
        if data.shape()[info.axis] != 1 {
            bail!("Expected streaming dim to be 1")
        }
        let buffer = buffer
            .downcast_mut::<BatchToSpaceBuffer>()
            .ok_or("The buffer can't be downcasted to Buffer<T>.")?;
        if buffer.cropped < crops[(info.axis - 1, 0)].to_integer()? as usize {
            buffer.cropped += 1;
            return Ok(None);
        }
        crops[(info.axis - 1, 0)] = 0.to_dim();
        crops[(info.axis - 1, 1)] = 0.to_dim();
        Ok(Some(self.eval(tvec!(
            data.into(),
            Tensor::from(block_shape).into(),
            Tensor::from(crops).into()
        ))?))
    }
}

impl<T: Datum> InferenceRulesOp for BatchToSpace<T> {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver.equals(&inputs.len, 3).equals(&outputs.len, 1);
        rules(solver, &inputs[0], &outputs[0], &inputs[1], &inputs[2]);
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use tfdeploy::ops::arr4;
    use tfdeploy::ops::InferenceOp;

    // https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
    #[test]
    fn space_to_batch_nd_1() {
        assert_eq!(
            SpaceToBatch::<i32>::new()
                .eval(tvec![
                    arr4(&[[[[1], [2]], [[3], [4]]]]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![arr4(&[[[[1i32]]], [[[2]]], [[[3]]], [[[4]]]]).into()],
        )
    }

    #[test]
    fn space_to_batch_nd_2() {
        assert_eq!(
            SpaceToBatch::<i32>::new()
                .eval(tvec![
                    arr4(&[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![
                arr4(&[
                    [[[1i32, 2, 3]]],
                    [[[4, 5, 6]]],
                    [[[7, 8, 9]]],
                    [[[10, 11, 12]]],
                ]).into(),
            ],
        )
    }

    #[test]
    fn space_to_batch_nd_3() {
        assert_eq!(
            SpaceToBatch::<i32>::new()
                .eval(tvec![
                    arr4(&[[
                        [[1], [2], [3], [4]],
                        [[5], [6], [7], [8]],
                        [[9], [10], [11], [12]],
                        [[13], [14], [15], [16]],
                    ]]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![
                arr4(&[
                    [[[1], [3]], [[9], [11]]],
                    [[[2], [4]], [[10], [12]]],
                    [[[5], [7]], [[13], [15]]],
                    [[[6], [8]], [[14], [16]]],
                ]).into(),
            ],
        )
    }

    #[test]
    fn space_to_batch_nd_4() {
        assert_eq!(
            SpaceToBatch::<i32>::new()
                .eval(tvec![
                    arr4(&[
                        [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
                        [[[9], [10], [11], [12]], [[13], [14], [15], [16]]],
                    ]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [2, 0]]).into(),
                ])
                .unwrap(),
            tvec![
                arr4(&[
                    [[[0], [1], [3]]],
                    [[[0], [9], [11]]],
                    [[[0], [2], [4]]],
                    [[[0], [10], [12]]],
                    [[[0], [5], [7]]],
                    [[[0], [13], [15]]],
                    [[[0], [6], [8]]],
                    [[[0], [14], [16]]],
                ]).into(),
            ],
        )
    }

    #[test]
    fn space_to_batch_nd_infer_1() {
        let op = SpaceToBatch::<f32>::new();
        let data = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 4, 16));
        let block_shape = TensorFact::from(Tensor::from(arr1(&[2])));
        let paddings = TensorFact::from(Tensor::from(arr2(&[[0.to_dim(), 0.to_dim()]])));

        let (_, outputs) =
            op.infer(
                tvec!(data, block_shape, paddings),
                tvec!(TensorFact::default()),
            ).unwrap();

        assert_eq!(
            outputs[0],
            TensorFact::dt_shape(DatumType::F32, shapefact!(2, 2, 16))
        );
    }

    #[test]
    fn space_to_batch_nd_infer_2() {
        let op = SpaceToBatch::<f32>::new();
        let data = TensorFact::dt_shape(DatumType::F32, shapefact!(1, (TDim::s() - 4), 16));
        let block_shape = TensorFact::from(Tensor::from(arr1(&[2])));
        let paddings = TensorFact::from(Tensor::from(arr2(&[[0.to_dim(), (TDim::s() % 2)]])));

        let (_, mut outputs) =
            op.infer(
                tvec!(data, block_shape, paddings),
                tvec!(TensorFact::default()),
            ).unwrap();
        println!("raw: {:?}", outputs[0]);
        outputs[0].reduce();
        println!("reduced: {:?}", outputs[0]);
        assert_eq!(
            outputs[0],
            TensorFact::dt_shape(
                DatumType::F32,
                shapefact!(2, ((TDim::s() + TDim::s() % 2 - 4) / 2), 16)
            )
        );
    }

    #[test]
    fn batch_to_space_nd_1() {
        assert_eq!(
            BatchToSpace::<i32>::new()
                .eval(tvec![
                    arr4(&[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![arr4(&[[[[1], [2]], [[3], [4]]]]).into()]
        )
    }

    #[test]
    fn batch_to_space_nd_2() {
        assert_eq!(
            BatchToSpace::<i32>::new()
                .eval(tvec![
                    arr4(&[
                        [[[1i32, 2, 3]]],
                        [[[4, 5, 6]]],
                        [[[7, 8, 9]]],
                        [[[10, 11, 12]]],
                    ]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![arr4(&[[[[1i32, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).into()]
        )
    }

    #[test]
    fn batch_to_space_nd_3() {
        assert_eq!(
            BatchToSpace::<i32>::new()
                .eval(tvec![
                    arr4(&[
                        [[[1i32], [3]], [[9], [11]]],
                        [[[2], [4]], [[10], [12]]],
                        [[[5], [7]], [[13], [15]]],
                        [[[6], [8]], [[14], [16]]],
                    ]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![
                arr4(&[[
                    [[1i32], [2], [3], [4]],
                    [[5], [6], [7], [8]],
                    [[9], [10], [11], [12]],
                    [[13], [14], [15], [16]],
                ]]).into(),
            ]
        )
    }

    #[test]
    fn batch_to_space_nd_4() {
        assert_eq!(
            BatchToSpace::<i32>::new()
                .eval(tvec![
                    arr4(&[
                        [[[0i32], [1], [3]]],
                        [[[0], [9], [11]]],
                        [[[0], [2], [4]]],
                        [[[0], [10], [12]]],
                        [[[0], [5], [7]]],
                        [[[0], [13], [15]]],
                        [[[0], [6], [8]]],
                        [[[0], [14], [16]]],
                    ]).into(),
                    arr1(&[2, 2]).into(),
                    arr2(&[[0, 0], [2, 0]]).into(),
                ])
                .unwrap(),
            tvec![
                arr4(&[
                    [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
                    [[[9], [10], [11], [12]], [[13], [14], [15], [16]]],
                ]).into(),
            ]
        )
    }
}
