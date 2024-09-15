use tract_hir::internal::*;
use tract_ndarray::prelude::*;
use tract_num_traits::Zero;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub mod raw;
pub mod unary;

pub fn space_to_batch_nd(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let datum_type = pb.get_attr_datum_type("T")?;
    Ok(Box::new(raw::SpaceToBatch::new(datum_type)))
}

pub fn batch_to_space_nd(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let datum_type = pb.get_attr_datum_type("T")?;
    Ok(Box::new(raw::BatchToSpace::new(datum_type)))
}

fn space_to_batch<T: Copy + Datum + Zero>(
    input: TValue,
    block_shape: &ArrayView1<i32>,
    paddings: &ArrayView2<i32>,
) -> TractResult<TValue> {
    let mut data = input.into_tensor();

    for (ix, pad) in paddings.view().outer_iter().enumerate() {
        if pad[0] == 0 && pad[1] == 0 {
            continue;
        }
        let mut stack = tvec!();
        let mut pad_shape = data.shape().to_vec();
        if pad[0] != 0 {
            pad_shape[ix + 1] = pad[0] as usize;
            stack.push(Tensor::zero::<T>(&pad_shape)?);
        }
        stack.push(data);
        if pad[1] != 0 {
            pad_shape[ix + 1] = pad[1] as usize;
            stack.push(Tensor::zero::<T>(&pad_shape)?);
        }
        data = Tensor::stack_tensors(ix + 1, &stack)?;
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
    let data = data.into_shape(&reshaped)?;

    let mut permuted_axes: Vec<_> = (0..block_shape.len()).map(|x| 2 * x + 2).collect();
    permuted_axes.push(0);
    permuted_axes.extend((0..block_shape.len()).map(|x| 2 * x + 1));
    permuted_axes.extend((block_shape.len() * 2 + 1)..data.rank());
    let data = data.permute_axes(&permuted_axes)?;
    let data = data.into_shape(&final_shape)?;

    Ok(data.into_tvalue())
}

fn batch_to_space<T: Copy + Datum + Zero>(
    input: TValue,
    block_shape: &ArrayView1<i32>,
    crops: &ArrayView2<i32>,
) -> TractResult<TValue> {
    let data = input.into_tensor().into_array()?;
    let input_shape = data.shape().to_vec();
    let crops: ArrayView2<i32> = crops.view().into_dimensionality()?;

    let block_size = block_shape.iter().map(|a| *a as usize).product::<usize>();

    // block_dim_1 .. block_dim_n, batches/bloc_size, dim_1, .. dim_n, chan_1, .., chan_n
    let mut unflatten_blocked_shape = vec![];
    unflatten_blocked_shape.extend(block_shape.iter().map(|a| *a as usize));
    let batches = data.shape()[0] / block_size;
    unflatten_blocked_shape.push(batches);
    unflatten_blocked_shape.extend(&data.shape()[1..]);
    let data = data.into_shape_with_order(&*unflatten_blocked_shape)?;
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
    let data: Vec<T> = data.iter().copied().collect();
    let data = tract_ndarray::ArrayD::from_shape_vec(padded_shape, data)?;
    let mut data = data;
    for (i, crop) in crops.outer_iter().enumerate() {
        if crop[0] != 0 || crop[1] != 0 {
            let end = data.shape()[1 + i];
            let range = (crop[0] as usize)..(end - crop[1] as usize);
            data = data.slice_axis(Axis(i + 1), range.into()).map(|x| *x).to_owned();
        }
    }
    Ok(data.into_tvalue())
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::raw::{BatchToSpace, SpaceToBatch};
    use super::*;

    // https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
    #[test]
    fn space_to_batch_nd_1() {
        assert_eq!(
            SpaceToBatch::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[[[[1i32], [2]], [[3], [4]]]]).into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[[[[1i32]]], [[[2]]], [[[3]]], [[[4]]]]).into()],
        )
    }

    #[test]
    fn space_to_batch_nd_2() {
        assert_eq!(
            SpaceToBatch::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[[[[1i32, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]],])
                .into(),],
        )
    }

    #[test]
    fn space_to_batch_nd_3() {
        assert_eq!(
            SpaceToBatch::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[[
                        [[1], [2], [3], [4]],
                        [[5], [6], [7], [8]],
                        [[9], [10], [11], [12]],
                        [[13], [14], [15], [16]],
                    ]])
                    .into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[
                [[[1], [3]], [[9], [11]]],
                [[[2], [4]], [[10], [12]]],
                [[[5], [7]], [[13], [15]]],
                [[[6], [8]], [[14], [16]]],
            ])
            .into()],
        )
    }

    #[test]
    fn space_to_batch_nd_4() {
        assert_eq!(
            SpaceToBatch::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[
                        [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
                        [[[9], [10], [11], [12]], [[13], [14], [15], [16]]],
                    ])
                    .into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [2, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[
                [[[0], [1], [3]]],
                [[[0], [9], [11]]],
                [[[0], [2], [4]]],
                [[[0], [10], [12]]],
                [[[0], [5], [7]]],
                [[[0], [13], [15]]],
                [[[0], [6], [8]]],
                [[[0], [14], [16]]],
            ])
            .into(),],
        )
    }

    #[test]
    fn space_to_batch_nd_infer_1() {
        let mut op = SpaceToBatch::new(f32::datum_type());
        let data = f32::fact([1, 4, 16]).into();
        let block_shape = InferenceFact::from(Tensor::from(arr1(&[2])));
        let paddings = InferenceFact::from(Tensor::from(arr2(&[[0.to_dim(), 0.to_dim()]])));
        let any = InferenceFact::default();

        let (_, outputs, _) =
            op.infer_facts(tvec!(&data, &block_shape, &paddings), tvec!(&any), tvec!()).unwrap();

        assert_eq!(outputs[0], f32::fact([2, 2, 16]).into())
    }

    #[test]
    fn space_to_batch_nd_infer_2() {
        let table = SymbolScope::default();
        let s = table.sym("S");
        let mut op = SpaceToBatch::new(f32::datum_type());
        let data = f32::fact(dims!(1, s.to_dim() - 4, 16)).into();
        let block_shape = InferenceFact::from(Tensor::from(arr1(&[2])));
        let paddings = InferenceFact::from(Tensor::from(arr2(&[[0.to_dim(), (s.to_dim() % 2)]])));
        let any = InferenceFact::default();

        let (_, outputs, _) =
            op.infer_facts(tvec!(&data, &block_shape, &paddings), tvec!(&any), tvec!()).unwrap();
        assert_eq!(
            outputs[0],
            f32::fact(dims!(2, (s.to_dim() + s.to_dim() % 2 - 4) / 2, 16)).into()
        );
    }

    #[test]
    fn batch_to_space_nd_1() {
        assert_eq!(
            BatchToSpace::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]).into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[[[[1], [2]], [[3], [4]]]]).into()]
        )
    }

    #[test]
    fn batch_to_space_nd_2() {
        assert_eq!(
            BatchToSpace::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[[[[1i32, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]],])
                        .into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[[[[1i32, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).into()]
        )
    }

    #[test]
    fn batch_to_space_nd_3() {
        assert_eq!(
            BatchToSpace::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[
                        [[[1i32], [3]], [[9], [11]]],
                        [[[2], [4]], [[10], [12]]],
                        [[[5], [7]], [[13], [15]]],
                        [[[6], [8]], [[14], [16]]],
                    ])
                    .into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [0, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[[
                [[1i32], [2], [3], [4]],
                [[5], [6], [7], [8]],
                [[9], [10], [11], [12]],
                [[13], [14], [15], [16]],
            ]])
            .into(),]
        )
    }

    #[test]
    fn batch_to_space_nd_4() {
        assert_eq!(
            BatchToSpace::new(i32::datum_type())
                .eval(tvec![
                    tensor4(&[
                        [[[0i32], [1], [3]]],
                        [[[0], [9], [11]]],
                        [[[0], [2], [4]]],
                        [[[0], [10], [12]]],
                        [[[0], [5], [7]]],
                        [[[0], [13], [15]]],
                        [[[0], [6], [8]]],
                        [[[0], [14], [16]]],
                    ])
                    .into(),
                    tensor1(&[2, 2]).into(),
                    tensor2(&[[0, 0], [2, 0]]).into(),
                ])
                .unwrap(),
            tvec![tensor4(&[
                [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
                [[[9], [10], [11], [12]], [[13], [14], [15], [16]]],
            ])
            .into(),]
        )
    }
}
