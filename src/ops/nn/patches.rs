use ndarray::prelude::*;

use dim::DimLike;

#[derive(Debug, new)]
pub struct DataCoords<'a> {
    pub n: usize,
    pub chan: usize,
    pub space: &'a [usize],
}

#[derive(Debug, Clone, new)]
pub struct Patch<D: DimLike> {
    pub data_is_nhwc: bool, // default is nchw (onnx)
    pub dilations: Vec<usize>,
    pub kernel_spatial_shape: Vec<D>,
    pub pad_before: Vec<D>,
    pub pad_after: Vec<D>,
    pub strides: Vec<usize>,
    pub data_full_shape: Vec<D>,
}

impl<D: DimLike> Patch<D> {
    pub fn spatial_rank(&self) -> usize {
        self.kernel_spatial_shape.len()
    }

    pub fn axis_data_batch(&self) -> usize {
        0
    }

    pub fn axis_data_spatial(&self) -> usize {
        if self.data_is_nhwc {
            1
        } else {
            2
        }
    }

    pub fn axis_data_channel(&self) -> usize {
        if self.data_is_nhwc {
            1 + self.spatial_rank()
        } else {
            1
        }
    }

    pub fn split_data_coords<'a>(&self, coords: &'a [usize]) -> DataCoords<'a> {
        if self.data_is_nhwc {
            DataCoords::new(
                coords[0],
                coords[self.spatial_rank() + 1],
                &coords[1..self.spatial_rank() + 1],
            )
        } else {
            DataCoords::new(coords[0], coords[1], &coords[2..self.spatial_rank() + 2]) // nchw
        }
    }
}

impl<D: DimLike> Patch<D> {
    pub fn out_spatial_dim(&self, spatial_axis: usize) -> D {
        let one = D::one();
        let field =
            (self.kernel_spatial_shape[spatial_axis] - one) * self.dilations[spatial_axis] + one;
        let pad = self.pad_before[spatial_axis] + self.pad_after[spatial_axis];
        let input_spatial_dim =
            self.data_full_shape[spatial_axis + self.axis_data_spatial()] + pad - field;
        input_spatial_dim / self.strides[spatial_axis] + one
    }

    pub fn output_full_shape(&self, channels: D) -> Vec<D> {
        let mut v = self.data_full_shape.clone();
        v[self.axis_data_channel()] = channels;
        for i in 0..self.spatial_rank() {
            v[i + self.axis_data_spatial()] = self.out_spatial_dim(i);
        }
        v
    }
}

impl Patch<usize> {
    pub fn mk_kernel_field(&self) -> Array2<usize> {
        let shape: Vec<usize> = self
            .kernel_spatial_shape
            .iter()
            .map(|&a| a as usize)
            .collect();
        let square = ArrayD::from_shape_fn(&*shape, |id| id.slice().to_vec());
        let len = square.len();
        let points: Array1<Vec<usize>> = square.into_shape((len,)).unwrap();
        Array2::from_shape_fn((points.len(), self.spatial_rank()), |(pt, axis)| {
            points[pt][axis]
        })
    }

    pub fn mk_data_field(&self) -> Array2<usize> {
        let mut field = self.mk_kernel_field();
        ::ndarray::Zip::from(&mut field)
            .and_broadcast(&arr1(&*self.dilations))
            .and_broadcast(&arr1(&*self.pad_before))
            .apply(|offset, &dil, &pad| *offset = (*offset * dil).wrapping_sub(pad));
        field
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn compute_output_spatial_dim(
        input: usize,
        dilation: usize,
        kdim: usize,
        pad_before: usize,
        bad_after: usize,
        stride: usize,
    ) -> usize {
        let patch = Patch::new(
            false,
            vec![dilation],
            vec![kdim],
            vec![pad_before],
            vec![bad_after],
            vec![stride],
            vec![1, 1, input],
        );
        patch.out_spatial_dim(0)
    }

    #[test]
    fn test_output_spatial_dim() {
        // onnx test_basic_conv_without_padding
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 0, 0, 1), 3);

        // onnx test_conv_with_strides_no_padding
        assert_eq!(compute_output_spatial_dim(7, 1, 3, 0, 0, 2), 3);
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 0, 0, 2), 2);

        // onnx test_conv_with_strides_padding
        assert_eq!(compute_output_spatial_dim(7, 1, 3, 1, 1, 2), 4);
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 1, 1, 2), 3);

        // onnx test_conv_with_strides_and_asymmetric_padding
        assert_eq!(compute_output_spatial_dim(7, 1, 3, 1, 1, 2), 4);
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 0, 0, 2), 2);
    }

    fn field(kdim: &[usize], dilations: &[usize]) -> Array2<usize> {
        let patch = Patch::new(
            false,
            dilations.to_vec(),
            kdim.to_vec(),
            vec![0; kdim.len()],
            vec![0; kdim.len()],
            vec![0; kdim.len()],
            vec![0; kdim.len()],
        );
        patch.mk_data_field()
    }

    #[test]
    fn test_field() {
        assert_eq!(field(&[3], &[1]), arr2(&[[0], [1], [2]]));
        assert_eq!(field(&[3], &[2]), arr2(&[[0], [2], [4]]));
        assert_eq!(
            field(&[2, 2], &[1, 1]),
            arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]])
        );
        assert_eq!(
            field(&[2, 2], &[2, 1]),
            arr2(&[[0, 0], [0, 1], [2, 0], [2, 1]])
        );
    }
}
