use ndarray::prelude::*;

use super::PaddingSpec;
use dim::DimLike;
use tensor::Datum;

#[derive(Debug, Clone)]
pub struct Patch<D: DimLike> {
    pub data_is_nhwc: bool, // default is nchw (onnx)
    pub dilations: Vec<usize>,
    pub kernel_spatial_shape: Vec<D>,
    pub pad_before: Vec<D>,
    pub pad_after: Vec<D>,
    pub strides: Vec<usize>,
    pub input_full_shape: Vec<D>,
    pub output_spatial_shape: Vec<D>,
    pub data_field: Option<Array2<usize>>,
}

impl<D: DimLike> Patch<D> {
    pub fn new(
        data_is_nhwc: bool,
        dilations: Vec<usize>,
        kernel_spatial_shape: Vec<D>,
        padding: &PaddingSpec,
        strides: Vec<usize>,
        input_full_shape: Vec<D>,
    ) -> Patch<D> {
        assert_eq!(input_full_shape.len(), dilations.len() + 2);
        assert_eq!(kernel_spatial_shape.len(), dilations.len());
        assert_eq!(strides.len(), dilations.len());
        let (output_spatial_shape, pad_before, pad_after) = padding.compute(
            &input_full_shape[(1 + (!data_is_nhwc as usize))..][..kernel_spatial_shape.len()],
            &kernel_spatial_shape,
            &*dilations,
            &*strides,
        );
        Patch {
            data_is_nhwc,
            dilations,
            kernel_spatial_shape,
            pad_before,
            pad_after,
            strides,
            input_full_shape,
            output_spatial_shape,
            data_field: None,
        }
    }

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

    pub fn input_batch_size(&self) -> D {
        self.input_full_shape[0]
    }

    pub fn input_channels(&self) -> D {
        self.input_full_shape[self.axis_data_channel()]
    }
}

impl<D: DimLike> Patch<D> {
    pub fn output_full_shape(&self, channels: D) -> Vec<D> {
        let mut v = self.input_full_shape.clone();
        v[self.axis_data_channel()] = channels;
        for i in 0..self.spatial_rank() {
            v[i + self.axis_data_spatial()] = self.output_spatial_shape[i]
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

    pub fn cache_data_field(&mut self) {
        self.data_field = Some(self.mk_data_field())
    }

    pub fn patch_data_iter<'a, 'b, 'c, 'd: 'a, T: Datum>(
        &'b self,
        input: &'a ArrayViewD<'d, T>,
        coords: &'c [usize],
    ) -> PatchIterator<'a, 'b, 'c, 'd, T> {
        PatchIterator {
            patch: &*self,
            item: 0,
            input,
            coords,
            full_coords: vec![0; coords.len()],
        }
    }
}

pub struct PatchIterator<'a, 'b, 'c, 'd: 'a, T: Datum> {
    input: &'a ArrayViewD<'d, T>,
    patch: &'b Patch<usize>,
    item: usize,
    coords: &'c [usize],
    full_coords: Vec<usize>,
}

impl<'a, 'b, 'c, 'd, T: Datum> Iterator for PatchIterator<'a, 'b, 'c, 'd, T> {
    type Item = Option<T>;
    fn next(&mut self) -> Option<Option<T>> {
        if self.item == self.patch.data_field.as_ref().unwrap().rows() {
            return None;
        }
        let img_offset = self.patch.data_field.as_ref().unwrap().row(self.item);
        self.item += 1;

        (&mut *self.full_coords).copy_from_slice(self.coords);
        self.full_coords
            .iter_mut()
            .skip(self.patch.axis_data_spatial())
            .zip(img_offset.iter().zip(self.patch.strides.iter()))
            .for_each(|(x, (&i, &s))| *x = (*x * s).wrapping_add(i));
        Some(self.input.get(&*self.full_coords).cloned())
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
            &PaddingSpec::Explicit(vec![pad_before], vec![bad_after]),
            vec![stride],
            vec![1, 1, input],
        );
        patch.output_spatial_shape[0]
    }

    #[test]
    fn basic() {
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 0, 0, 1), 3);
    }

    #[test]
    fn strides() {
        assert_eq!(compute_output_spatial_dim(7, 1, 3, 0, 0, 2), 3);
    }

    #[test]
    fn padding() {
        assert_eq!(compute_output_spatial_dim(5, 1, 3, 1, 1, 1), 5);
    }

    #[test]
    fn strides_and_padding() {
        assert_eq!(compute_output_spatial_dim(7, 1, 3, 1, 1, 2), 4);
    }

    fn field(kdim: &[usize], dilations: &[usize]) -> Array2<usize> {
        let patch = Patch::new(
            false,
            dilations.to_vec(),
            kdim.to_vec(),
            &PaddingSpec::Explicit(vec![0; kdim.len()], vec![0; kdim.len()]),
            vec![1; kdim.len()],
            vec![10; kdim.len() + 2],
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
