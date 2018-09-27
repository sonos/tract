use ndarray::prelude::*;

use dim::DimLike;
use tensor::Datum;

#[derive(Debug, Clone)]
pub enum PaddingSpec {
    Explicit(Vec<usize>, Vec<usize>),
    Valid,
    SameUpper,
    SameLower,
}

impl Default for PaddingSpec {
    fn default() -> PaddingSpec {
        PaddingSpec::Valid
    }
}

impl PaddingSpec {
    pub fn compute<D: DimLike>(
        &self,
        input_spatial_shape: &[D],
        kernel_spatial_shape: &[D],
        dilations: &[usize],
        strides: &[usize],
    ) -> (Vec<D>, Vec<D>, Vec<D>) {
        assert_eq!(dilations.len(), strides.len());
        assert_eq!(dilations.len(), input_spatial_shape.len());
        assert_eq!(dilations.len(), kernel_spatial_shape.len());
        match self {
            PaddingSpec::Valid => Self::explicit(
                input_spatial_shape,
                kernel_spatial_shape,
                dilations,
                strides,
                &*vec![0; kernel_spatial_shape.len()],
                &*vec![0; kernel_spatial_shape.len()],
            ),
            PaddingSpec::Explicit(ref bef, ref aft) => Self::explicit(
                input_spatial_shape,
                kernel_spatial_shape,
                dilations,
                strides,
                bef,
                aft,
            ),
            PaddingSpec::SameUpper => self.same(
                input_spatial_shape,
                kernel_spatial_shape,
                dilations,
                strides,
                true,
            ),
            PaddingSpec::SameLower => self.same(
                input_spatial_shape,
                kernel_spatial_shape,
                dilations,
                strides,
                false,
            ),
        }
    }

    fn explicit<D: DimLike>(
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[D],
        dilations: &[usize],
        strides: &[usize],
        bef: &[usize],
        aft: &[usize],
    ) -> (Vec<D>, Vec<D>, Vec<D>) {
        let spatial_rank = data_spatial_shape.len();
        assert_eq!(spatial_rank, kernel_spatial_shape.len());
        assert_eq!(spatial_rank, dilations.len());
        assert_eq!(spatial_rank, strides.len());
        assert_eq!(spatial_rank, aft.len());
        assert_eq!(spatial_rank, bef.len());
        let output_spatial_shape = (0..spatial_rank)
            .map(|ax| {
                let kernel_field = (kernel_spatial_shape[ax] - 1) * dilations[ax] + 1;
                let dim = (data_spatial_shape[ax] + bef[ax] + aft[ax] - kernel_field + 1)
                    .div_ceil(strides[ax]);
                dim
            }).collect();
        (
            output_spatial_shape,
            bef.iter().map(|&x| D::from(x)).collect(),
            aft.iter().map(|&x| D::from(x)).collect(),
        )
    }

    fn same<D: DimLike>(
        &self,
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[D],
        dilations: &[usize],
        strides: &[usize],
        upper: bool,
    ) -> (Vec<D>, Vec<D>, Vec<D>) {
        let spatial_rank = data_spatial_shape.len();
        let mut dims = vec![];
        let mut pad_before = vec![];
        let mut pad_after = vec![];
        for ax in 0..spatial_rank {
            let dim = data_spatial_shape[ax].div_ceil(strides[ax]);
            let kernel_field = (kernel_spatial_shape[ax] - 1) * dilations[ax] + 1;
            dims.push(dim);
            let pad = (dim - 1) * strides[ax] + kernel_field - data_spatial_shape[ax];
            let lower_pad = pad / 2;
            let higher_pad = pad - pad / 2;
            if upper {
                pad_before.push(lower_pad);
                pad_after.push(higher_pad);
            } else {
                pad_after.push(lower_pad);
                pad_before.push(higher_pad);
            }
        }
        (dims, pad_before, pad_after)
    }
}

#[derive(Debug, new)]
pub struct DataCoords<'a> {
    pub n: usize,
    pub chan: usize,
    pub space: &'a [usize],
}

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

    pub fn patch_data_iter<'a, 'b, 'c, T: Datum>(
        &'b mut self,
        input: &'a ArrayViewD<'a, T>,
        coords: &'c [usize],
    ) -> PatchIterator<'a, 'b, 'c, T> {
        if self.data_field.is_none() {
            self.data_field = Some(self.mk_data_field());
        }
        PatchIterator {
            patch: &*self,
            item: 0,
            input,
            coords,
        }
    }
}

pub struct PatchIterator<'a, 'b, 'c, T: Datum> {
    input: &'a ArrayViewD<'a, T>,
    patch: &'b Patch<usize>,
    item: usize,
    coords: &'c [usize],
}

impl<'a, 'b, 'c, T: Datum> Iterator for PatchIterator<'a, 'b, 'c, T> {
    type Item = Option<T>;
    fn next(&mut self) -> Option<Option<T>> {
        if self.item == self.patch.data_field.as_ref().unwrap().rows() {
            return None;
        }
        let splitted = self.patch.split_data_coords(self.coords);
        let img_offset = self.patch.data_field.as_ref().unwrap().row(self.item);
        self.item += 1;
        let i_coords: Vec<usize> = izip!(
            splitted.space.iter(),
            img_offset.iter(),
            self.patch.strides.iter()
        ).map(|(x, i, s)| (x * s).wrapping_add(*i))
        .collect();
        Some(
            self.input
            .subview(Axis(self.patch.axis_data_channel()), splitted.chan)
            .subview(Axis(self.patch.axis_data_batch()), splitted.n) // careful, need to start with higher ranking
            .get(&*i_coords)
            .cloned(),
        )
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
