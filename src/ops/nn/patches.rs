use ndarray::prelude::*;

use dim::DimLike;

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
    ) -> PaddedGeometry<D> {
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
    ) -> PaddedGeometry<D> {
        let spatial_rank = data_spatial_shape.len();
        assert_eq!(spatial_rank, kernel_spatial_shape.len());
        assert_eq!(spatial_rank, dilations.len());
        assert_eq!(spatial_rank, strides.len());
        assert_eq!(spatial_rank, aft.len());
        assert_eq!(spatial_rank, bef.len());
        let output_spatial_shape = (0..spatial_rank)
            .map(|ax| {
                let kernel_field = (kernel_spatial_shape[ax] - 1) * dilations[ax] + 1;
                let dim = (data_spatial_shape[ax] + bef[ax] + aft[ax] - kernel_field + 1).div_ceil(strides[ax]);
                dim
            }).collect();
        PaddedGeometry {
            output_spatial_shape,
            pad_before: bef.iter().map(|&x| D::from(x)).collect(),
            pad_after: aft.iter().map(|&x| D::from(x)).collect(),
        }
    }

    fn same<D: DimLike>(
        &self,
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[D],
        dilations: &[usize],
        strides: &[usize],
        upper: bool,
    ) -> PaddedGeometry<D> {
        let spatial_rank = data_spatial_shape.len();
        let mut result = PaddedGeometry::default();
        for ax in 0..spatial_rank {
            let dim = data_spatial_shape[ax].div_ceil(strides[ax]);
            let kernel_field = (kernel_spatial_shape[ax] - 1) * dilations[ax] + 1;
            result.output_spatial_shape.push(dim);
            let pad = (dim - 1) * strides[ax] + kernel_field - data_spatial_shape[ax];
            let lower_pad = pad / 2;
            let higher_pad = pad - pad / 2;
            if upper {
                result.pad_before.push(lower_pad);
                result.pad_after.push(higher_pad);
            } else {
                result.pad_after.push(lower_pad);
                result.pad_before.push(higher_pad);
            }
        }
        result
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct PaddedGeometry<D: DimLike> {
    pub output_spatial_shape: Vec<D>,
    pub pad_before: Vec<D>,
    pub pad_after: Vec<D>,
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
}

impl<D: DimLike> Patch<D> {
    pub fn new(
        data_is_nhwc: bool,
        dilations: Vec<usize>,
        kernel_spatial_shape: Vec<D>,
        pad_before: Vec<D>,
        pad_after: Vec<D>,
        strides: Vec<usize>,
        input_full_shape: Vec<D>,
        output_spatial_shape: Vec<D>) -> Patch<D> {
        Patch {
            data_is_nhwc,
            dilations,
            kernel_spatial_shape,
            pad_before,
            pad_after,
            strides,
            input_full_shape,
            output_spatial_shape,
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
