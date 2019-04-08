use crate::internal::*;

#[derive(Debug, Clone)]
pub enum PaddingSpec {
    Explicit(TVec<usize>, TVec<usize>),
    Valid,
    SameUpper,
    SameLower,
}

impl Default for PaddingSpec {
    fn default() -> PaddingSpec {
        PaddingSpec::Valid
    }
}

#[derive(Debug, Clone)]
pub struct ComputedPaddedDim<D: DimLike> {
    pub pad_before: TVec<D>,
    pub pad_after: TVec<D>,
    pub output: TVec<D>,
}

impl PaddingSpec {
    pub fn valid_dim(&self, d: usize) -> bool {
        match self {
            PaddingSpec::Valid => true,
            PaddingSpec::Explicit(a, b) => a[d] == 0 && b[d] == 0,
            _ => false,
        }
    }

    pub fn rm_axis(&self, d: usize) -> PaddingSpec {
        match self {
            PaddingSpec::Explicit(a, b) => {
                let mut a = a.clone();
                let mut b = b.clone();
                a.remove(d);
                b.remove(d);
                PaddingSpec::Explicit(a, b)
            }
            _ => self.clone(),
        }
    }

    pub fn compute<D: DimLike>(
        &self,
        input_spatial_shape: &[D],
        kernel_spatial_shape: &[usize],
        dilations: &[usize],
        strides: &[usize],
    ) -> ComputedPaddedDim<D> {
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
            PaddingSpec::SameUpper => {
                Self::same(input_spatial_shape, kernel_spatial_shape, dilations, strides, true)
            }
            PaddingSpec::SameLower => {
                Self::same(input_spatial_shape, kernel_spatial_shape, dilations, strides, false)
            }
        }
    }

    fn explicit<D: DimLike>(
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[usize],
        dilations: &[usize],
        strides: &[usize],
        bef: &[usize],
        aft: &[usize],
    ) -> ComputedPaddedDim<D> {
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
            })
            .collect();
        ComputedPaddedDim {
            output: output_spatial_shape,
            pad_before: bef.iter().map(|&x| D::from(x)).collect(),
            pad_after: aft.iter().map(|&x| D::from(x)).collect(),
        }
    }

    fn same<D: DimLike>(
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[usize],
        dilations: &[usize],
        strides: &[usize],
        upper: bool,
    ) -> ComputedPaddedDim<D> {
        let spatial_rank = data_spatial_shape.len();
        let mut dims = tvec![];
        let mut pad_before = tvec![];
        let mut pad_after = tvec![];
        for ax in 0..spatial_rank {
            let (d, b, a) = Self::same_one(
                data_spatial_shape[ax],
                kernel_spatial_shape[ax],
                dilations[ax],
                strides[ax],
                upper,
            );
            dims.push(d);
            pad_before.push(b);
            pad_after.push(a);
        }
        ComputedPaddedDim { pad_before, pad_after, output: dims }
    }

    fn same_one<D: DimLike>(
        data_spatial_dim: D,
        kernel_spatial_dim: usize,
        dilation: usize,
        stride: usize,
        upper: bool,
    ) -> (D, D, D) {
        let dim = data_spatial_dim.div_ceil(stride);
        let kernel_field = (kernel_spatial_dim - 1) * dilation + 1;
        let pad = if stride <= kernel_field {
            (dim - 1) * stride + kernel_field - data_spatial_dim
        } else {
            D::zero()
        };
        let lower_pad = pad / 2;
        let higher_pad = pad - pad / 2;
        if upper {
            (dim, lower_pad, higher_pad)
        } else {
            (dim, higher_pad, lower_pad)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_stride_1() {
        assert_eq!(PaddingSpec::same_one(1usize, 2usize, 1, 1, true), (1, 0, 1));
        assert_eq!(PaddingSpec::same_one(2usize, 2usize, 1, 1, true), (2, 0, 1));
        assert_eq!(PaddingSpec::same_one(3usize, 2usize, 1, 1, true), (3, 0, 1));
        assert_eq!(PaddingSpec::same_one(4usize, 2usize, 1, 1, true), (4, 0, 1));
    }

    #[test]
    fn same_stride_2() {
        assert_eq!(PaddingSpec::same_one(1usize, 2usize, 1, 2, true), (1, 0, 1));
        assert_eq!(PaddingSpec::same_one(2usize, 2usize, 1, 2, true), (1, 0, 0));
        assert_eq!(PaddingSpec::same_one(3usize, 2usize, 1, 2, true), (2, 0, 1));
        assert_eq!(PaddingSpec::same_one(4usize, 2usize, 1, 2, true), (2, 0, 0));
    }

    #[test]
    fn same_1() {
        assert_eq!(PaddingSpec::same_one(6usize, 1usize, 1, 2, true), (3, 0, 0));
    }

    #[test]
    fn same_ker_3() {
        assert_eq!(PaddingSpec::same_one(1usize, 3usize, 1, 1, true), (1, 1, 1));
        assert_eq!(PaddingSpec::same_one(2usize, 3usize, 1, 1, true), (2, 1, 1));
        assert_eq!(PaddingSpec::same_one(3usize, 3usize, 1, 1, true), (3, 1, 1));
        assert_eq!(PaddingSpec::same_one(4usize, 3usize, 1, 1, true), (4, 1, 1));
    }
}
