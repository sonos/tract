use std::str::FromStr;

use crate::ops::prelude::*;

#[derive(Debug, Clone)]
pub enum PaddingSpec {
    Explicit(TVec<usize>, TVec<usize>),
    Valid,
    SameUpper,
    SameLower,
}

impl FromStr for PaddingSpec {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "NOTSET" => Ok(PaddingSpec::Valid),
            "VALID" => Ok(PaddingSpec::Valid),
            "SAME_UPPER" => Ok(PaddingSpec::SameUpper),
            "SAME_LOWER" => Ok(PaddingSpec::SameLower),
            _ => Err(()),
        }
    }
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

    pub fn compute<D: DimLike, KD: Into<D> + Copy>(
        &self,
        input_spatial_shape: &[D],
        kernel_spatial_shape: &[KD],
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

    fn explicit<D: DimLike, KD: Into<D> + Copy>(
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[KD],
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
                let kernel_field = (kernel_spatial_shape[ax].into() - 1) * dilations[ax] + 1;
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

    fn same<D: DimLike, KD: Into<D> + Copy>(
        &self,
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[KD],
        dilations: &[usize],
        strides: &[usize],
        upper: bool,
    ) -> ComputedPaddedDim<D> {
        let spatial_rank = data_spatial_shape.len();
        let mut dims = tvec![];
        let mut pad_before = tvec![];
        let mut pad_after = tvec![];
        for ax in 0..spatial_rank {
            let dim = data_spatial_shape[ax].div_ceil(strides[ax]);
            let kernel_field = (kernel_spatial_shape[ax].into() - 1) * dilations[ax] + 1;
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
        ComputedPaddedDim {
            pad_before,
            pad_after,
            output: dims,
        }
    }
}
