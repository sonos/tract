use crate::internal::*;

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum PaddingSpec {
    Explicit(TVec<usize>, TVec<usize>, bool),
    Valid,
    SameUpper,
    SameLower,
}

impl Default for PaddingSpec {
    fn default() -> PaddingSpec {
        PaddingSpec::Valid
    }
}

#[derive(Debug, Clone, new, PartialEq)]
pub struct ComputedPaddedDim<D: DimLike> {
    pub deconvoluted: D,
    pub convoluted: D,
    pub pad_before: D,
    pub pad_after: D,
}

impl PaddingSpec {
    pub fn valid_dim(&self, d: usize) -> bool {
        match self {
            PaddingSpec::Valid => true,
            PaddingSpec::Explicit(a, b, ceil_mode) => *ceil_mode && a[d] == 0 && b[d] == 0,
            _ => false,
        }
    }

    pub fn rm_axis(&self, d: usize) -> PaddingSpec {
        match self {
            PaddingSpec::Explicit(a, b, ceil_mode) => {
                let mut a = a.clone();
                let mut b = b.clone();
                a.remove(d);
                b.remove(d);
                PaddingSpec::Explicit(a, b, *ceil_mode)
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
    ) -> TVec<ComputedPaddedDim<D>> {
        (0..input_spatial_shape.len())
            .map(|d| {
                self.compute_one(
                    d,
                    &input_spatial_shape[d],
                    kernel_spatial_shape[d],
                    dilations[d],
                    strides[d],
                )
            })
            .collect()
    }

    pub fn compute_for_deconv<D: DimLike>(
        &self,
        conv_spatial_shape: &[D],
        kernel_spatial_shape: &[usize],
        dilations: &[usize],
        strides: &[usize],
    ) -> TVec<ComputedPaddedDim<D>> {
        (0..conv_spatial_shape.len())
            .map(|d| {
                self.compute_one_for_deconv(
                    d,
                    &conv_spatial_shape[d],
                    kernel_spatial_shape[d],
                    dilations[d],
                    strides[d],
                )
            })
            .collect()
    }

    pub fn compute_one<D: DimLike>(
        &self,
        axis: usize,
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
    ) -> ComputedPaddedDim<D> {
        match self {
            PaddingSpec::Valid => Self::valid(input, kernel, dilation, stride),
            PaddingSpec::Explicit(ref bef, ref aft, ceil_mode) => {
                Self::explicit(input, kernel, dilation, stride, bef[axis], aft[axis], *ceil_mode)
            }
            PaddingSpec::SameUpper => Self::same(input, kernel, dilation, stride, true),
            PaddingSpec::SameLower => Self::same(input, kernel, dilation, stride, false),
        }
    }

    pub fn compute_one_for_deconv<D: DimLike>(
        &self,
        axis: usize,
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
    ) -> ComputedPaddedDim<D> {
        match self {
            PaddingSpec::Valid => Self::valid_for_deconv(input, kernel, dilation, stride),
            PaddingSpec::SameUpper => Self::same_for_deconv(input, kernel, dilation, stride, true),
            PaddingSpec::SameLower => Self::same_for_deconv(input, kernel, dilation, stride, false),
            _ => panic!(),
            /*
            PaddingSpec::Explicit(ref bef, ref aft, ceil_mode) => {
                Self::explicit(input, kernel, dilation, stride, bef[axis], aft[axis], *ceil_mode)
            }
            */
        }
    }

    fn valid<D: DimLike>(
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
    ) -> ComputedPaddedDim<D> {
        let kernel_field = (kernel - 1) * dilation + 1;
        let output = if let Ok(int) = input.to_usize() {
            D::from((int + 1).saturating_sub(kernel_field).div_ceil(stride))
        } else {
            (input.clone() + 1 - kernel_field).div_ceil(stride)
        };
        ComputedPaddedDim::new(input.clone(), output, 0.into(), 0.into())
    }

    fn valid_for_deconv<D: DimLike>(
        convoluted: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
    ) -> ComputedPaddedDim<D> {
        assert_eq!(stride, 1);
        let kernel_field = (kernel - 1) * dilation + 1;
        let deconvoluted = convoluted.clone() + kernel_field - 1;
        ComputedPaddedDim::new(deconvoluted, convoluted.clone(), 0.into(), 0.into())
    }

    fn explicit<D: DimLike>(
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
        ceil_mode: bool,
    ) -> ComputedPaddedDim<D> {
        let kernel_field = (kernel - 1) * dilation + 1;
        let dividend = if let Ok(int) = input.to_usize() {
            D::from((int + bef + aft).saturating_sub(kernel_field))
        } else {
            input.clone() + bef + aft - kernel_field
        };
        let output = if ceil_mode { dividend.div_ceil(stride) } else { dividend.div(stride) } + 1;
        ComputedPaddedDim::new(input.clone(), output, bef.into(), aft.into())
    }

    fn same<D: DimLike>(
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        upper: bool,
    ) -> ComputedPaddedDim<D> {
        let output = input.div_ceil(stride);
        let kernel_field = (kernel - 1) * dilation + 1;
        let pad = if let Ok(input) = input.to_usize() {
            let pad = (((output.clone() - 1) * stride + kernel_field).to_usize().unwrap())
                .saturating_sub(input);
            pad.into()
        } else {
            (output.clone() - 1) * stride + kernel_field - input
        };
        let lower_pad = pad.clone() / 2;
        let higher_pad = pad - &lower_pad;
        let (before, after) = if upper { (lower_pad, higher_pad) } else { (higher_pad, lower_pad) };
        ComputedPaddedDim::new(input.clone(), output, before, after) // TODO input is wrong for stride != 1
    }

    fn same_for_deconv<D: DimLike>(
        convoluted: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        upper: bool,
    ) -> ComputedPaddedDim<D> {
        assert_eq!(stride, 1);
        let kernel_field = (kernel - 1) * dilation + 1;
        let crop = kernel_field - 1;
        let lower_crop = crop.clone() / 2;
        let higher_crop = crop - &lower_crop;
        let (before, after) = if upper { (lower_crop, higher_crop) } else { (higher_crop, lower_crop) };
        ComputedPaddedDim::new(convoluted.clone(), convoluted.clone(), before.into(), after.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PaddingSpec as PS;

    #[test]
    fn same_stride_1() {
        assert_eq!(PS::same(&1usize, 2usize, 1, 1, true), ComputedPaddedDim::new(1, 1, 0, 1));
        assert_eq!(PS::same(&2usize, 2usize, 1, 1, true), ComputedPaddedDim::new(2, 2, 0, 1));
        assert_eq!(PS::same(&3usize, 2usize, 1, 1, true), ComputedPaddedDim::new(3, 3, 0, 1));
        assert_eq!(PS::same(&4usize, 2usize, 1, 1, true), ComputedPaddedDim::new(4, 4, 0, 1));
    }

    #[test]
    fn same_stride_2() {
        assert_eq!(PS::same(&1usize, 2usize, 1, 2, true), ComputedPaddedDim::new(1, 1, 0, 1));
        assert_eq!(PS::same(&2usize, 2usize, 1, 2, true), ComputedPaddedDim::new(2, 1, 0, 0));
        assert_eq!(PS::same(&3usize, 2usize, 1, 2, true), ComputedPaddedDim::new(3, 2, 0, 1));
        assert_eq!(PS::same(&4usize, 2usize, 1, 2, true), ComputedPaddedDim::new(4, 2, 0, 0));
    }

    #[test]
    fn same_1() {
        assert_eq!(PS::same(&6usize, 1usize, 1, 2, true), ComputedPaddedDim::new(6, 3, 0, 0));
    }

    #[test]
    fn same_lower() {
        assert_eq!(PS::same(&10usize, 2usize, 1, 3, false), ComputedPaddedDim::new(10, 4, 1, 0));
    }

    #[test]
    fn same_ker_3() {
        assert_eq!(PS::same(&1usize, 3usize, 1, 1, true), ComputedPaddedDim::new(1, 1, 1, 1));
        assert_eq!(PS::same(&2usize, 3usize, 1, 1, true), ComputedPaddedDim::new(2, 2, 1, 1));
        assert_eq!(PS::same(&3usize, 3usize, 1, 1, true), ComputedPaddedDim::new(3, 3, 1, 1));
        assert_eq!(PS::same(&4usize, 3usize, 1, 1, true), ComputedPaddedDim::new(4, 4, 1, 1));
    }

    #[test]
    fn valid_1() {
        assert_eq!(PS::valid(&10usize, 2usize, 1, 3), ComputedPaddedDim::new(10, 3, 0, 0));
    }

    #[test]
    fn explicit_2() {
        assert_eq!(
            PS::explicit(&28usize, 3usize, 1, 1, 2, 2, true),
            ComputedPaddedDim::new(28, 30, 2, 2)
        );
    }

    #[test]
    fn same_upper() {
        assert_eq!(PS::same(&7usize, 1usize, 1, 2, true), ComputedPaddedDim::new(7, 4, 0, 0));
    }
}
