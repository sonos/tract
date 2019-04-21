use crate::internal::*;

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, new, PartialEq)]
pub struct ComputedPaddedDim<D: DimLike> {
    pub output: D,
    pub pad_before: D,
    pub pad_after: D,
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
    ) -> TVec<ComputedPaddedDim<D>> {
        (0..input_spatial_shape.len())
            .map(|d| {
                self.compute_one(
                    d,
                    input_spatial_shape[d],
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
        input: D,
        kernel: usize,
        dilation: usize,
        stride: usize,
    ) -> ComputedPaddedDim<D> {
        match self {
            PaddingSpec::Valid => Self::explicit(input, kernel, dilation, stride, 0, 0),
            PaddingSpec::Explicit(ref bef, ref aft) => {
                Self::explicit(input, kernel, dilation, stride, bef[axis], aft[axis])
            }
            PaddingSpec::SameUpper => Self::same(input, kernel, dilation, stride, true),
            PaddingSpec::SameLower => Self::same(input, kernel, dilation, stride, false),
        }
    }

    fn explicit<D: DimLike>(
        input: D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
    ) -> ComputedPaddedDim<D> {
        let kernel_field = (kernel - 1) * dilation + 1;
        let output = (input + bef + aft - kernel_field + 1).div_ceil(stride);
        ComputedPaddedDim::new(output, bef.into(), aft.into())
    }

    fn same<D: DimLike>(
        input: D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        upper: bool,
    ) -> ComputedPaddedDim<D> {
        let output = input.div_ceil(stride);
        let kernel_field = (kernel - 1) * dilation + 1;
        let pad = if let Ok(input) = input.to_integer() {
            let pad = ((((output - 1) * stride + kernel_field)).to_integer().unwrap() - input).max(0);
            (pad as usize).into()
        } else {
            (output - 1) * stride + kernel_field - input
        };
        let lower_pad = pad / 2;
        let higher_pad = pad - pad / 2;
        let (before, after) = if upper { (lower_pad, higher_pad) } else { (higher_pad, lower_pad) };
        ComputedPaddedDim::new(output, before, after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_stride_1() {
        assert_eq!(PaddingSpec::same(1usize, 2usize, 1, 1, true), ComputedPaddedDim::new(1, 0, 1));
        assert_eq!(PaddingSpec::same(2usize, 2usize, 1, 1, true), ComputedPaddedDim::new(2, 0, 1));
        assert_eq!(PaddingSpec::same(3usize, 2usize, 1, 1, true), ComputedPaddedDim::new(3, 0, 1));
        assert_eq!(PaddingSpec::same(4usize, 2usize, 1, 1, true), ComputedPaddedDim::new(4, 0, 1));
    }

    #[test]
    fn same_stride_2() {
        assert_eq!(PaddingSpec::same(1usize, 2usize, 1, 2, true), ComputedPaddedDim::new(1, 0, 1));
        assert_eq!(PaddingSpec::same(2usize, 2usize, 1, 2, true), ComputedPaddedDim::new(1, 0, 0));
        assert_eq!(PaddingSpec::same(3usize, 2usize, 1, 2, true), ComputedPaddedDim::new(2, 0, 1));
        assert_eq!(PaddingSpec::same(4usize, 2usize, 1, 2, true), ComputedPaddedDim::new(2, 0, 0));
    }

    #[test]
    fn same_1() {
        assert_eq!(PaddingSpec::same(6usize, 1usize, 1, 2, true), ComputedPaddedDim::new(3, 0, 0));
    }

    #[test]
    fn same_lower() {
        assert_eq!(
            PaddingSpec::same(10usize, 2usize, 1, 3, false),
            ComputedPaddedDim::new(4, 1, 0)
        );
    }

    #[test]
    fn same_ker_3() {
        assert_eq!(PaddingSpec::same(1usize, 3usize, 1, 1, true), ComputedPaddedDim::new(1, 1, 1));
        assert_eq!(PaddingSpec::same(2usize, 3usize, 1, 1, true), ComputedPaddedDim::new(2, 1, 1));
        assert_eq!(PaddingSpec::same(3usize, 3usize, 1, 1, true), ComputedPaddedDim::new(3, 1, 1));
        assert_eq!(PaddingSpec::same(4usize, 3usize, 1, 1, true), ComputedPaddedDim::new(4, 1, 1));
    }

    #[test]
    fn valid_1() {
        assert_eq!(
            PaddingSpec::explicit(10usize, 2usize, 1, 3, 0, 0),
            ComputedPaddedDim::new(3, 0, 0)
        );
    }

}
