use crate::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum PaddingSpec {
    Explicit(TVec<usize>, TVec<usize>),
    ExplicitOnnxPool(TVec<usize>, TVec<usize>, bool),
    #[default]
    Valid,
    SameUpper,
    SameLower,
}

use PaddingSpec::*;

#[derive(Debug, Clone, new, PartialEq, Eq)]
pub struct ComputedPaddedDim<D: DimLike> {
    pub deconvoluted: D,
    pub convoluted: D,
    pub pad_before: D,
    pub pad_after: D,
}

impl PaddingSpec {
    pub fn valid_dim(&self, d: usize, stride_is_one: bool) -> bool {
        match self {
            Valid => true,
            Explicit(bef, aft) => bef[d] == 0 && aft[d] == 0,
            ExplicitOnnxPool(a, b, ceil_mode) => {
                (*ceil_mode || stride_is_one) && a[d] == 0 && b[d] == 0
            }
            _ => false,
        }
    }

    pub fn change_geo_axes(&self, op: &AxisOp) -> TractResult<PaddingSpec> {
        match &self {
            ExplicitOnnxPool(before, after, round) => {
                let mut before: TVec<usize> = before.clone();
                let mut after: TVec<usize> = after.clone();
                op.change_shape_array(&mut before, false)?;
                op.change_shape_array(&mut after, false)?;
                if let AxisOp::Add(add) = op {
                    before[*add] = 0;
                    after[*add] = 0;
                }
                Ok(ExplicitOnnxPool(before, after, *round))
            }
            Explicit(before, after) => {
                let mut before: TVec<usize> = before.clone();
                let mut after: TVec<usize> = after.clone();
                op.change_shape_array(&mut before, false)?;
                op.change_shape_array(&mut after, false)?;
                if let AxisOp::Add(add) = op {
                    before[*add] = 0;
                    after[*add] = 0;
                }
                Ok(Explicit(before, after))
            }
            Valid | SameLower | SameUpper => Ok(self.clone()),
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
        adjustments: &[usize],
    ) -> TractResult<TVec<ComputedPaddedDim<D>>> {
        (0..conv_spatial_shape.len())
            .map(|d| {
                self.compute_one_for_deconv(
                    d,
                    &conv_spatial_shape[d],
                    kernel_spatial_shape[d],
                    dilations[d],
                    strides[d],
                    adjustments[d],
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
            Valid => Self::valid(input, kernel, dilation, stride),
            Explicit(ref bef, ref aft) => {
                Self::explicit(input, kernel, dilation, stride, bef[axis], aft[axis])
            }
            ExplicitOnnxPool(ref bef, ref aft, ceil_mode) => Self::explicit_onnx_pool(
                input, kernel, dilation, stride, bef[axis], aft[axis], *ceil_mode,
            ),
            SameUpper => Self::same(input, kernel, dilation, stride, true),
            SameLower => Self::same(input, kernel, dilation, stride, false),
        }
    }

    pub fn compute_one_for_deconv<D: DimLike>(
        &self,
        axis: usize,
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        adjustment: usize,
    ) -> TractResult<ComputedPaddedDim<D>> {
        match self {
            Valid => Self::valid_for_deconv(input, kernel, dilation, stride, adjustment),
            SameUpper => Self::same_for_deconv(input, kernel, dilation, stride, adjustment, true),
            SameLower => Self::same_for_deconv(input, kernel, dilation, stride, adjustment, false),
            Explicit(ref bef, ref aft) => Self::explicit_for_deconv(
                input, kernel, dilation, stride, bef[axis], aft[axis], adjustment,
            ),
            // unreachable ?
            ExplicitOnnxPool(ref bef, ref aft, _ceil_mode) => Self::explicit_for_deconv(
                input, kernel, dilation, stride, bef[axis], aft[axis], adjustment,
            ),
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
            D::from((int + 1).saturating_sub(kernel_field).divceil(stride))
        } else {
            (input.clone() + 1 - kernel_field).divceil(stride)
        };
        ComputedPaddedDim::new(input.clone(), output, 0.into(), 0.into())
    }

    fn valid_for_deconv<D: DimLike>(
        convoluted: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        adjustment: usize,
    ) -> TractResult<ComputedPaddedDim<D>> {
        let kernel_field = (kernel - 1) * dilation + 1;
        let deconvoluted = (convoluted.clone() - 1) * stride + kernel_field + adjustment;
        Ok(ComputedPaddedDim::new(deconvoluted, convoluted.clone(), 0.into(), 0.into()))
    }

    fn explicit<D: DimLike>(
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
    ) -> ComputedPaddedDim<D> {
        if let Ok(i) = input.to_dim().to_usize() {
            let ints = Self::explicit_usize(i, kernel, dilation, stride, bef, aft);
            ComputedPaddedDim::new(
                input.clone(),
                ints.convoluted.into(),
                ints.pad_before.into(),
                ints.pad_after.into(),
            )
        } else {
            let kernel_field = (kernel - 1) * dilation + 1;
            let dividend = input.clone() + bef + aft - kernel_field;
            let output = dividend.div(stride) + 1;
            ComputedPaddedDim::new(input.clone(), output, bef.into(), aft.into())
        }
    }

    fn explicit_usize(
        input: usize,
        kernel: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
    ) -> ComputedPaddedDim<usize> {
        let kernel_field = (kernel - 1) * dilation + 1;
        let dividend = (input + bef + aft).saturating_sub(kernel_field);
        let output = dividend / stride + 1;
        ComputedPaddedDim::new(input, output, bef, aft)
    }

    fn explicit_onnx_pool<D: DimLike>(
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
        ceil_mode: bool,
    ) -> ComputedPaddedDim<D> {
        if let Ok(i) = input.to_dim().to_usize() {
            let ints =
                Self::explicit_onnx_pool_usize(i, kernel, dilation, stride, bef, aft, ceil_mode);
            ComputedPaddedDim::new(
                input.clone(),
                ints.convoluted.into(),
                ints.pad_before.into(),
                ints.pad_after.into(),
            )
        } else {
            // output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
            let kernel_field = (kernel - 1) * dilation + 1;
            let dividend = input.clone() + bef + aft - kernel_field;
            let output =
                if ceil_mode { dividend.divceil(stride) } else { dividend.div(stride) } + 1;
            ComputedPaddedDim::new(input.clone(), output, bef.into(), aft.into())
        }
    }

    fn explicit_onnx_pool_usize(
        input: usize,
        kernel: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
        ceil_mode: bool,
    ) -> ComputedPaddedDim<usize> {
        // output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
        let kernel_field = (kernel - 1) * dilation + 1;
        let dividend = (input + bef + aft).saturating_sub(kernel_field);
        let mut output = if ceil_mode { dividend.divceil(stride) } else { dividend / stride } + 1;
        if ceil_mode {
            // ensure that the last pooling starts inside the image
            // needed to avoid problems in ceil mode
            if (output - 1) * stride >= input + bef {
                output -= 1;
            }
        }
        ComputedPaddedDim::new(input, output, bef, aft)
    }

    fn explicit_for_deconv<D: DimLike>(
        convoluted: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
        adjustment: usize,
    ) -> TractResult<ComputedPaddedDim<D>> {
        let kernel_field = (kernel - 1) * dilation + 1;
        let deconvoluted =
            (convoluted.clone() - 1) * stride + kernel_field - bef - aft + adjustment;
        Ok(ComputedPaddedDim::new(deconvoluted, convoluted.clone(), bef.into(), aft.into()))
    }

    fn same<D: DimLike>(
        input: &D,
        kernel: usize,
        dilation: usize,
        stride: usize,
        upper: bool,
    ) -> ComputedPaddedDim<D> {
        let output = input.divceil(stride);
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
        adjustment: usize,
        upper: bool,
    ) -> TractResult<ComputedPaddedDim<D>> {
        if (kernel - 1) * dilation < stride {
            bail!("Invalid axis geometry for SAME padding: expect (kernel_len - 1) * dilation > stride - 1");
        }
        let kernel_field = (kernel - 1) * dilation + 1;
        let crop = kernel_field + adjustment - stride;
        let lower_crop = crop / 2;
        let higher_crop = crop - lower_crop;
        let (before, after) =
            if upper { (lower_crop, higher_crop) } else { (higher_crop, lower_crop) };
        let deconvoluted = (convoluted.clone() - 1) * stride + kernel_field - before - after;
        Ok(ComputedPaddedDim::new(deconvoluted, convoluted.clone(), before.into(), after.into()))
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
    fn same_ker_3_stride_3() {
        assert_eq!(PS::same(&3usize, 3usize, 1, 3, true), ComputedPaddedDim::new(3, 1, 0, 0));
    }

    #[test]
    fn valid_1() {
        assert_eq!(PS::valid(&10usize, 2usize, 1, 3), ComputedPaddedDim::new(10, 3, 0, 0));
    }

    #[test]
    fn explicit_2() {
        assert_eq!(
            PS::explicit_onnx_pool(&28usize, 3usize, 1, 1, 2, 2, true),
            ComputedPaddedDim::new(28, 30, 2, 2)
        );
    }

    #[test]
    #[ignore = "ONNX weird output computation for explicit"]
    fn explicit_3() {
        assert_eq!(
            PS::explicit_onnx_pool(&2usize, 1usize, 1, 2, 0, 0, true),
            ComputedPaddedDim::new(2, 2, 0, 0)
        );
    }

    #[test]
    fn same_upper() {
        assert_eq!(PS::same(&7usize, 1usize, 1, 2, true), ComputedPaddedDim::new(7, 4, 0, 0));
    }

    // 0 1 2 3 4 5 6 7 8 9 a b
    // 012 345 678 9ab
    #[test]
    fn bug_explicit_stride() {
        assert_eq!(
            PS::explicit_onnx_pool(&12usize, 3usize, 1, 3, 0, 0, false),
            ComputedPaddedDim::new(12, 4, 0, 0)
        );
    }
}
