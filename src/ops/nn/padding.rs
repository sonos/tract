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
    pub fn compute<D: DimLike, KD: Into<D> + Copy>(
        &self,
        input_spatial_shape: &[D],
        kernel_spatial_shape: &[KD],
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

    fn explicit<D: DimLike, KD: Into<D> + Copy>(
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[KD],
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
                let kernel_field = (kernel_spatial_shape[ax].into() - 1) * dilations[ax] + 1;
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

    fn same<D: DimLike, KD: Into<D> + Copy>(
        &self,
        data_spatial_shape: &[D],
        kernel_spatial_shape: &[KD],
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
        (dims, pad_before, pad_after)
    }
}
