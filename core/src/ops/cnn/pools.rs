use crate::internal::*;

use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::{DataFormat, DataShape};

#[derive(Debug, Clone, new, Default, Hash)]
pub struct PoolSpec {
    pub data_format: DataFormat,
    pub kernel_shape: TVec<usize>,
    pub padding: PaddingSpec,
    pub dilations: Option<TVec<usize>>,
    pub strides: Option<TVec<usize>>,
    pub output_channel_override: Option<usize>,
}

impl PoolSpec {
    pub fn info(&self) -> Vec<String> {
        vec![
            format!("Data format: {:?}", self.data_format),
            format!(
                "Kernel shape:{:?} (strides:{:?}, padding:{:?}, dilations:{:?})",
                self.kernel_shape, self.strides, self.padding, self.dilations,
            ),
        ]
    }

    pub fn rank(&self) -> usize {
        self.kernel_shape.len()
    }

    pub fn dilation(&self, geo_axis: usize) -> usize {
        self.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1)
    }

    pub fn dilations(&self) -> Cow<[usize]> {
        self.dilations
            .as_deref()
            .map_or_else(|| vec![1; self.kernel_shape.len()].into(), |d| d.into())
    }

    pub fn stride(&self, geo_axis: usize) -> usize {
        self.strides.as_ref().map(|s| s[geo_axis]).unwrap_or(1)
    }

    pub fn strides(&self) -> Cow<[usize]> {
        self.strides
            .as_deref()
            .map_or_else(|| vec![1; self.kernel_shape.len()].into(), |d| d.into())
    }

    pub fn compute_geo(
        &self,
        input_full_shape: &[usize],
    ) -> TractResult<(DataShape, Patch, DataShape)> {
        let input_shape = self.data_format.shape(input_full_shape.into())?;
        let output_inner_stride = match self.data_format {
            DataFormat::NCHW | DataFormat::CHW => 1,
            DataFormat::NHWC | DataFormat::HWC => {
                self.output_channel_override.clone().unwrap_or(*input_shape.c())
            }
        };
        let mut spec = PatchSpec::for_full_shape(self.data_format, input_full_shape)?
            .with_output_inner_stride(output_inner_stride)
            .with_kernel_shape(self.kernel_shape.clone())
            .with_padding(self.padding.clone());
        if let Some(strides) = self.strides.clone() {
            spec = spec.with_strides(strides);
        }
        if let Some(dilations) = self.dilations.clone() {
            spec = spec.with_dilations(dilations);
        }
        let patch = spec.into_patch();
        let output_shape = input_shape.fmt.from_n_c_hw(
            *input_shape.n().unwrap_or(&1),
            self.output_channel_override.unwrap_or(*input_shape.c()),
            &*patch.output_shape,
        )?;
        Ok((input_shape, patch, output_shape))
    }

    pub fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let ishape = self.data_format.shape(inputs[0].shape.to_tvec())?;
        let computed = self.padding.compute(
            ishape.hw_dims(),
            &*self.kernel_shape,
            &self.dilations(),
            &self.strides(),
        );
        let spatial_dims = computed.into_iter().map(|d| d.convoluted).collect::<TVec<TDim>>();
        let oshape = self.data_format.from_n_c_hw(
            ishape.n().cloned().unwrap_or(1.to_dim()),
            self.output_channel_override.map(|i| i.to_dim()).unwrap_or(ishape.c().clone()),
            spatial_dims,
        )?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, oshape.shape)))
    }

    pub fn dispose_n_axis(&self) -> PoolSpec {
        PoolSpec { data_format: self.data_format.dispose_n_axis(), ..self.clone() }
    }
}
