use crate::internal::*;

use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::{BaseDataShape, DataFormat, DataShape, SymDataShape};

use super::padding::ComputedPaddedDim;

#[derive(Debug, Clone, new, Default, Hash, PartialEq)]
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

    pub fn computed_padding<D: DimLike>(&self, input_hw: &[D]) -> TVec<ComputedPaddedDim<D>> {
        self.padding.compute(input_hw, &*self.kernel_shape, &self.dilations(), &self.strides())
    }

    pub fn output_shape<D: DimLike>(&self, input: &[D]) -> TractResult<BaseDataShape<D, TVec<D>>> {
        let ishape: BaseDataShape<D, TVec<D>> = self.data_format.shape(input.into())?;
        let computed = self.computed_padding(ishape.hw_dims());
        let spatial_dims = computed.into_iter().map(|d| d.convoluted).collect::<TVec<D>>();
        let oshape = self.data_format.from_n_c_hw(
            ishape.n().cloned().unwrap_or(1.into()),
            self.output_channel_override.map(|i| i.into()).unwrap_or(ishape.c().clone()),
            spatial_dims,
        )?;
        Ok(oshape)
    }

    pub fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let oshape = self.output_shape(&inputs[0].shape)?;
        Ok(tvec!(inputs[0].datum_type.fact(oshape.shape)))
    }

    pub fn dispose_n_axis(&self) -> PoolSpec {
        PoolSpec { data_format: self.data_format.dispose_n_axis(), ..self.clone() }
    }

    pub fn compute_geo(&self, input_full_shape: &[TDim]) -> TractResult<PoolGeometry> {
        let output_shape = self.output_shape(input_full_shape)?;
        let input_shape: SymDataShape = self.data_format.shape(input_full_shape.into())?;
        Ok(PoolGeometry::Symbolic(SymbolicPoolGeometry {
            pool_spec: self.clone(),
            input_shape,
            output_shape,
        }))
    }
}

pub type PoolGeometry = super::GeometryBound<SymbolicPoolGeometry, ConcretePoolGeometry>;

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct SymbolicPoolGeometry {
    pub pool_spec: PoolSpec,
    pub input_shape: SymDataShape,
    pub output_shape: SymDataShape,
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct ConcretePoolGeometry {
    pub input_shape: DataShape,
    pub patch: Patch,
    pub output_shape: DataShape,
}

impl super::ResolveTo<ConcretePoolGeometry> for SymbolicPoolGeometry {
    type Param = [usize];
    fn resolve(&self, input_full_shape: &[usize]) -> TractResult<ConcretePoolGeometry> {
        let input_shape = self.pool_spec.data_format.shape(input_full_shape.into())?;
        let output_inner_stride = match self.pool_spec.data_format {
            DataFormat::NCHW | DataFormat::CHW => 1,
            DataFormat::NHWC | DataFormat::HWC => {
                self.pool_spec.output_channel_override.clone().unwrap_or(*input_shape.c())
            }
        };
        let mut spec = PatchSpec::for_full_shape(self.pool_spec.data_format, input_full_shape)?
            .with_output_inner_stride(output_inner_stride)
            .with_kernel_shape(self.pool_spec.kernel_shape.clone())
            .with_padding(self.pool_spec.padding.clone());
        if let Some(strides) = self.pool_spec.strides.clone() {
            spec = spec.with_strides(strides);
        }
        if let Some(dilations) = self.pool_spec.dilations.clone() {
            spec = spec.with_dilations(dilations);
        }
        let patch = spec.into_patch();
        let output_shape = input_shape.fmt.from_n_c_hw(
            *input_shape.n().unwrap_or(&1),
            self.pool_spec.output_channel_override.unwrap_or(*input_shape.c()),
            &*patch.output_shape,
        )?;
        Ok(ConcretePoolGeometry { input_shape, patch, output_shape })
    }
}
