use crate::internal::*;

use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::{BaseDataShape, DataFormat, DataShape, SymDataShape};

use super::padding::ComputedPaddedDim;

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct PoolSpec {
    pub data_format: DataFormat,
    pub kernel_shape: TVec<usize>,
    pub padding: PaddingSpec,
    pub dilations: Option<TVec<usize>>,
    pub strides: Option<TVec<usize>>,
    pub input_channels: usize,
    pub output_channels: usize,
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
        self.padding.compute(input_hw, &self.kernel_shape, &self.dilations(), &self.strides())
    }

    pub fn output_shape<D: DimLike>(&self, input: &[D]) -> TractResult<BaseDataShape<D, TVec<D>>> {
        let ishape: BaseDataShape<D, TVec<D>> = self.data_format.shape(input.into())?;
        ensure!(ishape.c().to_dim() == self.input_channels.to_dim());
        let computed = self.computed_padding(ishape.hw_dims());
        let spatial_dims = computed.into_iter().map(|d| d.convoluted).collect::<TVec<D>>();
        let oshape = self.data_format.from_n_c_hw(
            ishape.n().cloned().unwrap_or_else(|| 1.into()),
            self.output_channels.into(),
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

    pub fn change_geo_axes(&self, op: &AxisOp) -> TractResult<PoolSpec> {
        let mut dilations = self.dilations().into_owned().into();
        op.change_shape_array(&mut dilations, false)?;
        let mut kernel_shape = self.kernel_shape.clone();
        op.change_shape_array(&mut kernel_shape, false)?;
        let mut strides = self.strides().into_owned().into();
        op.change_shape_array(&mut strides, false)?;
        let padding = self.padding.change_geo_axes(op)?;
        Ok(PoolSpec {
            kernel_shape,
            padding,
            dilations: Some(dilations),
            strides: Some(strides),
            ..self.clone()
        })
    }

    pub fn declutter(&self, input: &[TDim]) -> TractResult<Option<PoolSpec>> {
        if let PaddingSpec::ExplicitOnnxPool(before, after, _) = &self.padding {
            let input = self.data_format.shape(input)?;
            let input_hw = input.hw_dims();
            let reference = self.computed_padding(input_hw);
            for replacement in [
                PaddingSpec::Valid,
                PaddingSpec::SameUpper,
                PaddingSpec::SameLower,
                PaddingSpec::Explicit(before.clone(), after.clone()),
            ] {
                let new_pool_spec = PoolSpec { padding: replacement, ..self.clone() };
                if new_pool_spec.computed_padding(input_hw) == reference {
                    return Ok(Some(new_pool_spec));
                }
            }
        }
        Ok(None)
    }
}

pub type PoolGeometry = super::GeometryBound<SymbolicPoolGeometry, ConcretePoolGeometry>;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SymbolicPoolGeometry {
    pub pool_spec: PoolSpec,
    pub input_shape: SymDataShape,
    pub output_shape: SymDataShape,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
            DataFormat::NHWC | DataFormat::HWC => self.pool_spec.output_channels,
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
            self.pool_spec.output_channels,
            &*patch.output_shape,
        )?;
        Ok(ConcretePoolGeometry { input_shape, patch, output_shape })
    }
}
