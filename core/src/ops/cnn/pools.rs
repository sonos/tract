use crate::internal::*;

use num_traits::Zero;

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
        let spatial_dims = computed.into_iter().map(|d| d.output).collect::<TVec<TDim>>();
        let oshape = self.data_format.from_n_c_hw(
            ishape.n().cloned().unwrap_or(1.to_dim()),
            self.output_channel_override.map(|i| i.to_dim()).unwrap_or(ishape.c().clone()),
            spatial_dims,
        )?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*oshape.shape)?))
    }

    pub fn pulsify(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<(OutletId, PoolSpec)> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        let input_shape = self.data_format.shape(&*fact.shape)?;
        if Some(fact.axis) == input_shape.n_axis() {
            Ok((input, self.clone()))
        } else if fact.axis == input_shape.c_axis() {
            bail!("Can not pulsify cnn pooling ops along the input channel axis");
        } else {
            let geo_axis = fact.axis - input_shape.h_axis();
            let stride = self.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
            let pulse = fact.pulse();
            if fact.pulse() % stride != 0 {
                bail!("Pulsificaton requires pulse to be a stride multiple")
            }
            let dilation = self.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
            let kernel_len = (self.kernel_shape[geo_axis] - 1) * dilation;
            let mut wire = input;

            dbg!(self);
            let computed_padding = self.padding.compute_one(
                geo_axis,
                &fact.dim,
                self.kernel_shape[geo_axis],
                self.dilation(geo_axis),
                self.stride(geo_axis),
            );
            println!("{}", computed_padding.pad_before);
            println!("100->{:?}", computed_padding.pad_before.eval(100));
            println!("101->{:?}", computed_padding.pad_before.eval(101));
            let has_padding = computed_padding.pad_before != TDim::zero()
                || computed_padding.pad_after != TDim::zero();

            /*
            if has_padding {
                let mut pads = vec![(0, 0); input_shape.rank()];
                pads[fact.axis] = (
                    computed_padding.pad_before.to_integer()? as usize,
                    computed_padding.pad_after.to_integer()? as usize,
                );
                let pad_op =
                    crate::ops::array::Pad { pads, mode: crate::ops::array::PadMode::default() };
                wire = pad_op.pulsify(source, node, target, mapping, pulse)?[0];
            }
            */

            let overlap = (kernel_len + 1).saturating_sub(stride);
            let pad_before = computed_padding.pad_before.to_integer()? as usize;
            let start_at = pad_before.max(fact.delay + overlap);
            let padded_start_at = start_at - pad_before;
            let misalignment = padded_start_at % stride;
            if overlap > 0 || misalignment > 0 || pad_before > fact.delay {
                let aligned_padded_start_at = padded_start_at.div_ceil(stride) * stride;
                let start_at = aligned_padded_start_at + pad_before;
                let delay = start_at - overlap - fact.delay;
                wire = target.wire_node(
                    format!("{}.Delay", node.name),
                    crate::pulse::delay::Delay::new(&fact, delay, overlap),
                    &[wire],
                )?[0];
            }

            if has_padding {
                let mut bef = tvec!();
                let mut aft = tvec!();
                for ix in 0..input_shape.hw_rank() {
                    if ix == geo_axis {
                        bef.push(0);
                        aft.push(0);
                    } else {
                        let c = self.padding.compute_one(
                            ix,
                            &input_shape.hw_dims()[ix],
                            self.kernel_shape[ix],
                            self.dilations()[ix],
                            self.strides()[ix],
                        );
                        bef.push(c.pad_before);
                        aft.push(c.pad_after);
                    };
                }
                Ok((wire, PoolSpec { padding: PaddingSpec::Explicit(bef, aft), ..self.clone() }))
            } else {
                Ok((wire, self.clone()))
            }
        }
    }

    pub fn dispose_n_axis(&self) -> PoolSpec {
        PoolSpec { data_format: self.data_format.dispose_n_axis(), ..self.clone() }
    }

    pub fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let ishape = self.data_format.shape(&inputs[0].shape)?;
        let computed = self.padding.compute(
            ishape.hw_dims(),
            &*self.kernel_shape,
            &self.dilations(),
            &self.strides(),
        );
        let spatial_dims = computed.into_iter().map(|d| d.output).collect::<TVec<usize>>();
        let oshape = self.data_format.from_n_c_hw(
            ishape.n().cloned().unwrap_or(1),
            self.output_channel_override.unwrap_or(*ishape.c()),
            spatial_dims,
        )?;
        let mut fact = inputs[0].clone();
        let input_shape = self.data_format.shape(&*fact.shape)?;
        let geo_axis = fact.axis - input_shape.h_axis();
        let dilation = self.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
        let kernel_len = (self.kernel_shape[geo_axis] - 1) * dilation;
        let stride = self.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
        fact.delay /= stride;
        fact.dim = (fact.dim.clone() - kernel_len.to_dim()).div_ceil(stride as u32);
        fact.shape = oshape.shape;
        Ok(tvec!(fact))
    }
}
