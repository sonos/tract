use crate::internal::*;

use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::{DataFormat, DataShape};

#[derive(Debug, Clone, new, Default)]
pub struct PoolSpec {
    data_format: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
}

impl PoolSpec {
    pub fn info(&self) -> Vec<String> {
        vec![
            format!("Data format: {:?}", self.data_format),
            format!(
                "Kernel shape:{:?} (strides:{:?}, padding:{:?})",
                self.kernel_shape, self.strides, self.padding,
            ),
        ]
    }

    pub fn compute_geo(&self, input_full_shape: &[usize]) -> (DataShape, Patch, DataShape) {
        let input_shape = self.data_format.shape(input_full_shape.into());
        let mut spec = PatchSpec::for_full_shape(self.data_format, input_full_shape)
            .with_output_inner_stride(*input_shape.w_stride())
            .with_kernel_shape(self.kernel_shape.clone())
            .with_padding(self.padding.clone());
        if let Some(strides) = self.strides.clone() {
            spec = spec.with_strides(strides);
        }
        let patch = spec.into_patch();
        let output_shape =
            input_shape.fmt.from_n_c_hw(*input_shape.n(), *input_shape.c(), &*patch.output_shape);
        (input_shape, patch, output_shape)
    }

    pub fn rules_for_shape<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_format.shape(ishape);
            let ones = tvec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for o in 0..outputs.len() {
                for (ix, d) in computed.iter().enumerate() {
                    s.equals(&outputs[o].shape[ix + ishape.h_axis()], &d.output)?;
                }
                s.equals(&outputs[o].shape[ishape.n_axis()], ishape.n_dim())?;
                s.equals(&outputs[o].shape[ishape.c_axis()], ishape.c_dim())?;
            }
            Ok(())
        })
    }

    pub fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let ishape = self.data_format.shape(inputs[0].shape.to_tvec());
        let ones = tvec![1; ishape.hw_rank()];
        let computed = self.padding.compute(
            ishape.hw_dims(),
            &*self.kernel_shape,
            &ones,
            self.strides.as_ref().unwrap_or(&ones),
        );
        let spatial_dims = computed.into_iter().map(|d| d.output).collect::<TVec<TDim>>();
        let oshape =
            self.data_format.from_n_c_hw(ishape.n().clone(), ishape.c().clone(), spatial_dims);
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, &*oshape.shape)?))
    }

    pub fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        let input_shape = self.data_format.shape(&*fact.shape);
        if fact.axis == input_shape.n_axis() {
            let (_input_shape, _patch, output_shape) = self.compute_geo(&*fact.shape);
            fact.shape = output_shape.shape;
            let id = target.chain_after(input, &*node.name, node.op.clone(), tvec!(fact))?;
            Ok(tvec!(OutletId::new(id, 0)))
        } else if fact.axis == input_shape.c_axis() {
            bail!("Can not pulsify cnn pooling ops along the input channel axis");
        } else {
            let geo_axis = fact.axis - input_shape.h_axis();
            let stride = self.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
            if fact.pulse() % stride != 0 {
                bail!("Pulsificaton requires pulse to be a stride multiple")
            }
            let dilation = 1;
            let kernel_len = (self.kernel_shape[geo_axis] - 1) * dilation;

            if kernel_len < stride {
                let misalignment = fact.delay % stride;
                if misalignment != 0 {
                    unimplemented!();
                }
                let mut final_fact = fact.clone();
                final_fact.shape[fact.axis] = final_fact.shape[fact.axis] / stride;
                final_fact.dim = final_fact.dim.clone() / stride;
                final_fact.delay = final_fact.delay / stride;
                let id =
                    target.chain_after(input, &*node.name, node.op.clone(), tvec!(final_fact))?;
                return Ok(tvec!(OutletId::new(id, 0)));
            }

            // overlap case, need delay with augmented output

            let mut augmented_fact = fact.clone();
            augmented_fact.shape[augmented_fact.axis] += kernel_len;
            augmented_fact.delay += kernel_len;

            let mut final_fact = fact.clone();
            final_fact.shape[fact.axis] =
                (augmented_fact.shape[augmented_fact.axis] - kernel_len) / stride;
            final_fact.delay += kernel_len;
            final_fact.dim = (final_fact.dim.clone() - kernel_len.to_dim()) / stride;

            let delay = crate::pulse::delay::Delay::new(fact, 0, kernel_len);
            target.chain_after(
                input,
                format!("{}/Delay", node.name),
                delay,
                tvec!(augmented_fact),
            )?;
            let id = target.chain(&*node.name, node.op.clone(), tvec!(final_fact))?;

            Ok(tvec!(OutletId::new(id, 0)))
        }
    }
}
