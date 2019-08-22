use crate::internal::*;
use crate::ops::source::Source;
use std::fmt;

use std::convert::TryFrom;

pub mod delay;

#[derive(Clone, PartialEq)]
pub struct PulsedTensorFact {
    pub dt: DatumType,
    pub shape: TVec<usize>,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
}

impl fmt::Debug for PulsedTensorFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(
            fmt,
            "{}x{:?} [pulse axis:{} ∂:{} full dim:{:?}]",
            self.shape.iter().join("x"),
            self.dt,
            self.axis,
            self.delay,
            self.dim
        )
    }
}

impl TensorInfo for PulsedTensorFact {
    fn to_tensor_fact(&self) -> TensorFact {
        TensorFact::dt_shape(self.dt, &self.shape)
    }
}

impl TryFrom<PulsedTensorFact> for TypedTensorInfo {
    type Error = TractError;
    fn try_from(fact: PulsedTensorFact) -> TractResult<TypedTensorInfo> {
        TypedTensorInfo::dt_shape(fact.dt, &*fact.shape)
    }
}

impl PulsedTensorFact {
    pub fn from_tensor_fact_pulse(
        tf: &NormalizedTensorInfo,
        pulse: usize,
    ) -> TractResult<PulsedTensorFact> {
        let dt = tf.datum_type;
        let stream =
            tf.shape.stream_info.as_ref().ok_or("Can not pulse a tensor with no streaming dim")?;
        let shape =
            tf.shape.iter().map(|d| d.to_integer().map(|d| d as usize).unwrap_or(pulse)).collect();
        Ok(PulsedTensorFact { dt, shape, axis: stream.axis, dim: stream.len.clone(), delay: 0 })
    }

    pub fn pulse(&self) -> usize {
        self.shape[self.axis]
    }

    pub fn to_pulse_fact(&self) -> NormalizedTensorInfo {
        NormalizedTensorInfo::dt_shape(self.dt, &*self.shape).unwrap()
    }

    pub fn streaming_shape(&self) -> Vec<TDim> {
        self.shape
            .iter()
            .enumerate()
            .map(|(ix, &d)| if ix == self.axis { self.dim.clone() } else { d.to_dim() })
            .collect()
    }

    pub fn to_streaming_fact(&self) -> NormalizedTensorInfo {
        let mut info = self.to_pulse_fact();
        info.shape.stream_info = Some(StreamInfo { axis: self.axis, len: self.dim.clone() });
        info
    }
}

pub type PulsedModel = ModelImpl<PulsedTensorFact, Box<dyn TypedOp>>;

impl PulsedModel {
    pub fn new(source: &NormalizedModel, pulse: usize) -> TractResult<PulsedModel> {
        Ok(PulsedModel::new_with_mapping(source, pulse)?.0)
    }

    pub fn new_with_mapping(
        source: &NormalizedModel,
        pulse: usize,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)> {
        let mut target = PulsedModel::default();
        let mut mapping = HashMap::new();
        for old_id in source.eval_order()? {
            trace!(
                "Pulsify node {} {} ({})",
                old_id,
                source.node(old_id).name,
                source.node(old_id).op().name()
            );
            if source.node(old_id).op_as::<Source>().is_some() {
                let node = source.node(old_id);
                let pulsed_fact =
                    PulsedTensorFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
                let id = target.add_source(node.name.clone(), pulsed_fact)?;
                mapping.insert(OutletId::new(old_id, 0), OutletId::new(id, 0));
            } else {
                let node = &source.nodes()[old_id];
                let outlets = node
                    .op
                    .pulsify(&source, node, &mut target, &mapping)
                    .chain_err(|| format!("Pulsifying {:?}", node))?;
                for (ix, outlet) in outlets.into_iter().enumerate() {
                    mapping.insert(OutletId::new(node.id, ix), outlet);
                }
            }
            trace!("Target is now {}", target.nodes().len());
        }
        // maintaining order of i/o interface
        target.inputs = source.input_outlets()?.iter().map(|i| mapping[&i]).collect();
        target.outputs = source.output_outlets()?.iter().map(|o| mapping[&o]).collect();
        Ok((target, mapping))
    }

    pub fn into_typed(self) -> TractResult<TypedModel> {
        crate::model::compact::compact(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_must_stream() {
        let mut model = InferenceModel::default();
        let _a =
            model.add_source("a", TensorFact::dt_shape(DatumType::F32, vec![1, 2, 3])).unwrap();
        model.auto_outputs().unwrap();
        assert!(
            PulsedModel::new(&model.into_typed().unwrap().into_normalized().unwrap(), 4).is_err()
        );

        let mut model = InferenceModel::default();
        let _a = model
            .add_source(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![1.to_dim(), TDim::s(), 3.to_dim()]),
            )
            .unwrap();
        model.auto_outputs().unwrap();
        let pulse =
            PulsedModel::new(&model.into_typed().unwrap().into_normalized().unwrap(), 4).unwrap();
        assert_eq!(
            pulse.outlet_fact(OutletId::new(0, 0)).unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(1, 4, 3))
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = InferenceModel::default();
        let _a = model
            .add_source(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![TDim::s(), 2.to_dim(), 3.to_dim()]),
            )
            .unwrap();
        model.auto_outputs().unwrap();

        let pulse = PulsedModel::new(&model.into_normalized().unwrap(), 4).unwrap();

        assert_eq!(
            pulse.input_fact(0).unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
        assert_eq!(
            pulse.output_fact(0).unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
    }

}
