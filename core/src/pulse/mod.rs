use crate::internal::*;
use crate::model::translator::Translate;
use std::fmt;

pub mod delay;

#[derive(Clone, PartialEq)]
pub struct PulsedFact {
    pub datum_type: DatumType,
    pub shape: TVec<usize>,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
}

impl fmt::Debug for PulsedFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(
            fmt,
            "{}x{:?} [pulse axis:{} ∂:{} full dim:{:?}]",
            self.shape.iter().join("x"),
            self.datum_type,
            self.axis,
            self.delay,
            self.dim
        )
    }
}

impl Fact for PulsedFact {
    fn to_typed_fact(&self) -> TractResult<TypedFact> {
        Ok(self.into())
    }
}

impl<'a> From<&'a PulsedFact> for TypedFact {
    fn from(fact: &'a PulsedFact) -> TypedFact {
        TypedFact::dt_shape(fact.datum_type, &*fact.shape).unwrap()
    }
}

impl<'a> From<&'a Box<dyn PulsedOp>> for Box<dyn TypedOp> {
    fn from(op: &'a Box<dyn PulsedOp>) -> Box<dyn TypedOp> {
        op.to_typed()
    }
}

impl PulsedFact {
    pub fn from_tensor_fact_pulse(tf: &NormalizedFact, pulse: usize) -> TractResult<PulsedFact> {
        let datum_type = tf.datum_type;
        let stream =
            tf.shape.stream_info.as_ref().ok_or("Can not pulse a tensor with no streaming dim")?;
        let shape =
            tf.shape.iter().map(|d| d.to_integer().map(|d| d as usize).unwrap_or(pulse)).collect();
        Ok(PulsedFact { datum_type, shape, axis: stream.axis, dim: stream.len.clone(), delay: 0 })
    }

    pub fn pulse(&self) -> usize {
        self.shape[self.axis]
    }

    pub fn to_pulse_fact(&self) -> NormalizedFact {
        NormalizedFact::dt_shape(self.datum_type, &*self.shape).unwrap()
    }

    pub fn streaming_shape(&self) -> Vec<TDim> {
        self.shape
            .iter()
            .enumerate()
            .map(|(ix, &d)| if ix == self.axis { self.dim.clone() } else { d.to_dim() })
            .collect()
    }

    pub fn to_streaming_fact(&self) -> NormalizedFact {
        let mut info = self.to_pulse_fact();
        info.shape.stream_info = Some(StreamFact { axis: self.axis, len: self.dim.clone() });
        info
    }
}

pub type PulsedModel = ModelImpl<PulsedFact, Box<dyn PulsedOp>>;
pub type PulsedNode = BaseNode<PulsedFact, Box<dyn PulsedOp>>;

impl PulsedModel {
    pub fn new(source: &NormalizedModel, pulse: usize) -> TractResult<PulsedModel> {
        Ok(PulsedModel::new_with_mapping(source, pulse)?.0)
    }

    pub fn new_with_mapping(
        source: &NormalizedModel,
        pulse: usize,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)> {
        Pulsifier(pulse).translate_model_with_mappings(source)
    }

    pub fn into_typed(self) -> TractResult<TypedModel> {
        crate::model::translator::IntoTranslator.translate_model(&self)
    }
}

#[derive(Debug)]
struct Pulsifier(usize);
impl
    crate::model::translator::Translate<
        NormalizedFact,
        Box<dyn TypedOp>,
        crate::pulse::PulsedFact,
        Box<dyn PulsedOp>,
    > for Pulsifier
{
    fn translate_node(
        &self,
        source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        node.op.pulsify(source, node, target, mapping, self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_must_stream() {
        let mut model = TypedModel::default();
        let _a = model
            .add_source("a", TypedFact::dt_shape(f32::datum_type(), [1, 2, 3].as_ref()).unwrap())
            .unwrap();
        model.auto_outputs().unwrap();
        assert!(PulsedModel::new(&model.into_normalized().unwrap(), 4).is_err());

        let mut model = TypedModel::default();
        let _a = model
            .add_source(
                "a",
                TypedFact::dt_shape(
                    f32::datum_type(),
                    [1.to_dim(), TDim::s(), 3.to_dim()].as_ref(),
                )
                .unwrap(),
            )
            .unwrap();
        model.auto_outputs().unwrap();
        let pulse = PulsedModel::new(&model.into_normalized().unwrap(), 4).unwrap();
        assert_eq!(
            pulse.outlet_fact(OutletId::new(0, 0)).unwrap().to_typed_fact().unwrap(),
            TypedFact::dt_shape(DatumType::F32, [1usize, 4, 3].as_ref()).unwrap()
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = TypedModel::default();
        let _a = model
            .add_source(
                "a",
                TypedFact::dt_shape(
                    f32::datum_type(),
                    [TDim::s(), 2.to_dim(), 3.to_dim()].as_ref(),
                )
                .unwrap(),
            )
            .unwrap();
        model.auto_outputs().unwrap();

        let pulse = PulsedModel::new(&model.into_normalized().unwrap(), 4).unwrap();

        assert_eq!(
            pulse.input_fact(0).unwrap().to_typed_fact().unwrap(),
            TypedFact::dt_shape(DatumType::F32, &*vec!(4, 2, 3)).unwrap()
        );
        assert_eq!(
            pulse.output_fact(0).unwrap().to_typed_fact().unwrap(),
            TypedFact::dt_shape(DatumType::F32, &*vec!(4, 2, 3)).unwrap()
        );
    }
}
