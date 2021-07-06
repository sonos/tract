use crate::internal::*;
use tract_core::model::translator::Translate;

pub type PulsedModel = Graph<PulsedFact, Box<dyn PulsedOp>>;
pub type PulsedNode = Node<PulsedFact, Box<dyn PulsedOp>>;

pub trait PulsedModelExt {
    fn new(source: &TypedModel, pulse: usize) -> TractResult<PulsedModel>;

    fn new_with_mapping(
        source: &TypedModel,
        pulse: usize,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)>;

    fn into_typed(self) -> TractResult<TypedModel>;
}

impl PulsedModelExt for PulsedModel {
    fn new(source: &TypedModel, pulse: usize) -> TractResult<PulsedModel> {
        Ok(PulsedModel::new_with_mapping(source, pulse)?.0)
    }

    fn new_with_mapping(
        source: &TypedModel,
        pulse: usize,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)> {
        let pulsifiers = crate::ops::OpPulsifier::inventory();
        Pulsifier(pulse, pulsifiers).translate_model_with_mappings(source)
    }

    fn into_typed(self) -> TractResult<TypedModel> {
        let mut typed = tract_core::model::translator::IntoTranslator.translate_model(&self)?;
        let delays = tensor1(
            &self
                .output_outlets()?
                .iter()
                .map(|oo| Ok(self.outlet_fact(*oo)?.delay as _))
                .collect::<TractResult<TVec<i64>>>()?,
        );
        typed.properties.insert("pulse.delay".to_string(), delays.into_arc_tensor());
        Ok(typed)
    }
}

impl SpecialOps<PulsedFact, Box<dyn PulsedOp>> for PulsedModel {
    fn is_source(op: &Box<dyn PulsedOp>) -> bool {
        op.as_op().downcast_ref::<crate::ops::source::PulsedSource>().is_some()
    }

    fn create_source(&self, fact: PulsedFact) -> Box<dyn PulsedOp> {
        Box::new(crate::ops::source::PulsedSource(fact))
    }

    fn create_dummy(&self) -> Box<dyn PulsedOp> {
        Box::new(tract_core::ops::dummy::Dummy::new())
    }

    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn PulsedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let output_facts = {
            let input_facts =
                inputs.iter().map(|o| self.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            op.pulsed_output_facts(&*input_facts)?
        };
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }
}

struct Pulsifier(usize, HashMap<TypeId, crate::ops::OpPulsifier>);

impl std::fmt::Debug for Pulsifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pulsifier({})", self.0)
    }
}

impl
    tract_core::model::translator::Translate<
        TypedFact,
        Box<dyn TypedOp>,
        PulsedFact,
        Box<dyn PulsedOp>,
    > for Pulsifier
{
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(pulsifier) = self.1.get(&node.op.type_id()) {
            (pulsifier.func)(source, node, target, mapping, self.0)
        } else {
            bail!("No pulsifier for {}", node);
        }
    }
}


#[derive(Debug, Clone)]
struct PulseWrappingOp(Box<dyn TypedOp>);

impl_dyn_hash!(PulseWrappingOp);

impl Op for PulseWrappingOp {
    fn name(&self) -> Cow<str> {
        format!("PulseWrapping({}", self.0.name()).into()
    }

    fn as_typed(&self) -> Option<&dyn TypedOp> {
        Some(self.0.as_ref())
    }

    op_pulse!();
}

impl EvalOp for PulseWrappingOp {
    fn is_stateless(&self) -> bool {
        self.0.is_stateless()
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.0.eval(inputs)
    }

    fn state(&self, session: &mut SessionState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        self.0.state(session, node_id)
    }
}

impl PulsedOp for PulseWrappingOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let input_stream_axis = inputs[0].axis;
        let self.0.
    }

    as_op!();
}
