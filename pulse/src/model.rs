use std::sync::RwLock;

use crate::fact::StreamInfo;
use crate::{internal::*, ops::sync_inputs};
use tract_core::model::translator::Translate;
use tract_pulse_opl::tract_core::ops::konst::Const;
use tract_pulse_opl::tract_core::ops::source::TypedSource;

pub type PulsedModel = Graph<PulsedFact, Box<dyn PulsedOp>>;
pub type PulsedNode = Node<PulsedFact, Box<dyn PulsedOp>>;

#[allow(clippy::new_ret_no_self)]
pub trait PulsedModelExt {
    fn new(source: &TypedModel, symbol: Symbol, pulse: &TDim) -> TractResult<PulsedModel>;

    fn new_with_mapping(
        source: &TypedModel,
        symbol: Symbol,
        pulse: &TDim,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)>;

    fn into_typed(self) -> TractResult<TypedModel>;
}

impl PulsedModelExt for PulsedModel {
    fn new(source: &TypedModel, symbol: Symbol, pulse: &TDim) -> TractResult<PulsedModel> {
        Ok(PulsedModel::new_with_mapping(source, symbol, pulse)?.0)
    }

    fn new_with_mapping(
        source: &TypedModel,
        symbol: Symbol,
        pulse: &TDim,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)> {
        let pulsifiers = crate::ops::OpPulsifier::inventory();
        Pulsifier(symbol, pulse.to_owned(), pulsifiers).translate_model_with_mappings(source)
    }

    fn into_typed(self) -> TractResult<TypedModel> {
        let mut typed = tract_core::model::translator::IntoTranslator.translate_model(&self)?;
        ensure!(self.input_outlets()?.iter().all(|o| self
            .outlet_fact(*o)
            .unwrap()
            .stream
            .is_some()));
        ensure!(self.output_outlets()?.iter().all(|o| self
            .outlet_fact(*o)
            .unwrap()
            .stream
            .is_some()));
        let delays = tensor1(
            &self
                .output_outlets()?
                .iter()
                .map(|oo| Ok(self.outlet_fact(*oo)?.stream.as_ref().unwrap().delay as _))
                .collect::<TractResult<TVec<i64>>>()?,
        );
        typed.properties.insert("pulse.delay".to_string(), delays.into_arc_tensor());
        let input_axes = tensor1(
            &self
                .input_outlets()?
                .iter()
                .map(|oo| Ok(self.outlet_fact(*oo)?.stream.as_ref().unwrap().axis as _))
                .collect::<TractResult<TVec<i64>>>()?,
        );
        typed.properties.insert("pulse.input_axes".to_string(), input_axes.into_arc_tensor());
        let output_axes = tensor1(
            &self
                .output_outlets()?
                .iter()
                .map(|oo| Ok(self.outlet_fact(*oo)?.stream.as_ref().unwrap().axis as _))
                .collect::<TractResult<TVec<i64>>>()?,
        );
        typed.properties.insert("pulse.output_axes".to_string(), output_axes.into_arc_tensor());
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
            op.pulsed_output_facts(&input_facts)?
        };
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }

    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        for node in &self.nodes {
            if let Some(op) = node.op_as::<Const>() {
                if op.val() == &v {
                    return Ok(node.id.into());
                }
            }
        }
        let op = NonPulsingWrappingOp(Box::new(Const::new(v)?));
        Ok(self.wire_node(name, op, &[])?[0])
    }
}

struct Pulsifier(
    Symbol,
    TDim,
    #[allow(dead_code)] Arc<RwLock<HashMap<TypeId, crate::ops::OpPulsifier>>>,
);

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
        if let Some(op) = node.op_as::<TypedSource>() {
            return Ok(crate::ops::source::pulsify(
                op, source, node, target, mapping, &self.0, &self.1,
            )?
            .unwrap());
        }
        log::debug!("Pulsifying node {node}");

        if !source
            .node_input_facts(node.id)?
            .iter()
            .any(|f| f.shape.iter().any(|d| d.symbols().contains(&self.0)))
            && !node
                .outputs
                .iter()
                .any(|o| o.fact.shape.iter().any(|d| d.symbols().contains(&self.0)))
        {
            let pulse_op = NonPulsingWrappingOp(node.op.clone());
            let inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
            log::debug!("Pulsified node {node} with NonPulsingWrappingOp");
            return target.wire_node(&node.name, pulse_op, &inputs);
        }

        if let Some(pulsified) =
            OpPulsifier::pulsify(source, node, target, mapping, &self.0, &self.1)?
        {
            log::debug!("Pulsified node {node} with adhoc pulsifier");
            return Ok(pulsified);
        }

        let pulse_facts: TVec<PulsedFact> =
            node.inputs.iter().map(|i| target.outlet_fact(mapping[i]).unwrap().clone()).collect();
        if pulse_facts.iter().all(|pf| pf.stream.is_none()) {
            let pulse_op = NonPulsingWrappingOp(node.op.clone());
            let inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
            log::debug!("Pulsified node {node} with NonPulsingWrappingOp");
            return target.wire_node(&node.name, pulse_op, &inputs);
        }

        let (stream_input_ix, pulse_fact) =
            pulse_facts.iter().enumerate().find(|(_ix, pf)| pf.stream.is_some()).unwrap();
        let (input_facts, output_facts) = source.node_facts(node.id)?;
        let axes_mapping = node.op.axes_mapping(&input_facts, &output_facts)?;
        let axis_info = axes_mapping
            .axis((InOut::In(stream_input_ix), pulse_fact.stream.as_ref().unwrap().axis))?;
        if axis_info.outputs[0].len() == 1 {
            let pulse_op = PulseWrappingOp(node.op.clone());
            let inputs = sync_inputs(node, target, mapping)?;
            log::debug!("Pulsified node {node} with PulsingWrappingOp");
            return target.wire_node(&node.name, pulse_op, &inputs);
        }

        bail!("No specific pulse transformation for {}, and could not track pulsing axis.", node)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PulseWrappingOp(pub Box<dyn TypedOp>);

impl Op for PulseWrappingOp {
    fn name(&self) -> Cow<str> {
        format!("PulseWrapping({}", self.0.name()).into()
    }

    fn as_typed(&self) -> Option<&dyn TypedOp> {
        Some(self.0.as_ref())
    }
}

impl EvalOp for PulseWrappingOp {
    fn is_stateless(&self) -> bool {
        self.0.is_stateless()
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.0.eval(inputs)
    }

    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.0.state(session, node_id)
    }
}

impl PulsedOp for PulseWrappingOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let (pulsing_input, stream) = if let Some((ix, fact)) =
            &inputs.iter().enumerate().find(|(_ix, f)| f.stream.is_some())
        {
            (*ix, fact.stream.as_ref().unwrap())
        } else {
            bail!("PulseWrappingOp used on non streaming input")
        };
        let input_facts =
            inputs.iter().map(|pf| pf.to_typed_fact()).collect::<TractResult<TVec<_>>>()?;
        let input_facts_ref = input_facts.iter().map(|f| f.as_ref()).collect::<TVec<_>>();
        let output_facts = self.0.output_facts(&input_facts_ref)?;
        let output_facts_ref = output_facts.iter().collect::<TVec<_>>();
        let axes_mapping = self.0.axes_mapping(&input_facts_ref, &output_facts_ref)?;
        let axis_info = axes_mapping.axis((InOut::In(pulsing_input), stream.axis))?;
        std::mem::drop(output_facts_ref);
        output_facts
            .into_iter()
            .enumerate()
            .map(|(ix, tf)| {
                if let &[axis] = &*axis_info.outputs[ix] {
                    Ok(PulsedFact {
                        shape: tf.shape,
                        datum_type: tf.datum_type,
                        stream: Some(StreamInfo {
                            delay: stream.delay,
                            axis,
                            dim: stream.dim.clone(),
                        }),
                    })
                } else {
                    bail!("Disappearing pulsing axis")
                }
            })
            .collect()
    }

    as_op!();

    fn to_typed(&self) -> Box<dyn TypedOp> {
        self.0.clone()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NonPulsingWrappingOp(pub Box<dyn TypedOp>);

impl Op for NonPulsingWrappingOp {
    fn name(&self) -> Cow<str> {
        format!("NonePulsingWrapping({}", self.0.name()).into()
    }

    fn as_typed(&self) -> Option<&dyn TypedOp> {
        Some(self.0.as_ref())
    }
}

impl EvalOp for NonPulsingWrappingOp {
    fn is_stateless(&self) -> bool {
        self.0.is_stateless()
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.0.eval(inputs)
    }

    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.0.state(session, node_id)
    }
}

impl PulsedOp for NonPulsingWrappingOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let input_facts =
            inputs.iter().map(|pf| pf.to_typed_fact()).collect::<TractResult<TVec<_>>>()?;
        let input_facts_ref = input_facts.iter().map(|f| f.as_ref()).collect::<TVec<_>>();
        let output_facts = self.0.output_facts(&input_facts_ref)?;
        let output_facts_ref = output_facts.iter().collect::<TVec<_>>();
        std::mem::drop(output_facts_ref);
        output_facts
            .into_iter()
            .map(|tf| Ok(PulsedFact { shape: tf.shape, datum_type: tf.datum_type, stream: None }))
            .collect()
    }

    as_op!();

    fn to_typed(&self) -> Box<dyn TypedOp> {
        self.0.clone()
    }
}
