use tract_nnef::internal::*;
use tract_nnef::tract_core::trivial_op_state_freeeze;
use tract_pulse::model::PulsedModel;
use tract_pulse::ops::OpPulsifier;
use tract_pulse::PulsedOp;
use tract_pulse::{internal::*, pulsed_op_to_typed_op};

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_extra_exp_unit_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("state"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("alpha"),
            TypeName::Integer.named("skip").default(0),
            TypeName::Logical.named("stateless").default(false),
            TypeName::Scalar.named("epsilon").default(1e-14f32),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_eun,
    );
    registry.register_dumper(ser_eun);

    OpPulsifier::register::<ExpUnitNorm>(pulsify).unwrap();
}

fn de_eun(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let state = invocation.named_arg_as(builder, "state")?;
    let axis = invocation.named_arg_as::<i64>(builder, "axis")? as usize;
    let alpha = invocation.named_arg_as(builder, "alpha")?;
    let epsilon = invocation.named_arg_as(builder, "epsilon")?;
    let stateless = invocation.named_arg_as::<bool>(builder, "stateless")?;
    let skip = invocation.named_arg_as::<i64>(builder, "skip")? as usize;
    let op = ExpUnitNorm { alpha, axis, epsilon, stateless, skip };
    builder.wire(op, &[wire, state])
}

fn ser_eun(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ExpUnitNorm,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let state = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_extra_exp_unit_norm",
        &[input, state],
        &[
            ("axis", numeric(op.axis)),
            ("alpha", numeric(op.alpha)),
            ("epsilon", numeric(op.epsilon)),
            ("stateless", logical(op.stateless)),
            ("skip", numeric(op.skip)),
        ],
    )))
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpUnitNorm {
    pub alpha: f32,
    pub epsilon: f32,
    pub axis: usize,
    pub skip: usize,
    pub stateless: bool,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct ExpUnitNormState {
    hidden: Option<Tensor>,
    index: usize,
}
trivial_op_state_freeeze!(ExpUnitNormState);

impl Op for ExpUnitNorm {
    fn name(&self) -> Cow<str> {
        "ExpUnitNorm".into()
    }

    op_as_typed_op!();
}

impl EvalOp for ExpUnitNorm {
    fn is_stateless(&self) -> bool {
        self.stateless
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<ExpUnitNormState>::default()))
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ExpUnitNormState::default().eval(self, inputs)
    }
}

impl ExpUnitNormState {
    fn eval(&mut self, op: &ExpUnitNorm, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, state0) = args_2!(inputs);
        let mut input = input.into_tensor();
        let mut view = input.to_array_view_mut::<f32>()?;
        if self.hidden.is_none() || op.stateless {
            self.hidden = Some(state0.into_tensor());
        }
        let mut state = self.hidden.as_mut().unwrap().to_array_view_mut::<f32>()?;
        for mut time_slice in view.axis_iter_mut(tract_ndarray::Axis(op.axis)) {
            if self.index >= op.skip {
                state.zip_mut_with(&time_slice, |s: &mut f32, x: &f32| {
                    *s = x.max(op.epsilon) * (1f32 - op.alpha) + *s * op.alpha;
                });
            }
            time_slice.zip_mut_with(&state, |x, s| *x /= s.sqrt());
            self.index += 1;
        }
        Ok(tvec!(input.into_tvalue()))
    }
}

impl OpState for ExpUnitNormState {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<ExpUnitNorm>().context("Wrong op")?;
        Self::eval(self, op, inputs)
    }
}

impl TypedOp for ExpUnitNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut state_shape = inputs[0].shape.clone();
        let _ = state_shape.remove_axis(self.axis);
        ensure!(inputs[1].without_value() == inputs[0].datum_type.fact(state_shape));
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}

impl PulsedOp for ExpUnitNorm {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

fn pulsify(
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let op = node.op_as::<ExpUnitNorm>().unwrap();
    let (input, state0) = (mapping[&node.inputs[0]], mapping[&node.inputs[1]]);
    let input_fact = target.outlet_fact(input)?;
    let pulsing_input_axis = input_fact
        .to_streaming_fact()
        .shape
        .iter()
        .position(|dim| dim.symbols().contains(symbol))
        .context("No pulsing axis found")?;
    if pulsing_input_axis == op.axis {
        let pulsed_op =
            ExpUnitNorm { skip: input_fact.stream.as_ref().unwrap().delay, ..op.clone() };
        target.wire_node(&node.name, pulsed_op, &[input, state0]).map(Some)
    } else {
        target.wire_node(&node.name, op.clone(), &[input, state0]).map(Some)
    }
}
