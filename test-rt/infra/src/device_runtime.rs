use std::fmt::Debug;
use std::sync::Arc;
use tract_core::internal::*;
use tract_core::ops::change_axes::perm_to_ops;
use tract_core::runtime::{RunOptions, Runnable, Runtime, State};

/// What one device backend contributes to [`DeviceTestRuntime`]: the transform
/// that moves a model onto the device, and the turn handler its arena flavour
/// installs on the plan.
pub trait DeviceTestBackend: Debug + Send + Sync + 'static {
    fn transform(&self, model: &mut TypedModel) -> TractResult<()>;

    fn with_arena(
        &self,
        plan: TypedSimplePlan,
        memory_hint: &SymbolValues,
    ) -> TractResult<TypedSimplePlan>;

    fn check(&self) -> TractResult<()> {
        Ok(())
    }
}

/// Runs a suite against a device backend, optionally with every input and
/// output axis-reversed (which exercises the backends' axis handling) and with
/// the arena turn handler installed.
#[derive(Debug)]
pub struct DeviceTestRuntime<B: DeviceTestBackend> {
    pub name: &'static str,
    pub backend: B,
    pub optimize: bool,
    pub transpose_inputs: bool,
    pub use_arena: bool,
}

impl<B: DeviceTestBackend> Runtime for DeviceTestRuntime<B> {
    fn name(&self) -> StaticName {
        self.name.into()
    }

    fn prepare_with_options(
        &self,
        mut model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        if self.transpose_inputs {
            transpose_interface(&mut model)?;
        }
        self.backend.transform(&mut model)?;
        if self.optimize {
            model = model.into_optimized()?;
        }
        let mut plan = Arc::unwrap_or_clone(model.into_runnable_with_options(options)?);
        if self.use_arena {
            let memory_hint = options.memory_sizing_hints.clone().unwrap_or_default();
            plan = self.backend.with_arena(plan, &memory_hint)?;
        }
        let runnable = Arc::new(plan);
        let facts =
            if self.transpose_inputs { Some(reversed_interface_facts(&runnable)?) } else { None };
        Ok(Box::new(DeviceTestRunnable {
            runnable,
            transpose_inputs: self.transpose_inputs,
            facts,
        }))
    }

    fn check(&self) -> TractResult<()> {
        self.backend.check()
    }
}

fn transpose_interface(model: &mut TypedModel) -> TractResult<()> {
    for ix in 0..model.inputs.len() {
        let input = model.input_outlets()?[ix];
        let in_fact = model.outlet_fact(input)?;
        let rank = in_fact.rank();
        let shape = in_fact.shape.dims().iter().rev().collect::<TVec<_>>();
        let fact = in_fact.datum_type.fact(shape);

        let transposed_input = model.add_source(format!("transposed_input_{ix}"), fact)?;

        let mut patch = TypedModelPatch::default();
        let mut wire = patch.tap_model(model, transposed_input)?;
        for (ax, op) in reverse_axes(rank).into_iter().enumerate() {
            wire = patch.wire_node(format!("transposed_input.{ix}_{ax}"), op, &[wire])?[0];
        }
        patch.shunt_outside(model, input, wire)?;
        patch.apply(model)?;
    }

    for _ in 0..model.inputs.len() / 2 {
        let input = model.inputs.remove(0);
        model.node_mut(input.node).op = model.create_dummy();
    }

    for (ix, output) in model.outputs.clone().iter().enumerate() {
        let rank = model.outlet_fact(*output)?.rank();
        let mut wire = *output;
        for (ax, op) in reverse_axes(rank).into_iter().enumerate() {
            wire = model.wire_node(format!("transposed_output.{ix}_{ax}"), op, &[wire])?[0];
        }
        model.outputs[ix] = wire;
    }
    Ok(())
}

fn reverse_axes(rank: usize) -> TVec<AxisOp> {
    perm_to_ops(&(0..rank).rev().collect::<TVec<usize>>())
}

fn transpose_tensors(values: TVec<TValue>) -> TractResult<TVec<TValue>> {
    values
        .into_iter()
        .map(|t| {
            let t = t.into_tensor();
            let perms: TVec<usize> = (0..t.rank()).rev().collect();
            Ok(t.permute_axes(&perms)?.into_tvalue())
        })
        .collect()
}

/// The interface facts callers see, axis-reversed back from the transposed
/// model's own: [`DeviceTestState`] transposes what it is handed, so the facts
/// a caller must feed are the reverse of the inner model's.
#[derive(Debug)]
struct InterfaceFacts {
    inputs: TVec<TypedFact>,
    outputs: TVec<TypedFact>,
}

fn reversed_interface_facts(runnable: &Arc<TypedRunnableModel>) -> TractResult<InterfaceFacts> {
    let reverse = |fact: &TypedFact| {
        let shape = fact.shape.dims().iter().rev().collect::<TVec<_>>();
        fact.datum_type.fact(shape)
    };
    Ok(InterfaceFacts {
        inputs: (0..runnable.input_count())
            .map(|ix| runnable.input_fact(ix).map(reverse))
            .collect::<TractResult<_>>()?,
        outputs: (0..runnable.output_count())
            .map(|ix| runnable.output_fact(ix).map(reverse))
            .collect::<TractResult<_>>()?,
    })
}

#[derive(Debug)]
struct DeviceTestRunnable {
    runnable: Arc<TypedRunnableModel>,
    transpose_inputs: bool,
    facts: Option<InterfaceFacts>,
}

impl Runnable for DeviceTestRunnable {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(DeviceTestState {
            state: self.runnable.spawn()?,
            transpose_inputs: self.transpose_inputs,
        }))
    }

    fn input_count(&self) -> usize {
        self.runnable.input_count()
    }

    fn output_count(&self) -> usize {
        self.runnable.output_count()
    }

    fn input_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        match &self.facts {
            Some(facts) => Ok(&facts.inputs[ix]),
            None => self.runnable.input_fact(ix),
        }
    }

    fn output_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        match &self.facts {
            Some(facts) => Ok(&facts.outputs[ix]),
            None => self.runnable.output_fact(ix),
        }
    }

    fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>> {
        self.runnable.typed_plan()
    }

    fn typed_model(&self) -> Option<&Arc<TypedModel>> {
        self.runnable.typed_model()
    }
}

#[derive(Clone, Debug)]
struct DeviceTestState {
    state: TypedSimpleState,
    transpose_inputs: bool,
}

impl State for DeviceTestState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        if !self.transpose_inputs {
            return self.state.run(inputs);
        }
        let outputs = self.state.run(transpose_tensors(inputs)?)?;
        transpose_tensors(outputs)
    }

    fn resolve_symbol(&mut self, symbol: &Symbol, value: i64) -> TractResult<()> {
        self.state.resolve_symbol(symbol, value)
    }

    fn seat(&mut self, seating: Seating) -> TractResult<()> {
        self.state.seat(seating);
        Ok(())
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        self.state.reset_lanes(lanes)
    }

    fn input_count(&self) -> usize {
        self.state.input_count()
    }

    fn output_count(&self) -> usize {
        self.state.output_count()
    }

    fn runnable(&self) -> &dyn Runnable {
        self.state.runnable()
    }
}
