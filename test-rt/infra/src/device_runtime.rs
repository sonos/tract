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

    fn with_arena(&self, plan: TypedSimplePlan) -> TractResult<TypedSimplePlan>;

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
            plan = self.backend.with_arena(plan)?;
        }
        Ok(Box::new(DeviceTestRunnable {
            runnable: Arc::new(plan),
            transpose_inputs: self.transpose_inputs,
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

#[derive(Debug)]
struct DeviceTestRunnable {
    runnable: Arc<TypedRunnableModel>,
    transpose_inputs: bool,
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
