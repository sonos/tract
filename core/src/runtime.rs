use std::fmt::Debug;

use crate::internal::*;

pub trait Runtime: Debug + Send + Sync + 'static {
    fn name(&self) -> StaticName;
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        self.prepare_with_options(model, &Default::default())
    }
    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &PlanOptions,
    ) -> TractResult<Box<dyn Runnable>>;
}

pub trait Runnable: Debug {
    fn run(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.spawn()?.run(inputs)
    }
    fn typed_model(&self) -> Option<&Arc<TypedModel>>;
    fn spawn(&self) -> TractResult<Box<dyn State>>;
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
}

pub trait State {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>>;

    fn runnable(&self) -> &dyn Runnable {
        panic!();
    }

    fn initializable_states_count(&self) -> usize;
    fn get_states_facts(&self) -> Vec<TypedFact>;
    fn init_state(&mut self, states: &[TValue]) -> TractResult<()>;
    fn get_states(&self) -> TractResult<Vec<TValue>>;
    fn input_count(&self) -> usize {
        self.runnable().input_count()
    }

    fn output_count(&self) -> usize {
        self.runnable().input_count()
    }
}

#[derive(Debug)]
pub struct DefaultRuntime;

impl Runtime for DefaultRuntime {
    fn name(&self) -> StaticName {
        Cow::Borrowed("default")
    }

    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &PlanOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        let model = model.into_optimized()?;
        Ok(Box::new(TypedSimplePlan::new_with_options(model, options)?))
    }
}

impl Runnable for Arc<TypedRunnableModel> {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(self.spawn()?))
    }

    fn typed_model(&self) -> Option<&Arc<TypedModel>> {
        Some(&self.model)
    }

    fn input_count(&self) -> usize {
        self.model.inputs.len()
    }

    fn output_count(&self) -> usize {
        self.model.outputs.len()
    }
}

impl State for TypedSimpleState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.run(inputs)
    }

    fn runnable(&self) -> &dyn Runnable {
        &self.plan
    }

    fn initializable_states_count(&self) -> usize {
        self.states
            .iter()
            .filter_map(Option::as_ref)
            .filter(|s| s.init_tensor_fact().is_some())
            .count()
    }

    fn get_states_facts(&self) -> Vec<TypedFact> {
        self.states
            .iter()
            .filter_map(|s| s.as_ref().and_then(|s| s.init_tensor_fact().map(|(_, fact)| fact)))
            .collect()
    }

    fn init_state(&mut self, states: &[TValue]) -> TractResult<()> {
        self.init_states(states)
    }

    fn get_states(&self) -> TractResult<Vec<TValue>> {
        let mut states = vec![];
        for op_state in self.states.iter().flatten() {
            if op_state.init_tensor_fact().is_some() {
                op_state.save_to(&mut states)?;
            }
        }
        Ok(states)
    }
}

pub struct InventorizedRuntime(pub &'static dyn Runtime);

impl Runtime for InventorizedRuntime {
    fn name(&self) -> StaticName {
        self.0.name()
    }

    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &PlanOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        self.0.prepare_with_options(model, options)
    }
}

impl Debug for InventorizedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

inventory::collect!(InventorizedRuntime);

pub fn runtimes() -> impl Iterator<Item = &'static dyn Runtime> {
    inventory::iter::<InventorizedRuntime>().map(|ir| ir.0)
}

pub fn runtime_for_name(s: &str) -> Option<&'static dyn Runtime> {
    runtimes().find(|rt| rt.name() == s)
}

#[macro_export]
macro_rules! register_runtime {
    ($type: ty= $val:expr) => {
        static D: $type = $val;
        inventory::submit! { $crate::runtime::InventorizedRuntime(&D) }
    };
}

register_runtime!(DefaultRuntime = DefaultRuntime);
