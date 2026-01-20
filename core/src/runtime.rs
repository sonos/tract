use std::any::Any;
use std::fmt::Debug;

use downcast_rs::Downcast;
use dyn_clone::DynClone;
use tract_linalg::multithread::Executor;

use crate::internal::*;

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Use the simple ordering instead of the newer memory friendly one
    pub skip_order_opt_ram: bool,

    /// Override default global executor
    pub executor: Option<Executor>,

    /// Memory arena hint value
    pub memory_sizing_hints: SymbolValues,
}

pub trait Runtime: Debug + Send + Sync + 'static {
    fn name(&self) -> StaticName;
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        self.prepare_with_options(model, &Default::default())
    }
    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>>;
}

pub trait Runnable: Any + Downcast + Debug + Send + Sync + 'static {
    fn run(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.spawn()?.run(inputs)
    }
    fn spawn(&self) -> TractResult<Box<dyn State>>;
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;

    fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>>;
    fn typed_model(&self) -> Option<&Arc<TypedModel>>;
}
impl_downcast!(Runnable);

pub trait State: Any + Downcast + Debug + 'static {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>>;

    fn runnable(&self) -> &dyn Runnable;

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

    fn freeze(&self) -> Box<dyn FrozenState>;
}
impl_downcast!(State);

pub trait FrozenState: Any + Debug + DynClone + Send {
    fn unfreeze(&self) -> Box<dyn State>;
}
dyn_clone::clone_trait_object!(FrozenState);

#[derive(Debug)]
pub struct DefaultRuntime;

impl Runtime for DefaultRuntime {
    fn name(&self) -> StaticName {
        Cow::Borrowed("default")
    }

    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        let model = model.into_optimized()?;
        Ok(Box::new(TypedSimplePlan::new_with_options(model, options)?))
    }
}

impl Runnable for Arc<TypedRunnableModel> {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(self.spawn()?))
    }

    fn typed_plan(&self) -> Option<&Self> {
        Some(self)
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

    fn freeze(&self) -> Box<dyn FrozenState> {
        Box::new(TypedSimpleState::freeze(self))
    }
}

impl FrozenState for TypedFrozenSimpleState {
    fn unfreeze(&self) -> Box<dyn State> {
        Box::new(TypedFrozenSimpleState::unfreeze(self))
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
        options: &RunOptions,
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
