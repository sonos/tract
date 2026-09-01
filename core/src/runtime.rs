use std::any::Any;
use std::fmt::Debug;

use downcast_rs::Downcast;
use dyn_clone::DynClone;
use lazy_static::lazy_static;
use tract_linalg::multithread::Executor;

use crate::internal::*;

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Use the simple ordering instead of the newer memory friendly one
    pub skip_order_opt_ram: bool,

    /// Override default global executor
    pub executor: Option<Executor>,

    /// Memory sizing hints
    pub memory_sizing_hints: Option<SymbolValues>,
}

pub trait Runtime: Debug + Send + Sync + 'static {
    fn name(&self) -> StaticName;
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        self.prepare_with_options(model, &Default::default())
    }
    fn check(&self) -> TractResult<()>;
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
    fn input_count(&self) -> usize {
        self.typed_model().context("Fallback implementation on typed_model()").unwrap().inputs.len()
    }
    fn output_count(&self) -> usize {
        self.typed_model()
            .context("Fallback implementation on typed_model()")
            .unwrap()
            .outputs
            .len()
    }
    fn input_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.typed_model()
            .context("Fallback implementation on typed_model()")
            .unwrap()
            .input_fact(ix)
    }
    fn output_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.typed_model()
            .context("Fallback implementation on typed_model()")
            .unwrap()
            .output_fact(ix)
    }
    fn properties(&self) -> &HashMap<String, Arc<Tensor>> {
        lazy_static! {
            static ref NO_PROPERTIES: HashMap<String, Arc<Tensor>> = Default::default();
        };
        self.typed_model().map(|model| &model.properties).unwrap_or(&NO_PROPERTIES)
    }

    fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>>;
    fn typed_model(&self) -> Option<&Arc<TypedModel>>;
}
impl_downcast!(Runnable);

pub trait State: Any + Downcast + Debug + Send + DynClone + 'static {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>>;

    /// Pin a symbol for the coming turn. A pulsed plan needs the stream length
    /// on the turn carrying the last, partial pulse: the input tensor is padded
    /// to a full pulse, so its shape can not carry it.
    fn resolve_symbol(&mut self, symbol: &Symbol, value: i64) -> TractResult<()> {
        let _ = (symbol, value);
        bail!("{self:?} can not resolve a symbol")
    }

    /// Seat the lanes carrying the coming turn's streams, one lane per row of
    /// axis 0 of its tensors. A turn seating more than one lane needs axis 0 of
    /// every stateful node to be the model's batch axis.
    fn seat(&mut self, seating: Seating) -> TractResult<()> {
        let _ = seating;
        bail!("{self:?} can not seat lanes")
    }

    /// Drop the session state `lanes` hold, handing them to new streams.
    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        let _ = lanes;
        bail!("{self:?} can not reset lanes")
    }

    fn runnable(&self) -> &dyn Runnable;

    fn input_count(&self) -> usize {
        self.runnable().input_count()
    }

    fn output_count(&self) -> usize {
        self.runnable().output_count()
    }
}
impl_downcast!(State);
dyn_clone::clone_trait_object!(State);

#[derive(Debug)]
pub struct DefaultRuntime;

impl Runtime for DefaultRuntime {
    fn name(&self) -> StaticName {
        Cow::Borrowed("cpu")
    }

    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        let model = model.into_optimized()?;
        Ok(Box::new(TypedSimplePlan::new_with_options(model, options)?))
    }

    fn check(&self) -> TractResult<()> {
        Ok(())
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

    fn input_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.model.input_fact(ix)
    }
    fn output_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.model.output_fact(ix)
    }
}

impl State for TypedSimpleState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.run(inputs)
    }

    fn resolve_symbol(&mut self, symbol: &Symbol, value: i64) -> TractResult<()> {
        self.turn_state.resolved_symbols.set(symbol, value);
        Ok(())
    }

    fn seat(&mut self, seating: Seating) -> TractResult<()> {
        self.seat(seating);
        Ok(())
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        self.reset_lanes(lanes)
    }

    fn runnable(&self) -> &dyn Runnable {
        &self.plan
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

    fn check(&self) -> TractResult<()> {
        self.0.check()
    }
}

impl Debug for InventorizedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

inventory::collect!(InventorizedRuntime);

pub fn runtimes() -> impl Iterator<Item = &'static dyn Runtime> {
    inventory::iter::<InventorizedRuntime>().filter(|rt| rt.check().is_ok()).map(|ir| ir.0)
}

/// Known GPU backends, tried in order when resolving the virtual `gpu`
/// (strict) / `gpu-or-cpu` (best-effort) names.
const GPU_RUNTIME_NAMES: &[&str] = &["metal", "cuda"];

pub fn runtime_for_name(s: &str) -> TractResult<Option<&'static dyn Runtime>> {
    // Back-compat: `default` was the original name for the CPU runtime
    // before it was renamed.  Keep it working as a plain alias.
    let s = if s == "default" { "cpu" } else { s };
    if s == "gpu" || s == "gpu-or-cpu" {
        let mut last_check_err: Option<TractError> = None;
        for name in GPU_RUNTIME_NAMES {
            let Some(rt) = inventory::iter::<InventorizedRuntime>().find(|rt| rt.name() == *name)
            else {
                continue;
            };
            match rt.check() {
                Ok(()) => return Ok(Some(rt.0)),
                Err(e) => last_check_err = Some(e),
            }
        }
        if s == "gpu" {
            let detail =
                last_check_err.map(|e| format!(" (last backend error: {e:#})")).unwrap_or_default();
            bail!("Runtime `gpu` requested but no GPU backend is available{detail}");
        }
        // gpu-or-cpu: fall through to the cpu runtime.
        return runtime_for_name("cpu");
    }
    rule_if_some!(rt = inventory::iter::<InventorizedRuntime>().find(|rt| rt.name() == s));
    rt.check()?;
    Ok(Some(rt.0))
}

#[macro_export]
macro_rules! register_runtime {
    ($type: ty= $val:expr) => {
        static D: $type = $val;
        inventory::submit! { $crate::runtime::InventorizedRuntime(&D) }
    };
}

register_runtime!(DefaultRuntime = DefaultRuntime);
