//! A runnable answering to the contract of the model it was prepared from as
//! that model was before batchify gave it a batch axis.

use crate::internal::*;

/// `inner`, prepared from a model batchified on axis 0, answering to the model
/// as it was before: facts without the batch axis, the properties it had (its
/// `pulse.*_axes` counting without the axis), and states that add axis 0 to
/// the batched inputs and drop it from the batched outputs.
#[derive(Clone, Debug)]
pub struct UnbatchedRunnable {
    inner: Arc<dyn Runnable>,
    interface: Arc<Interface>,
}

/// What the runnable answers to, and which inputs and outputs carry the batch
/// axis the states add and drop.
#[derive(Debug)]
struct Interface {
    inputs: Vec<TypedFact>,
    outputs: Vec<TypedFact>,
    properties: HashMap<String, Arc<Tensor>>,
    batch_in: Vec<bool>,
    batch_out: Vec<bool>,
}

impl UnbatchedRunnable {
    /// `inner` answering to `original`'s contract, when every input and output
    /// of `inner` is `original`'s with one batch symbol prepended on axis 0 or
    /// as it was, one of them batched at least. `inner` itself otherwise: no
    /// batch axis was added, or the model changed some other way.
    pub fn hide(inner: Box<dyn Runnable>, original: &TypedModel) -> TractResult<Box<dyn Runnable>> {
        match Interface::of(&*inner, original)? {
            Some(interface) => Ok(Box::new(UnbatchedRunnable {
                inner: inner.into(),
                interface: Arc::new(interface),
            })),
            None => Ok(inner),
        }
    }

    /// The runnable as prepared, batch axis and all.
    pub fn inner(&self) -> &Arc<dyn Runnable> {
        &self.inner
    }
}

impl Interface {
    fn of(inner: &dyn Runnable, original: &TypedModel) -> TractResult<Option<Interface>> {
        if original.inputs.len() != inner.input_count()
            || original.outputs.len() != inner.output_count()
        {
            return Ok(None);
        }
        let mut batch: Option<TDim> = None;
        let mut unbatched = |fact: &TypedFact, before: &TypedFact| -> Option<(TypedFact, bool)> {
            let dims = fact.shape.dims();
            if dims == before.shape.dims() {
                return Some((fact.clone(), false));
            }
            let (first, rest) = dims.split_first()?;
            if !matches!(first, TDim::Sym(_)) || rest != before.shape.dims() {
                return None;
            }
            if batch.get_or_insert_with(|| first.clone()) != first {
                return None;
            }
            let mut fact = fact.clone();
            fact.shape = rest.into();
            Some((fact, true))
        };
        let (mut inputs, mut batch_in) = (vec![], vec![]);
        for ix in 0..inner.input_count() {
            let Some((fact, batched)) = unbatched(inner.input_fact(ix)?, original.input_fact(ix)?)
            else {
                return Ok(None);
            };
            inputs.push(fact);
            batch_in.push(batched);
        }
        let (mut outputs, mut batch_out) = (vec![], vec![]);
        for ix in 0..inner.output_count() {
            let Some((fact, batched)) =
                unbatched(inner.output_fact(ix)?, original.output_fact(ix)?)
            else {
                return Ok(None);
            };
            outputs.push(fact);
            batch_out.push(batched);
        }
        if batch.is_none() {
            return Ok(None);
        }
        let properties = original.properties.clone();
        Ok(Some(Interface { inputs, outputs, properties, batch_in, batch_out }))
    }
}

impl Runnable for UnbatchedRunnable {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(UnbatchedState { inner: self.inner.spawn()?, runnable: self.clone() }))
    }

    fn input_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.interface.inputs.get(ix).with_context(|| format!("No input {ix}"))
    }

    fn output_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.interface.outputs.get(ix).with_context(|| format!("No output {ix}"))
    }

    fn properties(&self) -> &HashMap<String, Arc<Tensor>> {
        &self.interface.properties
    }

    fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>> {
        self.inner.typed_plan()
    }

    fn typed_model(&self) -> Option<&Arc<TypedModel>> {
        self.inner.typed_model()
    }
}

#[derive(Clone, Debug)]
struct UnbatchedState {
    inner: Box<dyn State>,
    runnable: UnbatchedRunnable,
}

/// Apply `edit` at axis 0 of the `values` flagged in `batched`.
fn reshape_batched(
    values: TVec<TValue>,
    batched: &[bool],
    edit: fn(&mut Tensor, usize) -> TractResult<()>,
) -> TractResult<TVec<TValue>> {
    values
        .into_iter()
        .enumerate()
        .map(|(ix, value)| {
            if !batched.get(ix).copied().unwrap_or(false) {
                return Ok(value);
            }
            let mut tensor = value.into_tensor();
            edit(&mut tensor, 0)?;
            Ok(tensor.into_tvalue())
        })
        .collect()
}

impl State for UnbatchedState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let interface = &self.runnable.interface;
        let inputs = reshape_batched(inputs, &interface.batch_in, Tensor::insert_axis)?;
        let outputs = self.inner.run(inputs)?;
        reshape_batched(outputs, &interface.batch_out, Tensor::remove_axis)
    }

    fn resolve_symbol(&mut self, symbol: &Symbol, value: i64) -> TractResult<()> {
        self.inner.resolve_symbol(symbol, value)
    }

    fn seat(&mut self, seating: Seating) -> TractResult<()> {
        self.inner.seat(seating)
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        self.inner.reset_lanes(lanes)
    }

    fn runnable(&self) -> &dyn Runnable {
        &self.runnable
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::math::mul;

    /// `[3] * 2`, its `pulse.input_axes` claiming axis 0.
    fn doubler() -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::fact([3]))?;
        let two = model.add_const("two", tensor1(&[2f32]))?;
        let doubled = model.wire_node("doubled", mul(), &[input, two])?;
        model.select_output_outlets(&doubled)?;
        model.properties.insert("pulse.input_axes".into(), rctensor1(&[0i64]));
        Ok(model)
    }

    #[test]
    fn a_batchified_model_keeps_the_contract_it_was_handed_over_with() -> TractResult<()> {
        let model = doubler()?;
        let batch = model.symbols.sym("B");
        let batched = crate::batchify::batchify(&model, &batch, &[])?;
        assert_eq!(*batched.properties["pulse.input_axes"], tensor1(&[1i64]));
        let inner = DefaultRuntime.prepare(batched)?;
        let runnable = UnbatchedRunnable::hide(inner, &model)?;
        assert!(runnable.is::<UnbatchedRunnable>());
        assert_eq!(runnable.input_fact(0)?.shape.dims(), &[3.to_dim()]);
        assert_eq!(runnable.output_fact(0)?.shape.dims(), &[3.to_dim()]);
        assert_eq!(*runnable.properties()["pulse.input_axes"], tensor1(&[0i64]));
        let output = runnable.run(tvec!(tensor1(&[1f32, 2., 3.]).into_tvalue()))?;
        assert_eq!(*output[0], tensor1(&[2f32, 4., 6.]));
        Ok(())
    }

    #[test]
    fn a_model_left_as_it_was_is_not_hidden() -> TractResult<()> {
        let model = doubler()?;
        let inner = DefaultRuntime.prepare(model.clone())?;
        assert!(!UnbatchedRunnable::hide(inner, &model)?.is::<UnbatchedRunnable>());
        Ok(())
    }
}
