use std::fmt;

use crate::internal::*;
use ndarray::prelude::*;

// Scan node outer interface:
// inputs: [ hidden_state_len initial values ][ num_scan_inputs inputs ][ implicit capture inputs ]
// outputs: [ hidden_state_len final values ][ aggregated outputs ]

#[derive(Debug, Clone, new, Default)]
pub struct Scan<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op> + Clone + 'static,
    ModelImpl<TI, O>: SomeModel,
{
    pub body: ModelImpl<TI, O>,
    num_scan_inputs: usize,
    closure_inputs: usize,
    scan_input_axes: Vec<usize>,
    scan_output_axes: Vec<usize>,
    scan_output_len_hint: Vec<Option<TDim>>,
    prune_scanning_dim: bool, // TODO check scanning dims == 1
}

impl<TI, O> Scan<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op> + Clone + 'static,
    ModelImpl<TI, O>: SomeModel,
{
    fn slice_input_t<T: Datum>(
        &self,
        scan_inputs: &[Arc<Tensor>],
        input: usize,
        i: usize,
        count: usize,
    ) -> TractResult<Tensor> {
        let view = scan_inputs[input].to_array_view::<T>()?;
        let axis = Axis(self.scan_input_axes.get(input).cloned().unwrap_or(0));
        let full_len = view.shape()[axis.0];
        let slice = if self.prune_scanning_dim {
            view.index_axis_move(axis, i).to_owned()
        } else if (i + 1) * count > full_len {
            let remain = full_len - i * count;
            let mut shape: TVec<usize> = view.shape().into();
            shape[axis.0] = count;
            let mut t = ArrayD::<T>::default(&*shape);
            t.slice_axis_mut(axis, (0..remain).into())
                .assign(&view.slice_axis(axis, (i * count..).into()));
            t
        } else {
            view.slice_axis(axis, (i * count..(i + 1) * count).into()).to_owned()
        };
        Ok(slice.into_tensor())
    }

    fn alloc_output_t<T: Datum + Default>(&self, shape: &[usize]) -> TractResult<Tensor> {
        unsafe { Tensor::uninitialized::<T>(&shape) }
    }

    fn assign_output_t<T: Datum + Default>(
        &self,
        output: &mut Tensor,
        output_id: usize,
        element_value: &Tensor,
        i: usize,
    ) -> TractResult<()> {
        let axis = self.scan_output_axes.get(output_id).cloned().unwrap_or(0);
        let mut view = output.to_array_view_mut::<T>()?;
        let element = element_value.to_array_view::<T>()?;
        if self.prune_scanning_dim {
            view.index_axis_move(Axis(axis), i).assign(&element);
        } else {
            let offset = i * element_value.shape()[axis];
            let count = element_value.shape()[axis].min(view.shape()[axis] - offset);
            view.slice_axis_mut(Axis(axis), (offset..offset + count).into())
                .assign(&element.slice_axis(Axis(axis), (..count).into()));
        };
        Ok(())
    }
}

impl Op for Scan<TensorFact, Box<InferenceOp>> {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &SomeModel)> {
        vec!(("loop".into(), &self.body as _))
    }

    fn to_typed(&self) -> TractResult<Option<Box<Op>>> {
        let typed_model = self.body.clone().into_typed()?;
        Ok(Some(Box::new(Scan::new(
            typed_model,
            self.num_scan_inputs,
            self.closure_inputs,
            self.scan_input_axes.clone(),
            self.scan_output_axes.clone(),
            self.scan_output_len_hint.clone(),
            self.prune_scanning_dim,
        ))))
    }
}

impl StatelessOp for Scan<TensorFact, Box<InferenceOp>> {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!()
    }
}

impl Op for Scan<TypedTensorInfo, Box<Op>> {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &SomeModel)> {
        vec!(("loop".into(), &self.body as _))
    }
}

impl StatelessOp for Scan<TypedTensorInfo, Box<Op>> {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let hidden_state_len = inputs.len() - self.num_scan_inputs - self.closure_inputs;

        // extract hidden state original values from inputs
        let mut state: TVec<Tensor> = tvec!();
        for _ in 0..hidden_state_len {
            state.push(inputs.remove(0).into_tensor());
        }

        let first_input_axis = self.scan_input_axes.get(0).cloned().unwrap_or(0);
        let iters = if self.prune_scanning_dim {
            inputs[0].shape()[first_input_axis]
        } else {
            inputs[0].shape()[first_input_axis].div_ceil(
                self.body.input_fact(hidden_state_len)?.shape.dim(first_input_axis).to_integer()?
                    as usize,
            )
        };

        let mut scan_outputs = tvec!();
        for i in 0..(self.body.output_outlets()?.len() - hidden_state_len) {
            let fact = self.body.output_fact(hidden_state_len + i)?;
            let dt = fact.datum_type;
            let mut shape: TVec<usize> = fact.shape.as_finite().unwrap().into();
            let axis = self.scan_output_axes.get(i).cloned().unwrap_or(0);
            let scanning_dim = self
                .scan_output_len_hint
                .get(0)
                .and_then(|x| x.as_ref())
                .and_then(|i| i.to_integer().ok())
                .map(|i| i as usize)
                .unwrap_or(if self.prune_scanning_dim { iters } else { shape[axis] * iters });
            if self.prune_scanning_dim {
                shape.insert(axis, scanning_dim)
            } else {
                shape[axis] = scanning_dim
            }
            let t = dispatch_datum!(Self::alloc_output_t(dt)(self, &*shape))?;
            scan_outputs.push(t);
        }

        let plan = SimplePlan::new(&self.body)?;

        for i in 0..iters {
            // body inputs are state + one slice of each input
            let mut iter_inputs: TVec<Tensor> = state.drain().collect();
            for input in 0..self.num_scan_inputs {
                let fact = self.body.input_fact(input + hidden_state_len)?;
                let axis = self.scan_input_axes.get(input).cloned().unwrap_or(0);
                let count = fact.shape.dim(axis);
                let tensor = dispatch_datum!(Self::slice_input_t(inputs[input].datum_type())(
                    self,
                    &*inputs,
                    input,
                    i,
                    count.to_integer()? as usize
                ))?;
                if cfg!(debug_assert) {
                    fact.to_tensor_fact().unify(&TensorFact::from(tensor.clone()))?;
                }
                iter_inputs.push(tensor);
            }
            for i in 0..self.closure_inputs {
                iter_inputs
                    .push(inputs[inputs.len() - self.closure_inputs + i].clone().into_tensor());
            }

            trace!("iter_inputs: {:?}", iter_inputs);
            let mut iter_outputs = plan.run(iter_inputs).chain_err(|| "Evaluating inner body")?;

            for _ in 0..hidden_state_len {
                state.push(iter_outputs.remove(0).into_tensor());
            }
            for (ix, o) in scan_outputs.iter_mut().enumerate() {
                dispatch_datum!(Self::assign_output_t(o.datum_type())(
                    self,
                    o,
                    ix,
                    &iter_outputs[ix],
                    i
                ))?;
            }
        }
        let mut output: TVec<Arc<Tensor>> =
            state.into_iter().map(|t| t.into_arc_tensor()).collect();
        output.extend(scan_outputs.into_iter().map(|t| t.into_arc_tensor()));
        Ok(output)
    }
}

impl Scan<TensorFact, Box<InferenceOp>> {
    fn unify_scanning_tensor_fact(
        outer: &mut TensorFact,
        inner: &mut TensorFact,
        outer_scan_axis: usize,
        prune_scanning_dim: bool,
    ) -> TractResult<()> {
        outer.datum_type.unify_with_mut(&mut inner.datum_type)?;
        let outer_rank = outer
            .shape
            .rank()
            .concretize()
            .or(inner.shape.rank().concretize().map(|r| r - prune_scanning_dim as usize as i32))
            .map(|r| r as usize);
        if let Some(outer_rank) = outer_rank {
            let inner_rank = outer_rank - prune_scanning_dim as usize;
            outer
                .shape
                .unify_with(&ShapeFact::closed(tvec!(GenericFact::Any; outer_rank as usize)))?;
            inner
                .shape
                .unify_with(&ShapeFact::closed(tvec!(GenericFact::Any; inner_rank as usize)))?;
            for outer_axis in 0..outer_rank {
                if outer_axis != outer_scan_axis {
                    let inner_axis =
                        outer_axis - (prune_scanning_dim && outer_axis > outer_scan_axis) as usize;
                    let value = outer.shape.dim(outer_axis).unwrap().concretize().or(inner
                        .shape
                        .dim(inner_axis)
                        .unwrap()
                        .concretize());
                    if let Some(value) = value {
                        outer.shape.set_dim(outer_axis, value.clone());
                        inner.shape.set_dim(inner_axis, value);
                    }
                }
            }
        }
        Ok(())
    }

    fn unify_facts(
        &mut self,
        inputs: &mut [TensorFact],
        outputs: &mut [TensorFact],
    ) -> TractResult<()> {
        let body_inputs = self.body.input_outlets()?.len();
        let body_outputs = self.body.output_outlets()?.len();
        let hidden_state_len = body_inputs - self.num_scan_inputs - self.closure_inputs;
        let num_scan_outputs = body_outputs - hidden_state_len;
        for i in 0..hidden_state_len {
            trace!("Unify hidden state #{}", i);
            let mut merged =
                self.body.input_fact(i)?.datum_type.unify(&self.body.output_fact(i)?.datum_type)?;
            Fact::unify_all(&mut [
                &mut merged,
                &mut inputs[i].datum_type,
                &mut outputs[i].datum_type,
            ])
            .map_err(|e| format!("while unifying hidden state datum_types #{}: {}", i, e))?;
            self.body.input_fact_mut(i)?.datum_type.unify_with(&mut merged)?;
            self.body.output_fact_mut(i)?.datum_type.unify_with(&mut merged)?;

            let mut merged =
                self.body.input_fact(i)?.shape.unify(&self.body.output_fact(i)?.shape)?;
            Fact::unify_all(&mut [&mut merged, &mut inputs[i].shape, &mut outputs[i].shape])
                .map_err(|e| format!("while unifying hidden state shapes #{}: {}", i, e))?;
            self.body.input_fact_mut(i)?.shape.unify_with(&mut merged)?;
            self.body.output_fact_mut(i)?.shape.unify_with(&mut merged)?;
        }
        for i in 0..self.num_scan_inputs {
            trace!("Unifying scan input #{}", hidden_state_len + i);
            let incoming = &mut inputs[hidden_state_len + i];
            let inner = self.body.input_fact_mut(hidden_state_len + i)?;
            let axis = self.scan_input_axes.get(i).cloned().unwrap_or(0);
            Self::unify_scanning_tensor_fact(incoming, inner, axis, self.prune_scanning_dim)?;
        }
        for i in 0..self.closure_inputs {
            let id = hidden_state_len + self.num_scan_inputs + i;
            trace!("Unifying closure input #{}", id);
            inputs[id].unify_with(self.body.input_fact_mut(id)?)?;
        }
        for i in 0..num_scan_outputs {
            trace!("Unifying scan output #{}", hidden_state_len + i);
            let outgoing = &mut outputs[hidden_state_len + i];
            let inner = self.body.output_fact_mut(hidden_state_len + i)?;
            let axis = self.scan_output_axes.get(i).cloned().unwrap_or(0);
            Self::unify_scanning_tensor_fact(outgoing, inner, axis, self.prune_scanning_dim)?;
        }
        Ok(())
    }
}

impl InferenceOp for Scan<TensorFact, Box<InferenceOp>> {
    fn infer_facts(
        &mut self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
        _observed: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>, TVec<TensorFact>)> {
        let body_inputs = self.body.input_outlets()?.len();
        let body_outputs = self.body.output_outlets()?.len();
        if inputs.len() != body_inputs {
            bail!("Scan receives {} inputs, inner model expects {}", inputs.len(), body_inputs)
        }
        if outputs.len() != body_outputs {
            bail!("Scan has {} outputs, inner model expects {}", outputs.len(), body_outputs)
        }
        let mut inputs: TVec<TensorFact> = inputs.into_iter().cloned().collect();
        let mut outputs: TVec<TensorFact> = outputs.into_iter().cloned().collect();
        self.unify_facts(&mut inputs, &mut outputs)?;
        trace!("Starting inner model analyse");
        for (ix, input) in self.body.input_outlets()?.iter().enumerate() {
            trace!("  Input inner model: {} {:?} {:?}", ix, input, self.body.input_fact(ix));
        }
        for (ix, output) in self.body.output_outlets()?.iter().enumerate() {
            trace!("  Output inner model: {} {:?} {:?}", ix, output, self.body.output_fact(ix));
        }
        self.body
            .analyse(false)
            .map_err(|e| format!("analysing inner model: {}\n{:#?}", e, self.body))?;
        trace!("Finished inner model analyse");
        self.unify_facts(&mut inputs, &mut outputs)?;
        Ok((inputs, outputs, tvec!()))
    }

    inference_op_as_op!();
}
