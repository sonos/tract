use ndarray::*;

use super::*;

#[derive(Debug, Clone, new)]
pub struct Codegen {
    pub skip: usize,
    pub plan: Arc<TypedSimplePlan<TypedModel>>,
    pub input_mapping: Vec<InputMapping<usize>>,
    pub output_mapping: Vec<OutputMapping<usize, TDim>>,
}

impl Op for Codegen {
    fn name(&self) -> Cow<str> {
        "Codegen".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &Model)> {
        vec![("loop".into(), self.plan.model())]
    }
}

impl StatefullOp for Codegen {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<OpState>>> {
        Ok(Some(Box::new(State {
            position: 0,
            hidden_state: tvec!(),
            model_state: TypedSimpleState::new(Arc::clone(&self.plan))?,
        })))
    }
}

#[derive(Debug, Clone)]
struct State {
    position: usize,
    hidden_state: TVec<Tensor>,
    model_state: TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
}

impl State {
    pub(super) fn slice_input_t<T: Datum>(
        &self,
        input: &Tensor,
        axis: usize,
        i: usize,
        count: usize,
    ) -> TractResult<Tensor> {
        let view = input.to_array_view::<T>()?;
        let full_len = view.shape()[axis];
        if (i + 1) * count > full_len {
            let remain = full_len - i * count;
            let mut shape: TVec<usize> = view.shape().into();
            shape[axis] = count;
            let mut t = ArrayD::<T>::default(&*shape);
            t.slice_axis_mut(Axis(axis), (0..remain).into())
                .assign(&view.slice_axis(Axis(axis), (i * count..).into()));
            Ok(t.into_tensor())
        } else {
            Ok(view
                .slice_axis(Axis(axis), (i * count..(i + 1) * count).into())
                .to_owned()
                .into_tensor())
        }
    }

    pub(super) fn alloc_output_t<T: Datum + Default>(
        &self,
        shape: &[usize],
    ) -> TractResult<Tensor> {
        unsafe { Tensor::uninitialized::<T>(&shape) }
    }

    pub(super) fn assign_output_t<T: Datum + Default>(
        &self,
        output: &mut Tensor,
        axis: usize,
        element_value: &Tensor,
        i: usize,
    ) -> TractResult<()> {
        let mut view = output.to_array_view_mut::<T>()?;
        let element = element_value.to_array_view::<T>()?;
        let offset = i * element_value.shape()[axis];
        let count = element_value.shape()[axis].min(view.shape()[axis] - offset);
        view.slice_axis_mut(Axis(axis), (offset..offset + count).into())
            .assign(&element.slice_axis(Axis(axis), (..count).into()));
        Ok(())
    }
}

impl OpState for State {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        op: &Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let mut _codegen_op_holder = None;
        let op = if let Some(op) = op.downcast_ref::<Codegen>() {
            op
        } else {
            _codegen_op_holder =
                Some(op.downcast_ref::<Typed>().ok_or("Wrong op")?.to_codegen_op()?);
            _codegen_op_holder.as_ref().unwrap()
        };

        // initialize state at first pass
        if self.hidden_state.len() == 0 {
            for input in &op.input_mapping {
                if let InputMapping::State { initializer } = input {
                    self.hidden_state.push(match initializer {
                        StateInitializer::FromInput(slot) => (*inputs[*slot]).to_owned(),
                        StateInitializer::Value(v) => (**v).to_owned(),
                    });
                }
            }
        }

        let iters = {
            let (outside_slot, axis, chunk) = op
                .input_mapping
                .iter()
                .filter_map(|it| match it {
                    InputMapping::Scan { axis, slot, chunk } => Some((*slot, *axis, *chunk)),
                    _ => None,
                })
                .next()
                .unwrap();
            inputs[outside_slot].shape()[axis].div_ceil(chunk)
        };

        let mut outputs = tvec!();
        for (ix, output) in op.output_mapping.iter().enumerate() {
            match output {
                OutputMapping::Scan { slot, axis, full_dim_hint, .. } => {
                    let fact = op.plan.model().output_fact(ix)?;
                    let mut shape: TVec<usize> = fact.shape.as_finite().unwrap().into();
                    let scanning_dim = full_dim_hint
                        .as_ref()
                        .and_then(|d| d.to_integer().ok().map(|i| i as usize))
                        .unwrap_or(shape[*axis] * iters);
                    shape[*axis] = scanning_dim;
                    let t = dispatch_datum!(Self::alloc_output_t(fact.datum_type)(self, &*shape))?;
                    outputs.push((slot, t));
                }
                OutputMapping::State { slot } => {
                    if let Some(slot) = slot {
                        outputs.push((slot, Tensor::default()));
                    }
                }
            }
        }
        outputs.sort_by_key(|a| a.0);
        let mut outputs: TVec<Tensor> = outputs.into_iter().map(|(_slot, v)| v).collect();

        for i in 0..iters {
            self.position += 1;
            if self.position <= op.skip {
                continue;
            }
            self.hidden_state.reverse();

            let iter_inputs: TVec<Tensor> = op
                .input_mapping
                .iter()
                .map(|m| {
                    Ok(match m {
                        InputMapping::State { .. } => self.hidden_state.pop().unwrap(),
                        InputMapping::Scan { slot, axis, chunk } => {
                            dispatch_datum!(Self::slice_input_t(inputs[*slot].datum_type())(
                                self,
                                inputs[*slot].as_ref(),
                                *axis,
                                i,
                                *chunk
                            ))?
                        }
                        InputMapping::Full { slot } => inputs[*slot].clone().into_tensor(),
                    })
                })
                .collect::<TractResult<_>>()?;

            trace!("iter_inputs: {:?}", iter_inputs);
            let iter_outputs =
                self.model_state.run(iter_inputs).chain_err(|| "Evaluating inner body")?;

            for (v, mapping) in iter_outputs.into_iter().zip(&op.output_mapping) {
                match mapping {
                    OutputMapping::State { .. } => self.hidden_state.push(v.into_tensor()),
                    OutputMapping::Scan { axis, slot, .. } => {
                        dispatch_datum!(Self::assign_output_t(v.datum_type())(
                            self,
                            &mut outputs[*slot],
                            *axis,
                            v.as_ref(),
                            i
                        ))?;
                    }
                }
            }
        }

        for (ix, map) in op.output_mapping.iter().enumerate() {
            if let OutputMapping::State { slot } = map {
                let value = self.hidden_state[ix].clone();
                if let Some(slot) = slot {
                    outputs[*slot] = value;
                }
            }
        }

        Ok(outputs.into_iter().map(Arc::new).collect())
    }
}
