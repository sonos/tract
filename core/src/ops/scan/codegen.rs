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

    fn nested_models(&self) -> Vec<(Cow<str>, &dyn Model, Vec<String>, Vec<String>)> {
        vec![(
            "loop".into(),
            self.plan.model(),
            self.input_mapping.iter().map(|m| format!("{:?}", m)).collect(),
            self.output_mapping.iter().map(|m| format!("{:?}", m)).collect(),
        )]
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut lines = vec![];
        for (ix, im) in self.input_mapping.iter().enumerate() {
            lines.push(format!("Model input  #{}: {:?}", ix, im));
        }
        for (ix, om) in self.output_mapping.iter().enumerate() {
            lines.push(format!("Model output #{}: {:?}", ix, om));
        }
        Ok(lines)
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatefullOp for Codegen {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
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
        op: &dyn Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let mut _codegen_op_holder = None;
        let op = if let Some(op) = op.downcast_ref::<Codegen>() {
            op
        } else if let Some(op) = op.downcast_ref::<TypedScan>() {
            _codegen_op_holder = Some(op.to_codegen_op()?);
            _codegen_op_holder.as_ref().unwrap()
        } else if let Some(op) = op.downcast_ref::<InferenceScan>() {
            _codegen_op_holder = Some(op.to_typed_scan()?.to_codegen_op()?);
            _codegen_op_holder.as_ref().unwrap()
        } else {
            panic!("Wrong op");
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
            if let Some(slot) = output.full_slot {
                let fact = op.plan.model().output_fact(ix)?;
                let mut shape: TVec<usize> = fact.shape.as_finite().unwrap().into();
                let scanning_dim = output
                    .full_dim_hint
                    .as_ref()
                    .and_then(|d| d.to_integer().ok().map(|i| i as usize))
                    .unwrap_or(shape[output.axis] * iters);
                shape[output.axis] = scanning_dim;
                let t = dispatch_datum!(Self::alloc_output_t(fact.datum_type)(self, &*shape))?;
                outputs.push((slot, t));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, Tensor::default()));
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
                        InputMapping::State { .. } => Some(self.hidden_state.pop().unwrap()),
                        InputMapping::Scan { slot, axis, chunk } => {
                            Some(dispatch_datum!(Self::slice_input_t(inputs[*slot].datum_type())(
                                self,
                                inputs[*slot].as_ref(),
                                *axis,
                                i,
                                *chunk
                            ))?)
                        }
                        InputMapping::Full { slot } => Some(inputs[*slot].clone().into_tensor()),
                    })
                })
                .collect::<TractResult<Vec<_>>>()?
                .into_iter()
                .filter_map(|x| x)
                .collect();

            trace!("iter_inputs: {:?}", iter_inputs);
            let iter_outputs =
                self.model_state.run(iter_inputs).chain_err(|| "Evaluating inner body")?;
            trace!("iter_outputs: {:?}", iter_outputs);

            for (v, mapping) in iter_outputs.into_iter().zip(&op.output_mapping) {
                if let Some(slot) = mapping.full_slot {
                    dispatch_datum!(Self::assign_output_t(v.datum_type())(
                        self,
                        &mut outputs[slot],
                        mapping.axis,
                        v.as_ref(),
                        i
                    ))?;
                }
                if i == iters - 1 {
                    if let Some(slot) = mapping.last_value_slot {
                        outputs[slot] = v.clone().into_tensor();
                    }
                }
                if mapping.state {
                    self.hidden_state.push(v.into_tensor());
                }
            }
        }

        Ok(outputs.into_iter().map(Arc::new).collect())
    }
}

impl TypedOp for Codegen {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut outputs = tvec!();
        let iters = {
            let (outside_slot, axis, chunk) = self
                .input_mapping
                .iter()
                .filter_map(|it| match it {
                    InputMapping::Scan { axis, slot, chunk } => Some((*slot, *axis, *chunk)),
                    _ => None,
                })
                .next()
                .unwrap();
            inputs[outside_slot].shape.dim(axis).div_ceil(chunk.to_dim())
        };
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.plan.model().output_fact(ix)?;
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, TypedFact::dt_shape(fact.datum_type, fact.shape.clone())?));
            }
            if let Some(slot) = output.full_slot {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape.dim(output.axis) * &iters);
                shape.set_dim(output.axis, scanning_dim)?;
                outputs.push((slot, TypedFact::dt_shape(fact.datum_type, shape)?));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }

    fn nested_model_multipliers(&self, inputs: &[&TypedFact]) -> Vec<(Cow<str>, f32)> {
        let iters = {
            let (outside_slot, axis, chunk) = self
                .input_mapping
                .iter()
                .filter_map(|it| match it {
                    InputMapping::Scan { axis, slot, chunk } => Some((*slot, *axis, *chunk)),
                    _ => None,
                })
                .next()
                .unwrap();
            inputs[outside_slot].shape.dim(axis).to_integer().unwrap() as f32 / chunk as f32
        };
        vec![("loop".into(), iters as f32)]
    }
}
