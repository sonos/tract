use super::*;
use tract_data::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct LirScanOpParams {
    pub skip: usize,
    pub plan: Arc<TypedSimplePlan<TypedModel>>,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
}

#[derive(Debug, Clone, new, Hash)]
pub struct LirScan(Arc<LirScanOpParams>);

impl std::ops::Deref for LirScan {
    type Target = LirScanOpParams;
    fn deref(&self) -> &LirScanOpParams {
        &self.0
    }
}

impl_dyn_hash!(LirScan);

impl LirScan {
    pub fn iteration_count(&self, inputs: &[&TypedFact]) -> Option<TDim> {
        let (outside_slot, axis, chunk) = self
            .input_mapping
            .iter()
            .filter_map(|it| match it {
                InputMapping::Scan { axis, slot, chunk } => Some((*slot, *axis, *chunk)),
                _ => None,
            })
            .next()
            .unwrap();
        let outside_dim = inputs[outside_slot].shape[axis].clone();
        Some(outside_dim / chunk)
    }
}

impl Op for LirScan {
    fn name(&self) -> Cow<str> {
        "Scan".into()
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

    op_core_lir!();
    op_as_typed_op!();
}

impl EvalOp for LirScan {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(State {
            mutable: MutableState {
                position: 0,
                hidden_state: tvec!(),
                model_state: TypedSimpleState::new(Arc::clone(&self.plan))?,
            },
            op: Arc::clone(&self.0),
        })))
    }
}

#[derive(Clone, Debug)]
struct State {
    op: Arc<LirScanOpParams>,
    mutable: MutableState,
}

#[derive(Clone, Debug)]
struct MutableState {
    position: usize,
    hidden_state: TVec<Tensor>,
    model_state: TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
}

impl MutableState {
    pub(super) fn slice_input(
        &self,
        input: &Tensor,
        axis: usize,
        chunk_ix: usize,
        chunk_dim: isize,
    ) -> TractResult<Tensor> {
        unsafe {
            let full_len = input.shape()[axis];
            let mut shape: TVec<usize> = input.shape().into();
            shape[axis] = chunk_dim.abs() as usize;
            let mut t = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
            if chunk_dim < 0 {
                let chunk_dim = (-chunk_dim) as usize;
                for i in 0..chunk_dim {
                    if chunk_dim * chunk_ix + i < full_len {
                        let dst_ix = chunk_dim - i - 1;
                        let src_ix = full_len - 1 - (chunk_ix * chunk_dim + i);
                        t.assign_slice_unchecked(dst_ix..=dst_ix, input, src_ix..=src_ix, axis);
                    }
                }
            } else if (chunk_ix + 1) * chunk_dim as usize > full_len {
                let chunk_dim = chunk_dim as usize;
                let remain = full_len - chunk_ix * chunk_dim;
                let mut shape: TVec<usize> = input.shape().into();
                shape[axis] = chunk_dim;
                t.assign_slice_unchecked(..remain, input, chunk_ix * chunk_dim.., axis);
            } else {
                let start = chunk_dim as usize * chunk_ix;
                let end = start + chunk_dim as usize;
                t.assign_slice_unchecked(.., input, start..end, axis);
            }
            Ok(t)
        }
    }

    pub(super) fn assign_output(
        &self,
        output: &mut Tensor,
        axis: usize,
        element_value: &Tensor,
        i: usize,
        backward: bool,
    ) {
        let full_len = output.shape()[axis];
        let offset = if backward {
            full_len - 1 - i * element_value.shape()[axis]
        } else {
            i * element_value.shape()[axis]
        };
        let count = element_value.shape()[axis].min(output.shape()[axis] - offset);
        unsafe {
            output.assign_slice_unchecked(offset..offset + count, element_value, ..count, axis)
        };
    }
}

impl OpState for State {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let State { op, ref mut mutable } = self;
        // initialize state at first pass
        if mutable.hidden_state.len() == 0 {
            for input in &op.input_mapping {
                if let InputMapping::State { initializer } = input {
                    mutable.hidden_state.push(match initializer {
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
            inputs[outside_slot].shape()[axis].divceil(chunk.abs() as usize)
        };

        let mut outputs = tvec!();
        for (ix, output) in op.output_mapping.iter().enumerate() {
            if let Some(slot) = output.full_slot {
                let fact = op.plan.model().output_fact(ix)?;
                let mut shape: TVec<usize> =
                    fact.shape.eval_to_usize(&session.resolved_symbols)?.into_owned();
                let scanning_dim = output
                    .full_dim_hint
                    .as_ref()
                    .and_then(|d| d.to_usize().ok())
                    .unwrap_or(shape[output.axis] * iters);
                shape[output.axis] = scanning_dim;
                let t = unsafe { Tensor::uninitialized_dt(fact.datum_type, &*shape)? };
                outputs.push((slot, t));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, Tensor::default()));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let mut outputs: TVec<Tensor> = outputs.into_iter().map(|(_slot, v)| v).collect();

        for i in 0..iters {
            mutable.position += 1;
            if mutable.position <= op.skip {
                continue;
            }
            mutable.hidden_state.reverse();

            let iter_inputs: TVec<Tensor> = op
                .input_mapping
                .iter()
                .map(|m| {
                    Ok(match m {
                        InputMapping::State { .. } => Some(mutable.hidden_state.pop().unwrap()),
                        InputMapping::Scan { slot, axis, chunk } => {
                            Some(MutableState::slice_input(
                                mutable,
                                inputs[*slot].as_ref(),
                                *axis,
                                i,
                                *chunk,
                            )?)
                        }
                        InputMapping::Full { slot } => Some(inputs[*slot].clone().into_tensor()),
                    })
                })
                .collect::<TractResult<Vec<_>>>()?
                .into_iter()
                .filter_map(|x| x)
                .collect();

            trace!("iter_inputs #{}: {:?}", i, iter_inputs);
            let iter_outputs =
                mutable.model_state.run(iter_inputs).with_context(|| "Evaluating inner body")?;
            trace!("iter_outputs #{}: {:?}", i, iter_outputs);

            for (v, mapping) in iter_outputs.into_iter().zip(&op.output_mapping) {
                if let Some(slot) = mapping.full_slot {
                    mutable.assign_output(
                        &mut outputs[slot],
                        mapping.axis,
                        v.as_ref(),
                        i,
                        mapping.chunk < 0,
                    );
                }
                if i == iters - 1 {
                    if let Some(slot) = mapping.last_value_slot {
                        outputs[slot] = v.clone().into_tensor();
                    }
                }
                if mapping.state {
                    mutable.hidden_state.push(v.into_tensor());
                }
            }
        }

        Ok(outputs.into_iter().map(Arc::new).collect())
    }
}

impl TypedOp for LirScan {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut outputs = tvec!();
        let iters = {
            let (outside_slot, axis, chunk) = self
                .input_mapping
                .iter()
                .find_map(|it| match it {
                    InputMapping::Scan { axis, slot, chunk } => Some((*slot, *axis, *chunk)),
                    _ => None,
                })
                .unwrap();
            inputs[outside_slot].shape[axis].clone().div_ceil(chunk.abs() as _)
        };
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.plan.model().output_fact(ix)?;
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, fact.datum_type.fact(fact.shape.clone())));
            }
            if let Some(slot) = output.full_slot {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape[output.axis].clone() * &iters);
                shape.set(output.axis, scanning_dim);
                outputs.push((slot, fact.datum_type.fact(shape)));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }
}
