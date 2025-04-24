use crate::ops::OpStateFreeze;

use super::*;
use tract_data::internal::*;

#[derive(Debug, Clone, new)]
pub struct ScanOpParams {
    pub skip: usize,
    pub reset_every_turn: bool,
    pub plan: Arc<TypedSimplePlan<TypedModel>>,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
}

#[derive(Debug, Clone, new)]
pub struct OptScan(Arc<ScanOpParams>);

impl std::ops::Deref for OptScan {
    type Target = ScanOpParams;
    fn deref(&self) -> &ScanOpParams {
        &self.0
    }
}

impl OptScan {
    pub fn iteration_count(&self, inputs: &[&TypedFact]) -> Option<TDim> {
        super::iteration_count(&self.input_mapping, inputs)
    }
}

impl Op for OptScan {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut lines = vec![];
        for (ix, im) in self.input_mapping.iter().enumerate() {
            lines.push(format!("Model input  #{ix}: {im:?}"));
        }
        for (ix, om) in self.output_mapping.iter().enumerate() {
            lines.push(format!("Model output #{ix}: {om:?}"));
        }
        Ok(lines)
    }

    op_as_typed_op!();
}

impl EvalOp for OptScan {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(State {
            position: 0,
            hidden_state: tvec!(),
            model_state: TypedSimpleState::new(Arc::clone(&self.plan))?,
            op: Arc::clone(&self.0),
        })))
    }
}

#[derive(Clone, Debug)]
pub struct State {
    op: Arc<ScanOpParams>,
    position: usize,
    hidden_state: TVec<TValue>,
    pub model_state: TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
}

#[derive(Debug, Clone)]
struct FrozenState {
    op: Arc<ScanOpParams>,
    position: usize,
    hidden_state: TVec<Tensor>,
    model_state: TypedFrozenSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
}

impl OpStateFreeze for State {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenState {
            op: self.op.clone(),
            position: self.position,
            hidden_state: self.hidden_state.iter().map(|t| t.clone().into_tensor()).collect(),
            model_state: self.model_state.freeze(),
        })
    }
}

impl FrozenOpState for FrozenState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(State {
            op: self.op.clone(),
            position: self.position,
            hidden_state: self.hidden_state.iter().map(|t| t.clone().into_tvalue()).collect(),
            model_state: self.model_state.unfreeze(),
        })
    }
}

impl State {
    pub fn iteration_count(&self, inputs: &TVec<TValue>) -> usize {
        let (slot, info) = self
            .op
            .input_mapping
            .iter()
            .enumerate()
            .find_map(|(ix, it)| it.as_scan().map(|scan| (ix, scan)))
            .unwrap();
        inputs[slot].shape()[info.axis].divceil(info.chunk.unsigned_abs())
    }

    pub(super) fn slice_input(
        input: &Tensor,
        axis: usize,
        chunk_ix: usize,
        chunk_dim: isize,
    ) -> TractResult<Tensor> {
        unsafe {
            let full_len = input.shape()[axis];
            let mut shape: TVec<usize> = input.shape().into();
            shape[axis] = chunk_dim.unsigned_abs();
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
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let iters = self.iteration_count(&inputs);

        let State { op, ref mut hidden_state, ref mut position, ref mut model_state } = self;

        // initialize state at first pass, or when forced
        if op.reset_every_turn {
            hidden_state.clear()
        }
        if hidden_state.len() == 0 {
            for (slot, input) in op.input_mapping.iter().enumerate() {
                if input.is_state() {
                    hidden_state.push(inputs[slot].clone());
                }
            }
        }

        let mut outputs = tvec!();
        for (ix, output) in op.output_mapping.iter().enumerate() {
            if let Some((slot, info)) = output.scan {
                let fact = op.plan.model().output_fact(ix)?;
                let mut shape: TVec<usize> =
                    fact.shape.eval_to_usize(&session.resolved_symbols)?.into_owned();
                let scanning_dim = output
                    .full_dim_hint
                    .as_ref()
                    .and_then(|d| d.to_usize().ok())
                    .unwrap_or(shape[info.axis] * iters);
                shape[info.axis] = scanning_dim;
                let t = unsafe { Tensor::uninitialized_dt(fact.datum_type, &shape)? };
                outputs.push((slot, t));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, Tensor::default()));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let mut outputs: TVec<Tensor> = outputs.into_iter().map(|(_slot, v)| v).collect();

        for i in 0..iters {
            *position += 1;
            if *position <= op.skip {
                continue;
            }
            hidden_state.reverse();

            let iter_inputs: TVec<TValue> = op
                .input_mapping
                .iter()
                .enumerate()
                .map(|(slot, m)| {
                    Ok(match m {
                        InputMapping::State => Some(hidden_state.pop().unwrap()),
                        InputMapping::Scan(info) => Some(
                            Self::slice_input(&inputs[slot], info.axis, i, info.chunk)?
                                .into_tvalue(),
                        ),
                        InputMapping::Full => Some(inputs[slot].clone()),
                    })
                })
                .collect::<TractResult<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            trace!("iter_inputs #{i}: {iter_inputs:?}");
            let iter_outputs =
                model_state.run(iter_inputs).with_context(|| "Evaluating inner body")?;
            trace!("iter_outputs #{i}: {iter_outputs:?}");

            for (v, mapping) in iter_outputs.into_iter().zip(&op.output_mapping) {
                if let Some((slot, info)) = mapping.scan {
                    Self::assign_output(&mut outputs[slot], info.axis, &v, i, info.chunk < 0);
                }
                if i == iters - 1 {
                    if let Some(slot) = mapping.last_value_slot {
                        outputs[slot] = v.clone().into_tensor();
                    }
                }
                if mapping.state {
                    hidden_state.push(v);
                }
            }
        }

        Ok(outputs.into_iter().map(|t| t.into_tvalue()).collect())
    }
}

impl TypedOp for OptScan {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut outputs = tvec!();
        let iters = super::iteration_count(&self.input_mapping, inputs).unwrap();
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.plan.model().output_fact(ix)?;
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, fact.datum_type.fact(fact.shape.clone())));
            }
            if let Some((slot, info)) = output.scan {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape[info.axis].clone() * &iters);
                shape.set(info.axis, scanning_dim);
                outputs.push((slot, fact.datum_type.fact(shape)));
            }
        }
        outputs.sort_by_key(|a| a.0);
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }

    fn nested_model_multipliers(&self, inputs: &[&TypedFact]) -> Vec<(Cow<str>, TDim)> {
        vec![(
            "loop".into(),
            super::iteration_count(&self.input_mapping, inputs).unwrap_or_else(|| 1.to_dim()),
        )]
    }
}
