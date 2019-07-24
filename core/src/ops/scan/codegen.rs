use crate::plan::SimplePlan;

use ndarray::*;

use super::*;

#[derive(Debug, Clone, new)]
pub struct Codegen {
    pub plan: Arc<SimplePlan<TypedTensorInfo, Box<Op>, ModelImpl<TypedTensorInfo, Box<Op>>>>,
    pub hidden_state_len: usize,
    pub input_mapping: Vec<InputMapping<usize>>,
    pub(super) scan_output_axes: Vec<usize>,
    pub(super) scan_output_len_hint: Vec<Option<TDim>>,
}

impl Codegen {
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
        output_id: usize,
        element_value: &Tensor,
        i: usize,
    ) -> TractResult<()> {
        let axis = self.scan_output_axes.get(output_id).cloned().unwrap_or(0);
        let mut view = output.to_array_view_mut::<T>()?;
        let element = element_value.to_array_view::<T>()?;
        let offset = i * element_value.shape()[axis];
        let count = element_value.shape()[axis].min(view.shape()[axis] - offset);
        view.slice_axis_mut(Axis(axis), (offset..offset + count).into())
            .assign(&element.slice_axis(Axis(axis), (..count).into()));
        Ok(())
    }
}

impl Op for Codegen {
    fn name(&self) -> Cow<str> {
        "Codegen".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &Model)> {
        vec![("loop".into(), self.plan.model())]
    }
}

impl StatelessOp for Codegen {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        // initialize state
        let mut state: TVec<Tensor> = tvec!();
        for input in &self.input_mapping {
            if let InputMapping::State { initializer } = input {
                state.push(match initializer {
                    StateInitializer::FromInput(slot) => (*inputs[*slot]).to_owned(),
                    StateInitializer::Value(v) => (**v).to_owned(),
                });
            }
        }

        let (inner_ix, outside_slot, axis, chunk) = self
            .input_mapping
            .iter()
            .enumerate()
            .filter_map(|(ix, it)| match it {
                InputMapping::Scan { axis, slot, chunk } => Some((ix, *slot, *axis, *chunk)),
                _ => None,
            })
            .next()
            .unwrap();

        let iters = inputs[outside_slot].shape()[axis].div_ceil(chunk);

        let mut scan_outputs = tvec!();
        for i in 0..(self.plan.model().output_outlets()?.len() - self.hidden_state_len) {
            let fact = self.plan.model().output_fact(self.hidden_state_len + i)?;
            let dt = fact.datum_type;
            let mut shape: TVec<usize> = fact.shape.as_finite().unwrap().into();
            let axis = self.scan_output_axes.get(i).cloned().unwrap_or(0);
            let scanning_dim = self
                .scan_output_len_hint
                .get(0)
                .and_then(|x| x.as_ref())
                .and_then(|i| i.to_integer().ok())
                .map(|i| i as usize)
                .unwrap_or(shape[axis] * iters);
            shape[axis] = scanning_dim;
            let t = dispatch_datum!(Self::alloc_output_t(dt)(self, &*shape))?;
            scan_outputs.push(t);
        }

        for i in 0..iters {
            state.reverse();

            let iter_inputs: TVec<Tensor> = self
                .input_mapping
                .iter()
                .map(|m| Ok(match m {
                    InputMapping::State { .. } => state.pop().unwrap(),
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
                }))
                .collect::<TractResult<_>>()?;

            trace!("iter_inputs: {:?}", iter_inputs);
            let mut iter_outputs =
                self.plan.run(iter_inputs).chain_err(|| "Evaluating inner body")?;

            for _ in 0..self.hidden_state_len {
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
