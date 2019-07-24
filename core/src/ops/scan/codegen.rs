use crate::internal::*;
use crate::plan::SimplePlan;

use ndarray::*;

#[derive(Debug, Clone, new)]
pub struct Codegen {
    pub plan: Arc<SimplePlan<TypedTensorInfo, Box<Op>, ModelImpl<TypedTensorInfo, Box<Op>>>>,
    pub(super) closure_inputs: usize,
    pub(super) scan_input_axes: Vec<usize>,
    pub(super) scan_output_axes: Vec<usize>,
    pub(super) scan_output_len_hint: Vec<Option<TDim>>,
}

impl Codegen {
    pub(super) fn slice_input_t<T: Datum>(
        &self,
        scan_inputs: &[Arc<Tensor>],
        input: usize,
        i: usize,
        count: usize,
    ) -> TractResult<Tensor> {
        let view = scan_inputs[input].to_array_view::<T>()?;
        let axis = Axis(self.scan_input_axes.get(input).cloned().unwrap_or(0));
        let full_len = view.shape()[axis.0];
        if (i + 1) * count > full_len {
            let remain = full_len - i * count;
            let mut shape: TVec<usize> = view.shape().into();
            shape[axis.0] = count;
            let mut t = ArrayD::<T>::default(&*shape);
            t.slice_axis_mut(axis, (0..remain).into())
                .assign(&view.slice_axis(axis, (i * count..).into()));
            Ok(t.into_tensor())
        } else {
            Ok(view.slice_axis(axis, (i * count..(i + 1) * count).into()).to_owned().into_tensor())
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
        let hidden_state_len = inputs.len() - self.scan_input_axes.len() - self.closure_inputs;

        // extract hidden state original values from inputs
        let mut state: TVec<Tensor> = tvec!();
        for _ in 0..hidden_state_len {
            state.push(inputs.remove(0).into_tensor());
        }

        let first_input_axis = self.scan_input_axes.get(0).cloned().unwrap_or(0);
        let iters = inputs[0].shape()[first_input_axis].div_ceil(
            self.plan
                .model()
                .input_fact(hidden_state_len)?
                .shape
                .dim(first_input_axis)
                .to_integer()? as usize,
        );
        let mut scan_outputs = tvec!();
        for i in 0..(self.plan.model().output_outlets()?.len() - hidden_state_len) {
            let fact = self.plan.model().output_fact(hidden_state_len + i)?;
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
            // body inputs are state + one slice of each input
            let mut iter_inputs: TVec<Tensor> = state.drain().collect();
            for input in 0..self.scan_input_axes.len() {
                let fact = self.plan.model().input_fact(input + hidden_state_len)?;
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
            let mut iter_outputs =
                self.plan.run(iter_inputs).chain_err(|| "Evaluating inner body")?;

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
