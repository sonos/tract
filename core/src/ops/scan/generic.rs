use std::fmt;

use crate::internal::*;
use ndarray::prelude::*;

// Generic node outer interface:
// inputs: [ hidden_state_len initial values ][ num_scan_inputs inputs ][ implicit capture inputs ]
// outputs: [ hidden_state_len final values ][ aggregated outputs ]

#[derive(Debug, Clone, new, Default)]
pub struct Generic<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op> + Clone + 'static,
    ModelImpl<TI, O>: Model,
{
    pub body: ModelImpl<TI, O>,
    pub(super) num_scan_inputs: usize,
    pub(super) closure_inputs: usize,
    pub(super) scan_input_axes: Vec<usize>,
    pub(super) scan_output_axes: Vec<usize>,
    pub(super) scan_output_len_hint: Vec<Option<TDim>>,
    pub(super) prune_scanning_dim: bool, // TODO check scanning dims == 1
}

impl<TI, O> Generic<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op> + Clone + 'static,
    ModelImpl<TI, O>: Model,
{
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

    pub(super) fn alloc_output_t<T: Datum + Default>(&self, shape: &[usize]) -> TractResult<Tensor> {
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

