use std::fmt;

use crate::internal::*;
use ndarray::prelude::*;

// Generic node outer interface:
// inputs: [ hidden_state_len initial values ][ num_scan_inputs inputs ][ implicit capture inputs ]
// outputs: [ hidden_state_len final values ][ aggregated outputs ]

/*
#[derive(Debug, Clone, new, Default)]
pub struct Generic<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: fmt::Debug + fmt::Display + AsRef<Op> + AsMut<Op> + Clone + 'static,
    ModelImpl<TI, O>: Model,
{
    pub body: Arc<ModelImpl<TI, O>>,
    pub(super) num_scan_inputs: usize,
    pub(super) closure_inputs: usize,
    pub(super) scan_input_axes: Vec<usize>,
    pub(super) scan_output_axes: Vec<usize>,
    pub(super) scan_output_len_hint: Vec<Option<TDim>>,
    pub(super) prune_scanning_dim: bool, // TODO check scanning dims == 1
}

*/
