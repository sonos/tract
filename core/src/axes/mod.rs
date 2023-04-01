use crate::internal::*;

mod mapping;
mod model;

pub use mapping::AxesMapping;
pub use model::{for_model, full_axis_tracking};

#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct Axis {
    pub inputs: TVec<TVec<usize>>,
    pub outputs: TVec<TVec<usize>>,
    pub repr: char,
}

impl Axis {
    pub fn new(repr: char, inputs: usize, outputs: usize) -> Axis {
        Axis { repr, inputs: tvec!(tvec!(); inputs), outputs: tvec!(tvec!(); outputs) }
    }

    pub fn natural(inputs: usize, outputs: usize, repr: char, axis_id: usize) -> Axis {
        let inputs = tvec!(tvec!(axis_id); inputs);
        let outputs = tvec!(tvec!(axis_id); outputs);
        Axis { inputs, outputs, repr }
    }

    #[allow(dead_code)]
    pub fn input(mut self, input_id: usize, axis: usize) -> Axis {
        self.add_input(input_id, axis);
        self
    }

    pub fn output(mut self, output_id: usize, axis: usize) -> Axis {
        self.add_output(output_id, axis);
        self
    }

    pub fn inputs_count(mut self, inputs: usize) -> Axis {
        self.inputs.resize(inputs, tvec!());
        self
    }

    pub fn outputs_count(mut self, outputs: usize) -> Axis {
        self.outputs.resize(outputs, tvec!());
        self
    }

    pub fn ensure_inputs_count(&mut self, inputs: usize) {
        if self.inputs.len() < inputs {
            self.inputs.resize(inputs, tvec!())
        }
    }

    pub fn ensure_outputs_count(&mut self, outputs: usize) {
        if self.outputs.len() < outputs {
            self.outputs.resize(outputs, tvec!())
        }
    }

    pub fn add_input(&mut self, input_id: usize, axis: usize) {
        self.ensure_inputs_count(input_id + 1);
        self.inputs[input_id].push(axis);
    }

    pub fn add_output(&mut self, output_id: usize, axis: usize) {
        self.ensure_outputs_count(output_id + 1);
        self.outputs[output_id].push(axis);
    }

    pub fn interface(&self, io: InOut) -> &[usize] {
        match io {
            InOut::In(ix) => &self.inputs[ix],
            InOut::Out(ix) => &self.outputs[ix],
        }
    }
}
