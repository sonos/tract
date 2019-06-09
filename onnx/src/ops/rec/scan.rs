use tract_core::ndarray::prelude::*;
use tract_core::internal::*;
use crate::model::{ ParseResult, ParsingContext };
use crate::pb::*;

pub fn scan(ctx: &ParsingContext, node: &NodeProto) -> TractResult<(Box<InferenceOp>, Vec<String>)> {
    let num_scan_inputs = node.get_attr("num_scan_inputs")?;
    let graph: &GraphProto = node.get_attr("body")?;
    let scan_input_axes = node.get_attr_opt_vec("scan_input_axes")?.unwrap_or(Vec::<usize>::new());
    let scan_output_axes = node.get_attr_opt_vec("scan_output_axes")?.unwrap_or(Vec::<usize>::new());
    let ParseResult { model, unresolved_inputs, .. } = ctx.parse_graph(graph)?;
    Ok((Box::new(Scan::new(model, num_scan_inputs,  unresolved_inputs.len(), scan_input_axes, scan_output_axes)), unresolved_inputs))
}

// Scan node outer interface:
// inputs: [ hidden_state_len initial values ][ num_scan_inputs inputs ][ implicit capture inputs ]
// outputs: [ hidden_state_len final values ][ aggregated outputs ]

#[derive(Debug, Clone, new, Default)]
pub struct Scan {
    body: InferenceModel,
    num_scan_inputs: usize,
    closure_inputs: usize,
    scan_input_axes: Vec<usize>,
    scan_output_axes: Vec<usize>,
}

impl Scan {
    fn slice_input_t<T:Datum>(&self, scan_inputs: &[Arc<Tensor>], input: usize, i: usize) -> TractResult<Tensor> {
        let view = scan_inputs[input].to_array_view::<T>()?;
        let axis = self.scan_input_axes.get(input).cloned().unwrap_or(0);
        let slice = view.index_axis_move(Axis(axis), i);
        Ok(slice.to_owned().into_tensor())
    }

    fn alloc_output_t<T:Datum+Default>(&self, element_shape: &[usize], output: usize, iters: usize) -> TractResult<Tensor> {
        let axis = self.scan_output_axes.get(output).cloned().unwrap_or(0);
        let mut shape = element_shape.to_vec();
        shape.insert(axis, iters);
        Ok(Array::<T,_>::default(&*shape).into())
    }

    fn assign_output_t<T:Datum+Default>(&self, output: &mut Tensor, output_id: usize, element_value: &Tensor, i: usize) -> TractResult<()> {
        let axis = self.scan_output_axes.get(output_id).cloned().unwrap_or(0);
        let view = output.to_array_view_mut::<T>()?;
        let mut slice = view.index_axis_move(Axis(axis), i);
        let element = element_value.to_array_view::<T>()?;
        slice.assign(&element);
        Ok(())
    }
}

impl Op for Scan {
    fn name(&self) -> Cow<str> {
        "onnx.Scan".into()
    }
}

impl StatelessOp for Scan {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let hidden_state_len = inputs.len() - self.num_scan_inputs - self.closure_inputs;

        // extract hidden state original values from inputs
        let mut state:TVec<Tensor> = tvec!();
        for _ in 0..hidden_state_len {
            state.push(inputs.remove(0).into_tensor());
        }

        let iters = inputs[0].shape()[self.scan_input_axes.get(0).cloned().unwrap_or(0)];

        let mut scan_outputs = tvec!();
        for i in 0..(self.body.output_outlets()?.len() - hidden_state_len) {
            let fact = self.body.output_fact(hidden_state_len + i)?;
            let dt = fact.datum_type.concretize().unwrap();
            let shape = fact.shape.as_concrete_finite().unwrap().unwrap();
            let t = dispatch_datum!(Self::alloc_output_t(dt)(self, &*shape, i, iters))?;
            scan_outputs.push(t);
        }

        let plan = SimplePlan::new(&self.body)?;

        for i in 0..iters {
            // body inputs are state + one slice of each input
            let mut iter_inputs:TVec<Tensor> = state.drain().collect();
            for input in 0..self.num_scan_inputs {
                let tensor = dispatch_datum!(Self::slice_input_t(inputs[input].datum_type())(self, &*inputs, input, i))?;
                iter_inputs.push(tensor);
            }
            for i in 0..self.closure_inputs {
                iter_inputs.push(inputs[inputs.len() - self.closure_inputs + i].clone().into_tensor());
            }
            let mut iter_outputs = plan.run(iter_inputs)?;
            for _ in 0..hidden_state_len {
                state.push(iter_outputs.remove(0).into_tensor());
            }
            for (ix, o) in scan_outputs.iter_mut().enumerate() {
                dispatch_datum!(Self::assign_output_t(o.datum_type())(self, o, ix, &iter_outputs[ix], i))?;
            }
        }
        let mut output:TVec<Arc<Tensor>> = state.into_iter().map(|t| t.into_arc_tensor()).collect();
        output.extend(scan_outputs.into_iter().map(|t| t.into_arc_tensor()));
        Ok(output)
    }
}

impl InferenceRulesOp for Scan {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let body_inputs = self.body.input_outlets()?.len();
        let body_outputs = self.body.output_outlets()?.len();
        let hidden_state_len = body_inputs - self.num_scan_inputs;
        if body_inputs != inputs.len() {
            bail!("Unexpected inputs count: body expects {} inputs, interface have {}", body_inputs, inputs.len())
        };
        if body_outputs != outputs.len() {
            bail!("Unexpected outputs count: body expects {} outputs, interface have {}", body_outputs, outputs.len())
        };
        for i in 0..hidden_state_len {
            s.equals(&inputs[i].shape, &outputs[i].shape)?;
            s.equals(&inputs[i].datum_type, &outputs[i].datum_type)?;
        }
        Ok(())
    }

    inference_op_as_op!();
}
