use std::sync::{Arc, Mutex};

use tract_core::ops::prelude::*;

use crate::tfpb::node_def::NodeDef;
use crate::model::TfOpRegister;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("VariableV2", variable_v2);
}

fn variable_v2(node: &NodeDef) -> TractResult<Box<Op>> {
    let shared_name = node.get_attr_str("shared_name")?;
    let shared_name = if shared_name != "" {
        Some(shared_name)
    } else {
        None
    };
    let container = node.get_attr_str("container")?;
    let container = if container != "" {
        Some(container)
    } else {
        None
    };
    let name = node.get_name().to_string();
    let shape = node.get_attr_shape("shape")?;
    let dt = node.get_attr_datum_type("dtype")?;
    Ok(Box::new(VariableV2::new(
        container,
        shared_name,
        name,
        shape,
        dt,
    )))
}

#[derive(Clone, Debug, new)]
struct VariableV2State(Arc<Mutex<Tensor>>);

fn make_buffer<T: Copy + Datum>(shape: &[usize]) -> Tensor {
    ::ndarray::ArrayD::<T>::default(shape).into()
}

impl OpState for VariableV2State {
    fn eval(&mut self, _session: &mut SessionState, op: &Op, _inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let op = op.downcast_ref::<VariableV2>().ok_or_else(|| format!("wrong of for variable state"))?;
        let locked = self.0.lock().map_err(|_| format!("poisoned lock on variable {}", op.name))?;
        Ok(tvec!(locked.clone().into()))
    }
}

#[derive(Clone, Debug, new)]
struct VariableV2 {
    container: Option<String>,
    shared_name: Option<String>,
    name: String,
    shape: TVec<usize>,
    dt: DatumType,
}

impl Op for VariableV2 {
    fn name(&self) -> Cow<str> {
        "tf.VariableV2".into()
    }
}

impl StatefullOp for VariableV2 {
    fn state(&self) -> TractResult<Option<Box<OpState>>> {
        let tensor = dispatch_copy!(self::make_buffer(self.dt)(&self.shape));
        Ok(Some(Box::new(VariableV2State(Arc::new(Mutex::new(tensor))))))
    }
}

impl InferenceRulesOp for VariableV2 {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 0)?;
        check_output_arity(inputs, 0)?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.shape))?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        Ok(())
    }
}
