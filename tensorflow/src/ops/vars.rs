use std::sync::{Arc, Mutex};

use tract_core::ops::prelude::*;

use crate::model::TfOpRegister;
use crate::tfpb::node_def::NodeDef;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Assign", |_| Ok(Box::new(Assign {})));
    reg.insert("VariableV2", variable_v2);
}

fn variable_v2(node: &NodeDef) -> TractResult<Box<Op>> {
    let shared_name = node.get_attr_str("shared_name")?;
    let shared_name = if shared_name != "" { Some(shared_name) } else { None };
    let container = node.get_attr_str("container")?;
    let container = if container != "" { Some(container) } else { None };
    let name = node.get_name().to_string();
    let shape = node.get_attr_shape("shape")?;
    let dt = node.get_attr_datum_type("dtype")?;
    Ok(Box::new(VariableV2::new(container, shared_name, name, shape, dt)))
}

#[derive(Clone, Debug, new)]
struct VariableV2State(Arc<Mutex<Tensor>>);

fn make_buffer<T: Copy + Datum>(shape: &[usize]) -> Tensor {
    ::ndarray::ArrayD::<T>::default(shape).into()
}

impl OpState for VariableV2State {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        op: &Op,
        _inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>> {
        let op = op
            .downcast_ref::<VariableV2>()
            .ok_or_else(|| format!("wrong of for variable state"))?;
        let state_ref = Tensor::from(self.0.as_ref() as *const _ as isize as i64);
        let locked = self.0.lock().map_err(|_| format!("poisoned lock on variable {}", op.name))?;
        Ok(tvec!(locked.clone().into(), state_ref.into()))
    }
}

#[derive(Clone, Debug, new)]
pub struct VariableV2 {
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
        check_output_arity(outputs, 2)?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&outputs[1].datum_type, i64::datum_type())?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.shape))?;
        s.equals(&outputs[1].rank, 0)?;
        Ok(())
    }
}

// need some dummy state to make sure Assign is a StatefullOp, and will not be
// eval-ed() in Stateless context
#[derive(Clone, Debug, new)]
struct AssignState;

#[derive(Clone, Debug, new, Default)]
pub struct Assign {}

impl Op for Assign {
    fn name(&self) -> Cow<str> {
        "tf.Assign".into()
    }
}

impl OpState for AssignState {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        _op: &Op,
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>> {
        let (_current, new, var_state) = args_3!(inputs);
        let var_state = *var_state.to_scalar::<i64>()? as isize;
        let var_state = unsafe {
            (var_state as *const Mutex<Tensor>).as_ref().ok_or("null pointer received from var")?
        };
        println!("assigning {:?}", new);
        let mut lock = var_state.lock().map_err(|_| "poisoned lock for assignment")?;
        *lock = new.clone().to_tensor();
        Ok(tvec!(new))
    }
}

impl StatefullOp for Assign {
    fn state(&self) -> TractResult<Option<Box<OpState>>> {
        Ok(Some(Box::new(AssignState)))
    }
}

impl InferenceRulesOp for Assign {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[2].datum_type, i64::datum_type())?;
        s.equals(&inputs[2].rank, 0)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].shape, &inputs[0].shape)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        s.equals(&outputs[0].value, &inputs[1].value)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::*;

    #[test]
    fn var_assign() {
        let mut model = Model::default();

        let var = model
            .add_node(
                "var",
                VariableV2::new(None, None, "var".into(), tvec![], f32::datum_type()),
                tvec!(TensorFact::default(), TensorFact::default()),
            )
            .unwrap();
        let zero = model.add_const("zero".to_string(), 0f32.into()).unwrap();
        let one = model.add_const("one".to_string(), 1f32.into()).unwrap();
        let reset = model.add_node_default("reset", Assign::default()).unwrap();
        model.add_edge(OutletId::new(var, 0), InletId::new(reset, 0)).unwrap();
        model.add_edge(OutletId::new(zero, 0), InletId::new(reset, 1)).unwrap();
        model.add_edge(OutletId::new(var, 1), InletId::new(reset, 2)).unwrap();
        let set = model.add_node_default("set", Assign::default()).unwrap();
        model.add_edge(OutletId::new(var, 0), InletId::new(set, 0)).unwrap();
        model.add_edge(OutletId::new(one, 0), InletId::new(set, 1)).unwrap();
        model.add_edge(OutletId::new(var, 1), InletId::new(set, 2)).unwrap();
        let model = std::rc::Rc::new(model);
        let plan_read = SimplePlan::new_for_output(model.clone(), OutletId::new(var, 0)).unwrap();
        let plan_set = SimplePlan::new_for_output(model.clone(), OutletId::new(set, 0)).unwrap();
        let plan_reset =
            SimplePlan::new_for_output(model.clone(), OutletId::new(reset, 0)).unwrap();
        let mut state = SimpleState::new_multiplan(vec![plan_read, plan_set, plan_reset]).unwrap();

        let read = state.run_plan(tvec!(), 0).unwrap(); // read
        assert_eq!(read, tvec!(Tensor::from(0.0f32).into()));
        let read = state.run_plan(tvec!(), 1).unwrap(); // set
        assert_eq!(read, tvec!(Tensor::from(1.0f32).into()));
        let read = state.run_plan(tvec!(), 0).unwrap(); // read
        assert_eq!(read, tvec!(Tensor::from(1.0f32).into()));
        let read = state.run_plan(tvec!(), 2).unwrap(); // reset
        assert_eq!(read, tvec!(Tensor::from(0.0f32).into()));
        let read = state.run_plan(tvec!(), 0).unwrap(); // read
        assert_eq!(read, tvec!(Tensor::from(0.0f32).into()));
    }
}
