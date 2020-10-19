use tract_hir::internal::*;

use crate::model::{ParsingContext, TfOpRegister};
use crate::tfpb::tensorflow::NodeDef;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Assign", |_, _| Ok(Box::new(Assign::default())));
    reg.insert("VariableV2", variable_v2);
}

fn variable_v2(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let shared_name = node.get_attr_str("shared_name")?;
    let shared_name = if shared_name != "" { Some(shared_name) } else { None };
    let container = node.get_attr_str("container")?;
    let container = if container != "" { Some(container) } else { None };
    let name = node.name.to_string();
    let id = format!("{:?}#{:?}#{}", container, shared_name, name);
    let shape = node.get_attr_shape("shape")?;
    let dt = node.get_attr_datum_type("dtype")?;
    let shape = shape
        .into_iter()
        .map(|d| {
            if d > 0 {
                Ok(d as usize)
            } else {
                bail!("VariableV2 shape contains forbidden negative dim.")
            }
        })
        .collect::<TractResult<TVec<usize>>>()?;
    Ok(Box::new(VariableV2::new(container, shared_name, name, id, shape, dt, None)))
}

#[derive(Clone, Debug, new)]
struct VariableV2State;

impl OpState for VariableV2State {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        _inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op
            .downcast_ref::<VariableV2>()
            .context("wrong op for variable state")?;
        let tensor = session
            .tensors
            .get(&op.id)
            .with_context(|| format!("Could not find state for variable {}", op.id))?;
        Ok(tvec!(tensor.clone().into()))
    }
}

#[derive(Clone, Debug, new, Hash)]
pub struct VariableV2 {
    container: Option<String>,
    shared_name: Option<String>,
    name: String,
    pub id: String,
    shape: TVec<usize>,
    dt: DatumType,
    pub initializer: Option<Arc<Tensor>>,
}

tract_data::impl_dyn_hash!(VariableV2);

impl Op for VariableV2 {
    fn name(&self) -> Cow<str> {
        "VariableV2".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        if let Some(init) = &self.initializer {
            Ok(vec![format!("Initialized to {:?}", init)])
        } else {
            Ok(vec![format!("Uninitialized")])
        }
    }

    op_tf!();
    op_as_typed_op!();
}

impl EvalOp for VariableV2 {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        state: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        let tensor = if let Some(init) = &self.initializer {
            init.clone().into_tensor()
        } else {
            unsafe { Tensor::uninitialized_dt(self.dt, &self.shape)? }
        };
        state.tensors.insert(self.id.clone(), tensor);
        Ok(Some(Box::new(VariableV2State)))
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
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&outputs[0].shape, ShapeFactoid::from(&*self.shape))?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for VariableV2 {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(self.dt, &*self.shape)?))
    }
}

// need some dummy state to make sure Assign is a EvalOp, and will not be
// eval-ed() in Stateless context
#[derive(Clone, Debug, new)]
struct AssignState;

#[derive(Clone, Debug, new, Default, Hash)]
pub struct Assign {
    pub var_id: Option<String>,
}

tract_data::impl_dyn_hash!(Assign);

impl Op for Assign {
    fn name(&self) -> Cow<str> {
        "Assign".into()
    }

    op_tf!();
    op_as_typed_op!();
}

impl OpState for AssignState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let (_current, new) = args_2!(inputs);
        let op =
            op.downcast_ref::<Assign>().context("wrong op for variable state")?;
        let var_id = if let Some(ref var_id) = op.var_id {
            var_id
        } else {
            bail!("Assign has not been linked to var")
        };
        let store = session.tensors.get_mut(var_id).unwrap();
        if cfg!(debug_assertions)
            && (store.shape() != new.shape() && store.datum_type() != new.datum_type())
        {
            bail!(
                "Invalid assignment to variable. Store is {:?}, assigned value is {:?}",
                store,
                new
            );
        }
        *store = new.clone().into_tensor();
        Ok(tvec!(new))
    }
}

impl EvalOp for Assign {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _state: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
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
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].shape, &inputs[1].shape)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].shape, &inputs[0].shape)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        s.equals(&outputs[0].value, &inputs[1].value)?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for Assign {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].datum_type != inputs[1].datum_type || inputs[0].shape != inputs[1].shape {
            bail!("Invalid assignement {:?}", inputs);
        }
        Ok(tvec!(inputs[0].clone()))
    }
}
