use std::borrow::Borrow;
use std::marker::PhantomData;

use model::{eval_order, Model, Node};
use ops::prelude::*;

#[derive(Debug, Clone)]
pub struct SimplePlan<M: Borrow<Model>> {
    pub model: M,
    pub order: Vec<usize>,
}

impl<M: Borrow<Model>> SimplePlan<M> {
    pub fn new(model: M) -> TfdResult<SimplePlan<M>> {
        let order = eval_order(model.borrow())?;
        Ok(SimplePlan { model, order })
    }

    pub fn run(&self, inputs: TVec<Tensor>) -> TfdResult<TVec<Tensor>> {
        let mut state = SimpleState::new(self)?;
        state.run(inputs)
    }

    pub fn model(&self) -> &Model {
        self.model.borrow()
    }
}

#[derive(Debug)]
pub struct SimpleState<M: Borrow<Model>, P: Borrow<SimplePlan<M>>> {
    plan: P,
    pub states: Vec<Option<Box<OpState>>>,
    pub values: Vec<Option<TVec<Value>>>,
    _phantom: PhantomData<M>,
}

impl<M: Borrow<Model>, P: Borrow<SimplePlan<M>> + Clone> Clone for SimpleState<M, P> {
    fn clone(&self) -> SimpleState<M, P> {
        let states = self
            .states
            .iter()
            .map(|opt: &Option<Box<OpState>>| -> Option<Box<OpState>> {
                opt.as_ref().map(|b| objekt::clone_box(&**b))
            }).collect();
        SimpleState {
            plan: self.plan.clone(),
            states,
            values: self.values.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<M: Borrow<Model>, P: Borrow<SimplePlan<M>>> SimpleState<M, P> {
    pub fn new(plan: P) -> TfdResult<SimpleState<M, P>> {
        let values = vec![None; plan.borrow().model.borrow().nodes().len()];
        let states = plan
            .borrow()
            .model()
            .nodes()
            .iter()
            .map(|n| n.op().state())
            .collect::<TfdResult<_>>()?;
        Ok(SimpleState {
            states,
            plan,
            values,
            _phantom: PhantomData,
        })
    }

    /// Reset wires state.
    pub fn reset_wires(&mut self) -> TfdResult<()> {
        self.values.iter_mut().for_each(|s| *s = None);
        Ok(())
    }

    /// Reset wires state.
    pub fn reset_op_states(&mut self) -> TfdResult<()> {
        self.states = self
            .plan
            .borrow()
            .model()
            .nodes()
            .iter()
            .map(|n| n.op().state())
            .collect::<TfdResult<_>>()?;
        Ok(())
    }

    pub fn run(&mut self, inputs: TVec<Tensor>) -> TfdResult<TVec<Tensor>> {
        use ops::source::Source;
        let mut result = tvec!();
        {
            let &mut SimpleState {
                ref plan,
                ref mut states,
                ref mut values,
                ..
            } = self;
            let model = plan.borrow().model();
            for (input, v) in model.inputs()?.iter().zip(inputs.into_iter()) {
                values[input.node] = Some(tvec!(v.into()));
            }
            for n in plan.borrow().order.iter() {
                let node: &Node = model.node(*n);
                if node.op_as::<Source>().is_none() {
                    let mut inputs: TVec<Value> = tvec![];
                    for i in &node.inputs {
                        let prec_node = model.node(i.node);
                        let prec = values[i.node].as_ref().ok_or_else(|| {
                            format!(
                                "Computing {}, precursor {} not done:",
                                node.name, prec_node.name
                            )
                        })?;
                        inputs.push(prec[i.slot].clone().into())
                    }
                    let vs = match states[node.id] {
                        Some(ref mut state) => state.eval(node.op(), inputs),
                        None => node.op().as_stateless().unwrap().eval(inputs),
                    }.map_err(|e| format!("Evaluating {} ({}): {}", node.id, node.name, e))?;

                    values[node.id] = Some(vs);
                }
            }
            for output in model.outputs()? {
                let mut val = Value::from(Tensor::from(0f32));
                ::std::mem::swap(
                    &mut val,
                    &mut values[output.node].as_mut().unwrap()[output.slot],
                );
                result.push(val.into_tensor());
            }
        }
        self.reset_wires()?;
        Ok(result)
    }

    pub fn set_inputs(&mut self, inputs: TVec<Tensor>) -> TfdResult<()> {
        let SimpleState {
            ref plan,
            ref mut values,
            ..
        } = self;
        plan.borrow()
            .model()
            .inputs()?
            .iter()
            .zip(inputs)
            .for_each(|(input, t)| values[input.node] = Some(tvec![t.into()]));
        Ok(())
    }

    pub fn set_input(&mut self, input: usize, t: Tensor) -> TfdResult<()> {
        let id = self.model().inputs()?[input].node;
        self.values[id] = Some(tvec![t.into()]);
        Ok(())
    }

    pub fn take_outputs(&mut self) -> TfdResult<Vec<Tensor>> {
        let SimpleState {
            ref plan,
            ref mut values,
            ..
        } = self;
        let mut v = vec![];
        for o in plan.borrow().model().outputs()?.iter() {
            let vs = values[o.node].as_mut().ok_or_else(|| {
                format!(
                    "Value for {:?} is not computed",
                    &plan.borrow().model().nodes()[o.node]
                )
            })?;
            let mut replacement: Value = Tensor::from(::std::f32::NAN).into();
            ::std::mem::swap(&mut replacement, &mut vs[o.slot]);
            v.push(replacement.into_tensor())
        }
        Ok(v)
    }

    pub fn set_values(&mut self, id: usize, values: TVec<Tensor>) -> TfdResult<()> {
        self.values[id] = Some(values.into_iter().map(|t| t.into()).collect());
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: Tensor) -> TfdResult<()> {
        self.set_values(id, tvec!(value))
    }

    pub fn compute_one(&mut self, node: usize) -> TfdResult<()> {
        let SimpleState {
            ref plan,
            ref mut values,
            ..
        } = self;
        let plan = plan.borrow();
        let nodes = plan.model().nodes();
        let node: &Node = &nodes[node];
        let mut inputs: TVec<Value> = tvec![];
        for i in &node.inputs {
            let prec_node = &nodes[i.node];
            let prec = values[i.node].as_ref().ok_or_else(|| {
                format!(
                    "Computing {}, precursor {} not done:",
                    node.name, prec_node.name
                )
            })?;
            inputs.push(prec[i.slot].clone().into())
        }
        let vs = match self.states[node.id] {
            Some(ref mut state) => state.eval(node.op(), inputs),
            None => node.op().as_stateless().unwrap().eval(inputs),
        }.map_err(|e| format!("Evaluating {} ({}): {}", node.id, node.name, e))?;
        values[node.id] = Some(vs);
        Ok(())
    }

    pub fn compute_recursively(&mut self, node: usize) -> TfdResult<()> {
        let values = {
            let precs: Vec<usize> = self.model().nodes()[node]
                .inputs
                .iter()
                .map(|i| i.node)
                .collect();
            for i in precs.into_iter() {
                if self.values[i].is_none() {
                    self.compute_recursively(i)?
                }
            }
            let mut inputs: TVec<Value> = tvec![];
            {
                let node: &Node = &self.model().nodes()[node];
                for i in &node.inputs {
                    inputs.push(self.values[i.node].as_ref().unwrap()[i.slot].clone().into())
                }
            }
            let Self {
                ref mut states,
                ref plan,
                ..
            } = self;
            match states[node] {
                Some(ref mut state) => state.eval(plan.borrow().model().nodes()[node].op(), inputs),
                None => plan.borrow().model().nodes()[node].op().as_stateless().unwrap().eval(inputs),
            }.map_err(|e| format!("Evaluating {:?}: {:?}", node, e))?
        };
        self.values[node] = Some(values);
        Ok(())
    }

    pub fn take_by_name(&mut self, name: &str) -> TfdResult<TVec<Tensor>> {
        let id = self.model().node_by_name(name)?.id;
        Self::take(self, id)
    }

    pub fn take(&mut self, id: usize) -> TfdResult<TVec<Tensor>> {
        Ok(self.values[id]
            .take()
            .ok_or("Value is not computed")?
            .into_iter()
            .map(Value::into_tensor)
            .collect())
    }

    pub fn plan(&self) -> &SimplePlan<M> {
        &self.plan.borrow()
    }

    pub fn model(&self) -> &Model {
        self.plan().model()
    }
}
