use std::ops::Deref;
use std::sync::Arc;

use model::{eval_order_for_nodes, Model, Node, TVec};
use ops::Value;
use tensor::Tensor;
use TfdResult;

#[derive(Debug, Clone)]
pub struct RawSimplePlan {
    pub model: Model,
    pub input_ids: Vec<usize>,
    pub output_ids: Vec<usize>,
    pub order: Vec<usize>,
}

impl RawSimplePlan {
    pub fn new(
        model: &Model,
        inputs: &[impl AsRef<str>],
        outputs: &[impl AsRef<str>],
    ) -> TfdResult<RawSimplePlan> {
        let input_ids: Vec<usize> = inputs
            .iter()
            .map(|n| Ok(model.node_by_name(n.as_ref())?.id))
            .collect::<TfdResult<_>>()?;
        let output_ids: Vec<usize> = outputs
            .iter()
            .map(|n| Ok(model.node_by_name(n.as_ref())?.id))
            .collect::<TfdResult<_>>()?;
        let order = eval_order_for_nodes(&model.nodes(), &*output_ids)?;
        Ok(RawSimplePlan {
            model: model.clone(),
            order,
            input_ids,
            output_ids,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SimplePlan(Arc<RawSimplePlan>);

impl Deref for SimplePlan {
    type Target = RawSimplePlan;
    fn deref(&self) -> &RawSimplePlan {
        &self.0
    }
}

impl SimplePlan {
    pub fn new(
        model: &Model,
        inputs: &[impl AsRef<str>],
        outputs: &[impl AsRef<str>],
    ) -> TfdResult<SimplePlan> {
        Ok(SimplePlan(Arc::new(RawSimplePlan::new(
            model, inputs, outputs,
        )?)))
    }

    pub fn run(&self, inputs: TVec<Tensor>) -> TfdResult<Vec<TVec<Tensor>>> {
        let mut state = SimpleState::new(&self)?;
        state.set_inputs(inputs)?;
        for &n in &self.order {
            if state.values[n].is_none() {
                state.compute_one(n)?;
            }
        }
        state.take_outputs()
    }

    pub fn state(&self) -> TfdResult<SimpleState> {
        SimpleState::new(self)
    }
}

#[derive(Clone, Debug)]
pub struct SimpleState {
    plan: SimplePlan,
    pub values: Vec<Option<TVec<Value>>>,
}

impl SimpleState {
    pub fn new(plan: &SimplePlan) -> TfdResult<SimpleState> {
        Ok(SimpleState {
            plan: plan.clone(),
            values: vec![None; plan.model.nodes().len()],
        })
    }

    /// Reset internal state.
    pub fn reset(&mut self) -> TfdResult<()> {
        self.values.iter_mut().for_each(|s| *s = None);
        Ok(())
    }

    pub fn set_inputs(&mut self, inputs: TVec<Tensor>) -> TfdResult<()> {
        let SimpleState {
            ref plan,
            ref mut values,
        } = self;
        inputs
            .into_iter()
            .zip(plan.input_ids.iter())
            .for_each(|(t, &id)| values[id] = Some(tvec![t.into()]));
        Ok(())
    }

    pub fn set_input(&mut self, input: usize, t: Tensor) -> TfdResult<()> {
        let id = self.plan.input_ids[input];
        self.values[id] = Some(tvec![t.into()]);
        Ok(())
    }

    pub fn take_outputs(&mut self) -> TfdResult<Vec<TVec<Tensor>>> {
        let mut v = vec![];
        for &id in &self.plan.output_ids {
            v.push(
                self.values[id]
                    .take()
                    .ok_or_else(|| format!("Value for {:?} is not computed", &self.model().nodes()[id]))?
                    .into_iter()
                    .map(Value::into_tensor)
                    .collect(),
            )
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
        let node: &Node = &self.plan.model.nodes()[node];
        let mut inputs: TVec<Value> = tvec![];
        for i in &node.inputs {
            let prec_node = &self.model().nodes()[i.node];
            let prec = self.values[i.node].as_ref().ok_or(format!(
                "Computing {}, precursor {} not done:",
                node.name, prec_node.name
            ))?;
            inputs.push(prec[i.slot].clone().into())
        }
        let values = node.op.eval(inputs)?;
        self.values[node.id] = Some(values);
        Ok(())
    }

    pub fn compute_recursively(&mut self, node: usize) -> TfdResult<()> {
        let precs: Vec<usize> = self.plan.model.nodes()[node]
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
        let node: &Node = &self.plan.model.nodes()[node];
        for i in &node.inputs {
            inputs.push(self.values[i.node].as_ref().unwrap()[i.slot].clone().into())
        }
        let values = node.op.eval(inputs)?;
        self.values[node.id] = Some(values);
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

    pub fn model(&self) -> &Model {
        &self.plan.model
    }
}
