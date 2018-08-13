use std::ops::Deref;
use std::sync::Arc;

use model::{eval_order_for_nodes, Model, Node};
use ops::Value;
use tensor::Tensor;
use Result;

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
    ) -> Result<RawSimplePlan> {
        let input_ids: Vec<usize> = inputs
            .iter()
            .map(|n| Ok(model.node_by_name(n.as_ref())?.id))
            .collect::<Result<_>>()?;
        let output_ids: Vec<usize> = outputs
            .iter()
            .map(|n| Ok(model.node_by_name(n.as_ref())?.id))
            .collect::<Result<_>>()?;
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
    ) -> Result<SimplePlan> {
        Ok(SimplePlan(Arc::new(RawSimplePlan::new(
            model, inputs, outputs,
        )?)))
    }

    pub fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Vec<Tensor>>> {
        let mut state = SimpleState::new(&self)?;
        state.set_inputs(inputs)?;
        for &n in &self.order {
            if state.values[n].is_none() {
                state.compute_one(n)?;
            }
        }
        state.take_outputs()
    }
}

#[derive(Clone, Debug)]
pub struct SimpleState {
    plan: SimplePlan,
    pub values: Vec<Option<Vec<Value>>>,
}

impl SimpleState {
    pub fn new(plan: &SimplePlan) -> Result<SimpleState> {
        Ok(SimpleState {
            plan: plan.clone(),
            values: vec![None; plan.model.nodes.len()],
        })
    }

    /// Reset internal state.
    pub fn reset(&mut self) -> Result<()> {
        self.values.iter_mut().for_each(|s| *s = None);
        Ok(())
    }

    pub fn set_inputs(&mut self, inputs: Vec<Tensor>) -> Result<()> {
        let SimpleState {
            ref plan,
            ref mut values,
        } = self;
        inputs
            .into_iter()
            .zip(plan.input_ids.iter())
            .for_each(|(t, &id)| values[id] = Some(vec![t.into()]));
        Ok(())
    }

    pub fn set_input(&mut self, input: usize, t: Tensor) -> Result<()> {
        let id = self.plan.input_ids[input];
        self.values[id] = Some(vec![t.into()]);
        Ok(())
    }

    pub fn take_outputs(&mut self) -> Result<Vec<Vec<Tensor>>> {
        let mut v = vec![];
        for &id in &self.plan.output_ids {
            v.push(
                self.values[id]
                    .take()
                    .ok_or("Value is not computed")?
                    .into_iter()
                    .map(Value::into_tensor)
                    .collect(),
            )
        }
        Ok(v)
    }

    pub fn set_value(&mut self, id: usize, value: Tensor) -> Result<()> {
        self.values[id] = Some(vec![value.into()]);
        Ok(())
    }

    pub fn compute_one(&mut self, node: usize) -> Result<()> {
        let node: &Node = &self.plan.model.nodes[node];
        let mut inputs: Vec<Value> = vec![];
        for i in &node.inputs {
            let prec_node = &self.model().nodes[i.0];
            let prec = self.values[i.0].as_ref().ok_or(format!(
                "Computing {}, precursor {} not done:",
                node.name, prec_node.name
            ))?;
            inputs.push(prec[i.1].clone().into())
        }
        let values = node.op.eval(inputs)?;
        self.values[node.id] = Some(values);
        Ok(())
    }

    pub fn take_by_name(&mut self, name: &str) -> Result<Vec<Tensor>> {
        let id = self.model().node_by_name(name)?.id;
        Self::take(self, id)
    }

    pub fn take(&mut self, id: usize) -> Result<Vec<Tensor>> {
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
