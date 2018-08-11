use std::ops::Deref;
use std::sync::Arc;

use Result;
use tensor::Tensor;
use model::{ Model, Node, eval_order_for_nodes };
use ops::{ TensorView };

#[derive(Debug,Clone)]
pub struct RawSimplePlan {
    pub model: Model,
    pub input_ids: Vec<usize>,
    pub output_ids: Vec<usize>,
    pub order: Vec<usize>,
}

impl RawSimplePlan {
    pub fn new(model:&Model, inputs: &[impl AsRef<str>], outputs: &[impl AsRef<str>]) -> Result<RawSimplePlan> {
        let input_ids:Vec<usize> = inputs.iter().map(|n| model.node_id_by_name(n.as_ref())).collect::<Result<_>>()?;
        let output_ids:Vec<usize> = outputs.iter().map(|n| model.node_id_by_name(n.as_ref())).collect::<Result<_>>()?;
        let order = eval_order_for_nodes(&model.nodes(), &*output_ids)?;
        Ok(RawSimplePlan { model:model.clone(), order, input_ids, output_ids })
    }
}

#[derive(Debug,Clone)]
pub struct SimplePlan(Arc<RawSimplePlan>);

impl Deref for SimplePlan {
    type Target = RawSimplePlan;
    fn deref(&self) -> &RawSimplePlan {
        &self.0
    }
}

impl SimplePlan {
    pub fn new(model:&Model, inputs: &[impl AsRef<str>], outputs: &[impl AsRef<str>]) -> Result<SimplePlan> {
        Ok(SimplePlan(Arc::new(RawSimplePlan::new(model, inputs, outputs)?)))
    }

    pub fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Vec<Tensor>>> {
        let mut state = SimpleState::new(&self)?;
        inputs.into_iter().zip(self.input_ids.iter()).try_for_each(|(t, &id)| state.set_value(id, t))?;
        for &n in &self.order {
            if state.outputs[n].is_none() {
                state.compute_one(n)?;
            }
        }
        self.output_ids.iter().map(|&id| state.take(id)).collect::<Result<Vec<Vec<Tensor>>>>()
    }
}

#[derive(Clone,Debug)]
pub struct SimpleState {
    plan: SimplePlan,
    pub outputs: Vec<Option<Vec<TensorView>>>,
}

impl SimpleState {
    pub fn new(plan: &SimplePlan) -> Result<SimpleState> {
        Ok(SimpleState {
            plan: plan.clone(),
            outputs: vec![None; plan.model.nodes.len()],
        })
    }

    /// Reset internal state.
    pub fn reset(&mut self) -> Result<()> {
        self.outputs.iter_mut().for_each(|s| *s = None);
        Ok(())
    }

    pub fn set_outputs(&mut self, id: usize, values: Vec<Tensor>) -> Result<()> {
        self.outputs[id] = Some(values.into_iter().map(TensorView::Owned).collect());
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: Tensor) -> Result<()> {
        self.set_outputs(id, vec![value])
    }

    pub fn set_values(&mut self, values: Vec<(&str, Tensor)>) -> Result<()> {
        for (name, mat) in values {
            let id = self.model().node_id_by_name(name)?;
            self.set_value(id, mat)?;
        }

        Ok(())
    }

    pub fn compute_one(&mut self, node: usize) -> Result<()> {
        let node: &Node = &self.plan.model.nodes[node];
        let mut inputs: Vec<TensorView> = vec![];
        for i in &node.inputs {
            let prec_node = &self.model().nodes[i.0];
            let prec = self.outputs[i.0].as_ref().ok_or(format!(
                "Computing {}, precursor {} not done:",
                node.name, prec_node.name
            ))?;
            inputs.push(prec[i.1].clone().into())
        }
        let outputs = node.op.eval(inputs)?;
        self.outputs[node.id] = Some(outputs);
        Ok(())
    }

    pub fn take_by_name(&mut self, name: &str) -> Result<Vec<Tensor>> {
        let id = self.model().node_id_by_name(name)?;
        Self::take(self, id)
    }

    pub fn take(&mut self, id: usize) -> Result<Vec<Tensor>> {
        Ok(self.outputs[id]
            .take()
            .ok_or("Value is not computed")?
            .into_iter()
            .map(TensorView::into_tensor)
            .collect())
    }

    pub fn model(&self) -> &Model {
        &self.plan.model
    }
}

