use crate::ops::prelude::*;

pub use super::{InletId, Model, Node, OutletId};

pub trait ModelDsl<TI: TensorInfo> {
    fn single_prec(&self, id: usize) -> TractResult<Option<&Node<TI>>>;
    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<TI>>>;
    fn single_succ(&self, id: usize) -> TractResult<Option<&Node<TI>>>;
    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<TI>>>;

    fn add_source(&mut self, name: impl Into<String>, fact: TI) -> TractResult<usize>;
    fn chain(&mut self, name: impl Into<String>, op: impl Into<Box<Op>>, facts:TVec<TI>) -> TractResult<usize>;

    fn chain_after(
        &mut self,
        tap: OutletId,
        name: impl Into<String>,
        op: impl Into<Box<Op>>,
        facts: TVec<TI>,
    ) -> TractResult<usize>;

}

impl<TI: TensorInfo> ModelDsl<TI> for Model<TI> {
    fn add_source(&mut self, name: impl Into<String>, fact: TI) -> TractResult<usize> {
        let id = self.add_node(
            name,
            crate::ops::source::Source::new(fact.to_tensor_fact()),
            tvec!(fact),
        )?;
        Ok(id)
    }

    fn chain(&mut self, name:impl Into<String>, op: impl Into<Box<Op>>, facts:TVec<TI>) -> TractResult<usize> {
        let previous_id = self.nodes().len() - 1;
        self.chain_after(OutletId::new(previous_id, 0), name, op.into(), facts)
    }

    fn single_prec(&self, id: usize) -> TractResult<Option<&Node<TI>>> {
        let node = &self.nodes()[id];
        if node.inputs.len() != 1 {
            return Ok(None);
        }
        let prec = &self.nodes()[node.inputs[0].node];
        if prec.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
            return Ok(None);
        }
        Ok(Some(prec))
    }

    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<TI>>> {
        let mut node = self.node(id);
        for _ in 0..count {
            if let Some(next) = self.single_prec(node.id)? {
                node = next
            } else {
                return Ok(None);
            }
        }
        Ok(Some(node))
    }

    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<TI>>> {
        let mut node = self.node(id);
        for _ in 0..count {
            if let Some(next) = self.single_succ(node.id)? {
                node = next
            } else {
                return Ok(None);
            }
        }
        Ok(Some(node))
    }

    fn single_succ(&self, id: usize) -> TractResult<Option<&Node<TI>>> {
        let node = &self.nodes()[id];
        if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
            return Ok(None);
        }
        let succ = node.outputs[0].successors[0];
        let succ = &self.nodes()[succ.node];
        if succ.inputs.len() != 1 {
            return Ok(None);
        }
        Ok(Some(succ))
    }

    fn chain_after(
        &mut self,
        tap: OutletId,
        name: impl Into<String>,
        op: impl Into<Box<Op>>,
        facts: TVec<TI>,
    ) -> TractResult<usize> {
        let id = self.add_node(name, op, facts)?;
        self.add_edge(tap, InletId::new(id, 0))?;
        Ok(id)
    }

}

pub trait ModelDslConst {
    fn add_const(&mut self, name: impl Into<String>, v: SharedTensor) -> TractResult<usize>;
    fn plug_const(&mut self, inlet: InletId, name: impl Into<String>, v: SharedTensor) -> TractResult<()>;
}

impl ModelDslConst for super::InferenceModel {
    fn add_const(&mut self, name: impl Into<String>, v: SharedTensor) -> TractResult<usize> {
        let facts = tvec!(v.clone().into());
        self.add_node(name, crate::ops::konst::Const::new(v), facts)
    }
    fn plug_const(&mut self, inlet: InletId, name: impl Into<String>, v: SharedTensor) -> TractResult<()> {
        let cst = self.add_const(name, v)?;
        self.add_edge(OutletId::new(cst, 0), inlet)?;
        Ok(())
    }
}

impl ModelDslConst for super::TypedModel {
    fn add_const(&mut self, name: impl Into<String>, v: SharedTensor) -> TractResult<usize> {
        let facts = tvec!(v.clone().into());
        self.add_node(name, crate::ops::konst::Const::new(v), facts)
    }
    fn plug_const(&mut self, inlet: InletId, name: impl Into<String>, v: SharedTensor) -> TractResult<()> {
        let cst = self.add_const(name, v)?;
        self.add_edge(OutletId::new(cst, 0), inlet)?;
        Ok(())
    }
}

pub trait ModelDslInfer {
    fn add_source_default(&mut self, name: impl Into<String>) -> TractResult<usize>;
    fn add_node_default(&mut self, name: impl Into<String>, op: impl Into<Box<Op>>,) -> TractResult<usize>;
    fn chain_default(&mut self, name: impl Into<String>, op: impl Into<Box<Op>>) -> TractResult<usize>;
}

impl ModelDslInfer for super::InferenceModel {
    fn add_source_default(&mut self, name: impl Into<String>) -> TractResult<usize> {
        self.add_source(name, TensorFact::default())
    }
    fn add_node_default(&mut self, name: impl Into<String>, op: impl Into<Box<Op>>,) -> TractResult<usize> {
        self.add_node(name, op, tvec!(TensorFact::default()))
    }
    fn chain_default(&mut self, name: impl Into<String>, op: impl Into<Box<Op>>) -> TractResult<usize> {
        self.chain(name, op, tvec!(TensorFact::default()))
    }
}
