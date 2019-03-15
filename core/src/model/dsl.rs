use crate::ops::prelude::*;

pub use super::{InletId, Model, Node, OutletId};

pub trait ModelDsl<TI: TensorInfo> {
    fn single_prec(&self, id: usize) -> TractResult<Option<&Node<TI>>>;
    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<TI>>>;
    fn single_succ(&self, id: usize) -> TractResult<Option<&Node<TI>>>;
    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&Node<TI>>>;

    fn add_source_fact<S: AsRef<str>>(&mut self, name: S, fact: TI) -> TractResult<usize>;
    fn chain_facts<S: AsRef<str>>(&mut self, name: S, op: Box<Op>, facts:TVec<TI>) -> TractResult<usize>;

    fn tap_and_chain<S: AsRef<str>>(
        &mut self,
        tap: OutletId,
        name: S,
        op: Box<Op>,
        facts: TVec<TI>,
    ) -> TractResult<usize>;

}

impl<TI: TensorInfo> ModelDsl<TI> for Model<TI> {
    fn add_source_fact<S: AsRef<str>>(&mut self, name: S, fact: TI) -> TractResult<usize> {
        let id = self.add_node(
            name.as_ref().to_owned(),
            Box::new(crate::ops::source::Source::new(fact.to_tensor_fact())),
            tvec!(fact),
        )?;
        Ok(id)
    }

    fn chain_facts<S: AsRef<str>>(&mut self, name: S, op: Box<Op>, facts: TVec<TI>) -> TractResult<usize> {
        let previous_id = self.nodes.len() - 1;
        self.tap_and_chain(OutletId::new(previous_id, 0), name, op, facts)
    }

    fn single_prec(&self, id: usize) -> TractResult<Option<&Node<TI>>> {
        let node = &self.nodes[id];
        if node.inputs.len() != 1 {
            return Ok(None);
        }
        let prec = &self.nodes[node.inputs[0].node];
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
        let node = &self.nodes[id];
        if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
            return Ok(None);
        }
        let succ = node.outputs[0].successors[0];
        let succ = &self.nodes[succ.node];
        if succ.inputs.len() != 1 {
            return Ok(None);
        }
        Ok(Some(succ))
    }

    fn tap_and_chain<S: AsRef<str>>(
        &mut self,
        tap: OutletId,
        name: S,
        op: Box<Op>,
        facts: TVec<TI>,
    ) -> TractResult<usize> {
        let id = self.add_node(name.as_ref().to_owned(), op.into(), facts)?;
        self.add_edge(tap, InletId::new(id, 0))?;
        Ok(id)
    }

}

pub trait ModelDslConst {
    fn add_const<S: AsRef<str>>(&mut self, name: S, v: SharedTensor) -> TractResult<usize>;
}

impl ModelDslConst for super::InferenceModel {
    fn add_const<S: AsRef<str>>(&mut self, name: S, v: SharedTensor) -> TractResult<usize> {
        let facts = tvec!(v.clone().into());
        self.add_node(name.as_ref().to_owned(), Box::new(crate::ops::konst::Const::new(v)), facts)
    }
}

impl ModelDslConst for super::TypedModel {
    fn add_const<S: AsRef<str>>(&mut self, name: S, v: SharedTensor) -> TractResult<usize> {
        let facts = tvec!(v.clone().into());
        self.add_node(name.as_ref().to_owned(), Box::new(crate::ops::konst::Const::new(v)), facts)
    }
}

pub trait ModelDslInfer {
    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TractResult<usize>;
    fn chain<S: AsRef<str>>(&mut self, name: S, op: Box<Op>) -> TractResult<usize>;
}

impl ModelDslInfer for super::InferenceModel {
    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TractResult<usize> {
        self.add_source_fact(name, TensorFact::default())
    }
    fn chain<S: AsRef<str>>(&mut self, name: S, op: Box<Op>) -> TractResult<usize> {
        self.chain_facts(name, op, tvec!(TensorFact::default()))
    }
}
