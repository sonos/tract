use ops::prelude::*;

pub use super::{InletId, Model, Node, OutletId};

pub trait ModelDsl {
    fn single_prec(&self, id: usize) -> TractResult<Option<&Node>>;
    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&Node>>;
    fn single_succ(&self, id: usize) -> TractResult<Option<&Node>>;
    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&Node>>;

    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TractResult<usize>;
    fn add_source_fact<S: AsRef<str>>(&mut self, name: S, fact: TensorFact) -> TractResult<usize>;
    fn add_const<S: AsRef<str>>(&mut self, name: S, v: SharedTensor) -> TractResult<usize>;
    fn chain<S: AsRef<str>>(&mut self, name: S, op: Box<Op>) -> TractResult<usize>;

    fn tap_and_chain<S: AsRef<str>>(
        &mut self,
        tap: OutletId,
        name: S,
        op: Box<Op>,
    ) -> TractResult<usize>;

    fn replace_nodes(
        &mut self,
        node: usize,
        before: usize,
        after: usize,
        nodes: Vec<(String, Box<Op>)>,
    ) -> TractResult<()>;

    fn unlink_node(&mut self, node: usize) -> TractResult<()>;
}

impl ModelDsl for ::model::Model {
    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TractResult<usize> {
        self.add_source_fact(name, TensorFact::default())
    }

    fn add_source_fact<S: AsRef<str>>(&mut self, name: S, fact: TensorFact) -> TractResult<usize> {
        let id = self.add_node(
            name.as_ref().to_owned(),
            Box::new(::ops::source::Source::new(fact.clone())),
        )?;
        self.set_fact(OutletId::new(id, 0), fact)?;
        Ok(id)
    }

    fn add_const<S: AsRef<str>>(&mut self, name: S, v: SharedTensor) -> TractResult<usize> {
        self.add_node(
            name.as_ref().to_owned(),
            Box::new(::ops::konst::Const::new(v)),
        )
    }

    fn chain<S: AsRef<str>>(&mut self, name: S, op: Box<Op>) -> TractResult<usize> {
        let previous_id = self.nodes.len() - 1;
        self.tap_and_chain(OutletId::new(previous_id, 0), name, op)
    }

    fn single_prec(&self, id: usize) -> TractResult<Option<&Node>> {
        let node = &self.nodes[id];
        if node.inputs.len() != 1 {
            return Ok(None);
        }
        let prec = &self.nodes[node.inputs[0].node];
        if prec
            .outputs
            .iter()
            .map(|of| of.successors.len())
            .sum::<usize>()
            != 1
        {
            return Ok(None);
        }
        Ok(Some(prec))
    }

    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&Node>> {
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

    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&Node>> {
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

    fn single_succ(&self, id: usize) -> TractResult<Option<&Node>> {
        let node = &self.nodes[id];
        if node
            .outputs
            .iter()
            .map(|of| of.successors.len())
            .sum::<usize>()
            != 1
        {
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
    ) -> TractResult<usize> {
        let id = self.add_node(name.as_ref().to_owned(), op.into())?;
        self.add_edge(tap, InletId::new(id, 0))?;
        Ok(id)
    }

    fn replace_nodes(
        &mut self,
        node: usize,
        before: usize,
        after: usize,
        nodes: Vec<(String, Box<Op>)>,
    ) -> TractResult<()> {
        let first_replaced = self
            .single_prec_at(node, before)?
            .ok_or("Failed to replace, geometry is not right")?
            .id;
        let mut tap = self.node(first_replaced).inputs[0];
        for (name, op) in nodes.into_iter() {
            let id = self.tap_and_chain(tap, name, op)?;
            tap = OutletId::new(id, 0);
        }
        self.unlink_node(first_replaced)?;
        let successors: Vec<InletId> = self
            .single_succ_at(node, after)?
            .ok_or("Failed to replace, geometry is not right")?
            .outputs[0]
            .successors
            .clone();
        for &succ in &successors {
            self.add_edge(tap, succ)?;
        }
        Ok(())
    }

    fn unlink_node(&mut self, node: usize) -> TractResult<()> {
        let inputs = self.nodes[node].inputs.clone();
        for (ix, &input) in inputs.iter().enumerate() {
            self.nodes[input.node].outputs[input.slot]
                .successors
                .retain(|&wire| wire != InletId::new(node, ix));
        }
        Ok(())
    }
}
