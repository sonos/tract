use super::*;
use ops::Op;
use tensor::Tensor;
use TfdResult;

pub trait ModelDsl {
    fn single_prec(&self, id: usize) -> TfdResult<Option<&Node>>;
    fn single_prec_at(&self, id: usize, count: usize) -> TfdResult<Option<&Node>>;
    fn single_succ(&self, id: usize) -> TfdResult<Option<&Node>>;
    fn single_succ_at(&self, id: usize, count: usize) -> TfdResult<Option<&Node>>;

    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TfdResult<usize>;
    fn add_const<S: AsRef<str>>(&mut self, name: S, t: Tensor) -> TfdResult<usize>;
    fn chain<S: AsRef<str>>(&mut self, name: S, op: Box<Op>) -> TfdResult<usize>;

    fn tap_and_chain<S: AsRef<str>>(
        &mut self,
        tap: OutletId,
        name: S,
        op: Box<Op>,
    ) -> TfdResult<usize>;

    fn replace_nodes(
        &mut self,
        node: usize,
        before: usize,
        after: usize,
        nodes: Vec<(String, Box<Op>)>,
    ) -> TfdResult<()>;
}

impl ModelDsl for ::model::Model {
    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TfdResult<usize> {
        self.add_node(
            name.as_ref().to_owned(),
            Box::new(::ops::source::Source::default()),
        )
    }

    fn add_const<S: AsRef<str>>(&mut self, name: S, t: Tensor) -> TfdResult<usize> {
        self.add_node(
            name.as_ref().to_owned(),
            Box::new(::ops::konst::Const::new(t.into())),
        )
    }

    fn chain<S: AsRef<str>>(&mut self, name: S, op: Box<Op>) -> TfdResult<usize> {
        let previous_id = self.nodes.len() - 1;
        self.tap_and_chain(OutletId::new(previous_id, 0), name, op)
    }

    fn single_prec(&self, id: usize) -> TfdResult<Option<&Node>> {
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

    fn single_prec_at(&self, id: usize, count: usize) -> TfdResult<Option<&Node>> {
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

    fn single_succ_at(&self, id: usize, count: usize) -> TfdResult<Option<&Node>> {
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

    fn single_succ(&self, id: usize) -> TfdResult<Option<&Node>> {
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
    ) -> TfdResult<usize> {
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
    ) -> TfdResult<()> {
        let first_replaced = self.single_prec_at(node, before)?
            .ok_or("Failed to replace, geometry is not right")?.id;
        let mut tap = self.node(first_replaced).inputs[0];
        for (name, op) in nodes.into_iter() {
            let id = self.tap_and_chain(tap, name, op)?;
            tap = OutletId::new(id, 0);
        }
        let successors:Vec<InletId> = self.single_succ_at(node, after)?
            .ok_or("Failed to replace, geometry is not right")?
            .outputs[0].successors.clone();
        for &succ in &successors {
            self.add_edge(tap, succ)?;
        }
        Ok(())
    }
}
