use TfdResult;
use ops::Op;
use super::*;
use tensor::Tensor;

pub trait ModelDsl {
    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TfdResult<usize>;
    fn add_const<S: AsRef<str>>(&mut self, name: S, t:Tensor) -> TfdResult<usize>;
    fn chain<S: AsRef<str>, O: Op>(&mut self, name: S, op: O) -> TfdResult<usize>;
}

impl ModelDsl for ::model::Model {
    fn add_source<S: AsRef<str>>(&mut self, name: S) -> TfdResult<usize> {
        self.add_node(name.as_ref().to_owned(), Box::new(::ops::source::Source::default()))
    }

    fn add_const<S: AsRef<str>>(&mut self, name: S, t:Tensor) -> TfdResult<usize> {
        self.add_node(name.as_ref().to_owned(), Box::new(::ops::konst::Const::new(t.into())))
    }

    fn chain<S: AsRef<str>, O: Op>(&mut self, name: S, op: O) -> TfdResult<usize> {
        let previous_id = self.nodes.len() - 1;
        let id = self.add_node(name.as_ref().to_owned(), Box::new(op))?;
        self.add_edge(OutletId::new(previous_id, 0), InletId::new(id, 0))?;
        Ok(id)
    }
}
