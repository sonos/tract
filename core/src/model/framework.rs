use std::fmt::Debug;
use crate::ops::prelude::*;

#[derive(Default)]
pub struct Framework<ProtoOp: Debug> {
    ops: HashMap<String, fn(&ProtoOp) -> TractResult<Box<Op>>>
}

impl<ProtoOp: Debug> Framework<ProtoOp> {
    pub fn build_op(&self, name: &str, payload: &ProtoOp) -> TractResult<Box<Op>> {
        match self.ops.get(name) {
            Some(builder) => builder(payload),
            None => Ok(Box::new(crate::ops::unimpl::UnimplementedOp::new(
                name,
                format!("{:?}", payload),
            ))),
        }
    }

    pub fn insert(&mut self, name: &str, builder: fn(&ProtoOp) -> TractResult<Box<Op>>) {
        self.ops.insert(name.to_string(), builder);
    }
}

