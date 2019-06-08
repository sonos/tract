use internal::*;
use std::fmt::Debug;

pub trait OpState: Debug {
    type Op: Op;
    fn eval(&mut self, op: &Self::Op, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>>;
}

impl OpState for Option<Box<OpState>> {
    fn eval(&mut self, op: &Op, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        match self {
            None => op.as_stateless().unwrap().eval(inputs),
            Some(state) => {
                let op = op.downcast_ref::<Self::Op>(op).unwrap();
                op.as_statefull().unwrap().eval(self, inputs)
            }
        }
    }
}

pub trait StatelessOp: Op {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>>;
}

pub trait StatefullOp: Op {
    type State: OpState;
    fn dispatch_eval(&self, state: &mut OpState, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
    }
}

pub trait OpStateDispatch {
    fn state(&self) -> TractResult<Option<Box<OpState>>>;
    fn as_stateless(&self) -> Option<&StatelessOp>;
}

impl<O: Op + StatelessOp + Clone> OpStateDispatch for O {
    fn state(&self) -> TractResult<Option<Box<OpState>>> {
        Ok(None)
    }

    fn as_stateless(&self) -> Option<&StatelessOp> {
        Some(self)
    }
}

