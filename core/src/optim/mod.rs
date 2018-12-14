use {Model, TractResult};
use std::fmt::Debug;

mod compact;
mod prop_const;
mod push_split_down;
mod reduce;

pub use self::compact::compact;
pub use self::prop_const::PropConst;
pub use self::push_split_down::PushSplitDown;
pub use self::reduce::Reduce;
pub use ops::ReductionPhase;

pub fn normalization() -> Vec<Box<OptimizerPass>> {
    vec![Box::new(PropConst) as Box<OptimizerPass>, Box::new(Reduce(ReductionPhase::Normalize))]
}

pub fn codegen() -> Vec<Box<OptimizerPass>> {
    vec![Box::new(Reduce(ReductionPhase::Codegen)), Box::new(PushSplitDown)]
}

pub trait OptimizerPass: Debug {
    fn pass(&self, model: &mut Model) -> TractResult<bool>;
}
