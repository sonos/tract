use {Model, TractResult};

mod compact;
mod prop_const;
mod reduce;

pub use self::compact::compact;
pub use self::prop_const::PropConst;
pub use self::reduce::Reduce;

pub trait OptimizerPass {
    fn pass(&self, model: &mut Model) -> TractResult<bool>;
}
