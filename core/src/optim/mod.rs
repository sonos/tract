use {Model, TractResult};

mod compact;
mod prop_const;
mod reduce;

pub use self::compact::compact;
pub use self::prop_const::prop_const;
pub use self::reduce::Reduce;

pub trait OptimizerPass {
    fn pass(model: &mut Model) -> TractResult<bool>;
}
