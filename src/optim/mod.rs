use { Model, TfdResult };

mod compact;
mod prop_const;
mod reduce;

pub use self::compact::compact;
pub use self::prop_const::prop_const;
pub use self::reduce::Reduce;

trait OptimizerPass {
    fn pass(model: &mut Model) -> TfdResult<bool>;
}

