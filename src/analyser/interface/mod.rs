//! A fluent interface for the analyser.
//!
//! This interface provides proxies for the different properties of tensors.
//! This allows inference rules to be stated in a clear, declarative fashion
//! inside the `rules` method of each operator.
//!
//! Take these rules for instance:
//! ```text
//! solver.equals(inputs.len, 2);
//! solver.equals(inputs[0].datum_type, outputs[0].datum_type);
//! ```
//! Here, `inputs.len`, `inputs[0].datum_type` and `outputs[0].datum_type` don't
//! actually hold the values of the length and datum_types, but instead act as
//! declarative placeholders for these values.

#[macro_export]
macro_rules! wrap {
    ($($x:expr),*) => ({
        vec![$( $crate::analyser::interface::exp2::IntoExp::bex($x) ),*]
    });

    ($($x:expr,)*) => (wrap![$($x),*]);
}

mod cache;
mod expressions;
pub mod exp2;
mod path;
mod proxies;
mod solver;

pub use self::expressions::{Expression, IntoExpression};
pub use self::proxies::*;
pub use self::solver::*;
pub use super::prelude::*;
pub use linear::{ LinearDim, ToDim };
pub use analyser::interface::exp2::{ IntoExp, ToDimExp };


pub fn bexp<T, IE, E>(fact: IE) -> Box<Expression<Output = T>>
where
    E: Expression<Output = T> + 'static,
    IE: IntoExpression<E>,
    T: expressions::Output + 'static,
{
    Box::new(fact.into_expr())
}
