//! A fluent interface for the analyser.
//!
//! This interface provides proxies for the different properties of tensors.
//! This allows inference rules to be stated in a clear, declarative fashion
//! inside the `rules` method of each operator.
//!
//! Take these rules for instance:
//! ```text
//! solver.equals(inputs.len, 2);
//! solver.equals(inputs[0].datatype, outputs[0].datatype);
//! ```
//! Here, `inputs.len`, `inputs[0].datatype` and `outputs[0].datatype` don't
//! actually hold the values of the length and datatypes, but instead act as
//! declarative placeholders for these values.

mod path;
mod cache;
mod proxies;
mod expressions;
mod solver;
pub use self::proxies::*;
pub use self::solver::*;