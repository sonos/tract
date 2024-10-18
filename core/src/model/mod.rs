//! ## Models and their lifecycle
//!
//! In order to reason on the model and performs optimisations, a model needs
//! to be `typed`. This means all tensor exchanged between the nodes have a
//! well defined element type (f32, i32, etc) and a shape ([1, 12, 53, 13]).
//!
//! A model typically starts as an `InferenceModel`, with minimum or partial
//! tensor type information. At this stage, the application developper can add
//! types and shapes hints (like the model inputs and output element types
//! and shapes), then `tract` will perform type inference propagating this
//! information. Hopefully `tract` will be able to infer a type and shape for
//! all tensors in the model graph.
//!
//! At this stage, the model can be converted into a `TypedModel`.
//!
//! InferanceModel and TypeModel are two variants of `Graph`, Parameterized
//! by a Fact implementation: TypedModel uses TypedFact, enforcing
//! complete determination of element type and shape, and allowing a constant
//! value for the tensor. InferenceModel uses InferenceFact, which can handle
//! partial information.
//!
//! We call `declutter` the process getting the network closer to a normal
//! form:.  This normal form is akin to an IR in compiler technologies. This is
//! the favourite form on which tract optimisation is implemented.
//!
//! For instance an Add node adding a constant input to a variable
//! tensor input would be replaced by an unary Add operator taking only the
//! variable input and for which the constant to add is a fixed construction
//! attribute. In the same decluttering process, we try and replace proprietary
//! operators (eg, from TensorFlow) by tract core operators: it is not always
//! possible to simply map TensorFlow operators to tract-core while loading the
//! network: their interfaces can be different (what is an input, what is an
//! attribute) and constant propagation may be necessary before the right
//! core operator could be chosen.
//!
use std::collections::HashMap;
use std::str;

mod fact;
mod graph;
pub mod memory;
mod node;
pub mod order;
mod patch;
mod rewriter;
pub mod translator;
pub mod typed;

pub use self::fact::*;
pub use self::graph::*;
pub use self::node::*;
pub use self::patch::ModelPatch;
pub use self::rewriter::Rewriter;
pub use crate::ops::{Op, TypedOp};

pub use typed::*;
