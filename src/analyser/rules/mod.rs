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
        vec![$( $crate::analyser::rules::exp2::IntoExp::bex($x) ),*]
    });

    ($($x:expr,)*) => (wrap![$($x),*]);
}

mod cache;
pub mod exp2;
mod expressions;
mod path;
mod proxies;
mod solver;

pub mod prelude {
    pub use self::super::InferenceRulesOp;
    pub use self::super::cache::*;
    pub use self::super::expressions::*;
    pub use self::super::exp2::*;
    pub use self::super::path::*;
    pub use self::super::proxies::*;
    pub use self::super::solver::*;
    pub use analyser::prelude::*;
    pub use model::TVec;
    pub use dim::{TDim, ToDim};
}

use self::prelude::*;

pub trait InferenceRulesOp {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    );
}

impl<O: InferenceRulesOp> ::ops::InferenceOp for O {
    fn infer(
        &self,
        inputs: TVec<TensorFact>,
        outputs: TVec<TensorFact>,
    ) -> ::Result<(TVec<TensorFact>, TVec<TensorFact>)> {
        let inputs_proxy = TensorsProxy::new(vec![0].into());
        let outputs_proxy = TensorsProxy::new(vec![1].into());

        let mut solver = Solver::default();
        self.rules(&mut solver, &inputs_proxy, &outputs_proxy);
        solver.infer((inputs, outputs))
    }
}
