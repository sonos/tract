use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Concat {
    axis: usize,
}

impl Concat {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let mats: TfdResult<Vec<ArrayViewD<T>>> =
            inputs.iter().map(|mat| mat.to_array_view()).collect();
        let result = ::ndarray::stack(Axis(self.axis as usize), &*mats?)?;
        Ok(tvec![result.into()])
    }
}

impl Op for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        dispatch_datum!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for Concat {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&outputs.len, 1)
            .equals(&outputs[0].datum_type, &inputs[0].datum_type)
            .equals(&outputs[0].rank, &inputs[0].rank)
            .given(&inputs.len, move |solver, n| {
                let n = n as usize;
                solver
                    .equals_all((0..n).map(|i| (&inputs[i].datum_type).bex()).collect())
                    .equals_all((0..n).map(|i| (&inputs[i].rank).bex()).collect())
                    .equals(
                        SumExp::new((0..n).map(|i| (&inputs[i].shape[self.axis]).bex()).collect()),
                        &outputs[0].shape[self.axis]);
                for axis in 0..self.axis {
                    solver
                        .equals(&outputs[0].shape[axis], &inputs[0].shape[axis])
                        .equals_all((0..n).map(|i| inputs[i].shape[axis].bex()).collect());
                }
                solver.given(&inputs[0].rank, move |solver, axes| {
                    let axes = axes as usize;
                    for axis in n..axes {
                        solver
                            .equals(&outputs[0].shape[axis], &inputs[0].shape[axis])
                            .equals_all((0..n).map(|i| inputs[i].shape[axis].bex()).collect());
                    }
                });
            });
    }
}
