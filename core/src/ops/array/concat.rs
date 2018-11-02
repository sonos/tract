use ndarray::*;

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
}

impl StatelessOp for Concat {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        dispatch_datum!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for Concat {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs.len, move |s, n| {
            let n = n as usize;
            s.equals_all((0..n).map(|i| (&inputs[i].datum_type).bex()).collect())?;
            s.equals_all((0..n).map(|i| (&inputs[i].rank).bex()).collect())?;
            s.equals(
                ::analyser::rules::expr::SumExp::new(
                    (0..n)
                        .map(|i| (&inputs[i].shape[self.axis]).bex())
                        .collect(),
                ),
                &outputs[0].shape[self.axis],
            )?;
            for axis in 0..self.axis {
                s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])?;
                s.equals_all((0..n).map(|i| inputs[i].shape[axis].bex()).collect())?;
            }
            s.given(&inputs[0].rank, move |s, axes| {
                let axes = axes as usize;
                for axis in (self.axis + 1)..axes {
                    s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])?;
                    s.equals_all((0..n).map(|i| inputs[i].shape[axis].bex()).collect())?;
                }
                Ok(())
            })
        })
    }
}
