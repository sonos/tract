use ndarray::prelude::*;
use tract_core::ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Slice {
    axes: Option<Vec<usize>>,
    starts: Vec<isize>,
    ends: Vec<isize>,
}

impl Slice {
    fn eval_t<T: Datum>(&self, input: Value) -> TractResult<Value> {
        let mut input = input.to_array_view::<T>()?;
        for (ix, (&b, &e)) in self.starts.iter().zip(self.ends.iter()).enumerate() {
            let axis = self.axes.as_ref().map(|axes| axes[ix]).unwrap_or(ix);
            let b = if b > input.shape()[axis] as isize {
                input.shape()[axis] as isize
            } else {
                b
            };
            let e = if e > input.shape()[axis] as isize {
                input.shape()[axis] as isize
            } else {
                e
            };
            input.slice_axis_inplace(
                Axis(axis),
                ::ndarray::Slice::from((b as isize)..(e as isize)),
            );
        }
        Ok(Tensor::from(input.to_owned()).into())
    }
}

impl Op for Slice {
    fn name(&self) -> &str {
        "onnx.Slice"
    }
}

impl StatelessOp for Slice {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(
            self, input
        ))?))
    }
}

impl InferenceRulesOp for Slice {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        if self.axes.is_none() {
            s.equals(&inputs[0].rank, self.starts.len() as i32)?;
            s.equals(&inputs[0].rank, self.ends.len() as i32)?;
        }
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, shape| {
            (0..shape.len()).try_for_each(move |axis| {
                let d = shape[axis];
                let spec = if let Some(axes) = self.axes.as_ref() {
                    if let Some(ix) = axes.iter().position(|&a| a == axis) {
                        Some((self.starts[ix], self.ends[ix]))
                    } else {
                        None
                    }
                } else {
                    Some((self.starts[axis].into(), self.ends[axis].into()))
                };
                if let Some((mut b, mut e)) = spec {
                    if let Ok(d) = d.to_integer() {
                        if b as i32 > d {
                            b = (d as isize).into();
                        }
                        if e as i32 > d {
                            e = (d as isize).into();
                        }
                    }
                    let mut b = if b < 0 {
                        d.bex() + TDim::from(b)
                    } else {
                        TDim::from(b).bex()
                    };
                    let mut e = if e < 0 {
                        d.bex() + TDim::from(e)
                    } else {
                        TDim::from(e).bex()
                    };
                    s.equals(&outputs[0].shape[axis], e - b)
                } else {
                    s.equals(&outputs[0].shape[axis], shape[axis])
                }
            })
        })?;
        Ok(())
    }
}
