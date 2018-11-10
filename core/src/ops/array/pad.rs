use ndarray::*;
use num::traits::AsPrimitive;
use ops::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum PadMode {
    Constant(f32),
    Reflect,
    Edge,
}
impl Default for PadMode {
    fn default() -> PadMode {
        PadMode::Constant(0.0)
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct Pad {
    pads: Vec<(usize, usize)>,
    mode: PadMode,
}

impl Pad {
    fn eval_t<T>(&self, input: Tensor) -> TractResult<Tensor>
    where
        T: Datum,
        f32: AsPrimitive<T>,
    {
        let input = input.to_array_view::<T>()?;
        let output_shape: Vec<usize> = input
            .shape()
            .iter()
            .zip(self.pads.iter())
            .map(|(&d, &(a, b))| d + a + b)
            .collect();
        let element = match self.mode {
            PadMode::Constant(f) => f.as_(),
            _ => T::default(),
        };
        let mut output = ArrayD::<T>::from_elem(output_shape, element);
        let slice_spec: Vec<SliceOrIndex> = self
            .pads
            .iter()
            .map(|&(a, b)| SliceOrIndex::Slice {
                start: a as isize,
                end: if b != 0 { Some(-(b as isize)) } else { None },
                step: 1,
            }).collect();
        let slice_info = SliceInfo::<_, IxDyn>::new(slice_spec).unwrap();
        output.slice_mut(slice_info.as_ref()).assign(&input);
        if self.mode == PadMode::Reflect || self.mode == PadMode::Edge {
            for (ax, &(bef, aft)) in self.pads.iter().enumerate() {
                let axis = Axis(ax);
                let dim = output.shape()[ax];
                {
                    let (mut pad, data) = output.view_mut().split_at(axis, bef);
                    for i in 0..bef {
                        let mut target = pad.slice_axis_mut(axis, Slice::from(i..i + 1));
                        let source_slice = match self.mode {
                            PadMode::Edge => 0,
                            PadMode::Reflect => bef - i,
                            _ => panic!(),
                        };
                        let source =
                            data.slice_axis(axis, Slice::from(source_slice..source_slice + 1));
                        target.assign(&source);
                    }
                }
                {
                    let (data, mut pad) = output.view_mut().split_at(axis, dim - aft);
                    for i in 0..aft {
                        let mut target = pad.slice_axis_mut(axis, Slice::from(i..i + 1));
                        let source_slice = match self.mode {
                            PadMode::Edge => dim - aft - 1,
                            PadMode::Reflect => dim - aft - 2 - i,
                            _ => panic!(),
                        };
                        let source =
                            data.slice_axis(axis, Slice::from(source_slice..source_slice + 1));
                        target.assign(&source);
                    }
                }
            }
        }
        Ok(output.into())
    }
}

impl Op for Pad {
    fn name(&self) -> &str {
        "Pad"
    }
}

impl StatelessOp for Pad {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Tensor>) -> TractResult<TVec<Tensor>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::eval_t(input.datum_type())(
            self, input
        ))?))
    }
}

impl InferenceRulesOp for Pad {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        for (ix, &(a, b)) in self.pads.iter().enumerate() {
            s.equals(
                &inputs[0].shape[ix],
                outputs[0].shape[ix].bex() - a.to_dim() - b.to_dim(),
            )?;
        }
        Ok(())
    }
}
