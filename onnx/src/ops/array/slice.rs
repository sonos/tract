use crate::model::ParsingContext;
use crate::pb::*;
use ndarray::prelude::*;
use tract_core::internal::*;

pub fn slice(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let v = ctx.onnx_operator_set_version;
    if v >= 1 && v < 10 {
        slice1(ctx, node)
    } else if v == 10 {
        slice10(ctx, node)
    } else {
        bail!("Only Slice-1 and Slice-10 are supported")
    }
}

fn slice1(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    let begin = node.get_attr_vec("starts")?;
    let end = node.get_attr_vec("ends")?;
    Ok((Box::new(Slice1::new(axes, begin, end)), vec![]))
}

#[derive(Debug, Clone, new, Default)]
pub struct Slice1 {
    axes: Option<Vec<usize>>,
    starts: Vec<isize>,
    ends: Vec<isize>,
}

impl Slice1 {
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        let mut input = input.to_array_view::<T>()?;
        for (ix, (&b, &e)) in self.starts.iter().zip(self.ends.iter()).enumerate() {
            let axis = self.axes.as_ref().map(|axes| axes[ix]).unwrap_or(ix);
            let b = if b > input.shape()[axis] as isize { input.shape()[axis] as isize } else { b };
            let e = if e > input.shape()[axis] as isize { input.shape()[axis] as isize } else { e };
            input
                .slice_axis_inplace(Axis(axis), ::ndarray::Slice::from((b as isize)..(e as isize)));
        }
        Ok(Tensor::from(input.to_owned()).into())
    }
}

impl Op for Slice1 {
    fn name(&self) -> Cow<str> {
        "onnx.Slice1".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Slice1 {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl InferenceRulesOp for Slice1 {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        if self.axes.is_none() {
            s.equals(&inputs[0].rank, self.starts.len() as i32)?;
            s.equals(&inputs[0].rank, self.ends.len() as i32)?;
        }
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, shape| {
            (0..shape.len()).try_for_each(move |axis| {
                let d = &shape[axis];
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
                    let b = if b < 0 { d.bex() + TDim::from(b) } else { TDim::from(b).bex() };
                    let e = if e < 0 { d.bex() + TDim::from(e) } else { TDim::from(e).bex() };
                    s.equals(&outputs[0].shape[axis], e - b)
                } else {
                    s.equals(&outputs[0].shape[axis], &shape[axis])
                }
            })
        })?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = target.outlet_fact(mapping[&node.inputs[0]])?.clone();
        let mut wire = mapping[&node.inputs[0]];
        for (ix, (&b, &e)) in self.starts.iter().zip(self.ends.iter()).enumerate() {
            let axis = self.axes.as_ref().map(|axes| axes[ix]).unwrap_or(ix);
            let dim = input.shape.dim(axis);
            if let Ok(dim) = dim.to_integer() {
                let b = (if b >= 0 { b.min(dim as isize) } else { dim as isize + b }) as usize;
                let e = (if e >= 0 { e.min(dim as isize) } else { dim as isize + e }) as usize;
                if b > 0 || e < dim as usize {
                    wire = target.wire_node(
                        format!("{}-axis-{}", node.name, axis),
                        tract_core::ops::array::Slice::new(axis, b, e),
                        [wire].as_ref(),
                    )?[0];
                }
            } else {
                bail!("Can't translate slice: axis={} dim={} b={} e={}", axis, dim, b, e)
            }
        }
        target.rename_node(wire.node, &*node.name)?;
        Ok(tvec!(wire))
    }

    inference_op_as_op!();
}

fn slice10(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut optional_inputs = crate::model::optional_inputs(node).skip(3);
    Ok((
        Box::new(tract_core::ops::array::StridedSlice::onnx10(
            optional_inputs.next().unwrap(),
            optional_inputs.next().unwrap(),
        )),
        vec![],
    ))
}
/*
#[derive(Debug, Clone, new, Default)]
pub struct Slice10 {
    pub optional_axes_input: Option<usize>,
    pub optional_steps_input: Option<usize>,
}

impl Slice10 {
    fn eval_t<T: Datum>(
        input: &Tensor,
        axis: usize,
        begin: isize,
        end: isize,
        step: isize,
    ) -> TractResult<Tensor> {
        dbg!(begin, end, step);
        let dim = input.shape()[axis] as isize;
        dbg!(dim);
        let mut begin = begin.min(dim).max(-dim);
        let mut end = end.min(dim).max(-dim);
        dbg!(begin, end);
        if step < 0 {
            std::mem::swap(&mut begin, &mut end);
        }
        dbg!(begin, end);
        Ok(input
            .to_array_view::<T>()?
            .slice_axis(Axis(axis), ndarray::Slice::new(begin, Some(end), step))
            .to_owned()
            .into_tensor())
    }
}

impl Op for Slice10 {
    fn name(&self) -> Cow<str> {
        "onnx.Slice10".into()
    }

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        Ok(None)
    }

    not_a_typed_op!();
}

impl StatelessOp for Slice10 {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
    }
}

impl InferenceRulesOp for Slice10 {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(
            &inputs,
            3 + self.optional_axes_input.is_some() as usize
                + self.optional_steps_input.is_some() as usize,
        )?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[1].shape, &inputs[2].shape)?;
        if let Some(input) = self.optional_axes_input {
            s.equals(&inputs[input].rank, 1)?;
            s.equals(&inputs[input].shape, &inputs[1].shape)?;
        };
        if let Some(input) = self.optional_steps_input {
            s.equals(&inputs[input].rank, 1)?;
            s.equals(&inputs[input].shape, &inputs[1].shape)?;
        };

        fn output_shape_rules<'r, 'p: 'r, 's: 'r>(
            s: &mut Solver<'r>,
            inputs: &'p [TensorProxy],
            outputs: &'p [TensorProxy],
            axes: TVec<usize>,
            steps: TVec<isize>,
        ) -> TractResult<()> {
            s.given_3(
                &inputs[0].shape,
                &inputs[1].value,
                &inputs[2].value,
                move |s, shape, beg, end| {
                    let beg = beg.cast_to::<TDim>()?;
                    let beg = beg.as_slice::<TDim>()?;
                    let end = end.cast_to::<TDim>()?;
                    let end = end.as_slice::<TDim>()?;
                    for (ix, &axis) in axes.iter().enumerate() {
                        let dim = &shape[axis];
                        let b = if let Ok(v) = beg[ix].to_integer() {
                            if v < 0 {
                                dim.clone() + v - 1
                            } else {
                                v.to_dim()
                            }
                        } else {
                            beg[ix].clone()
                        };
                        let e = if let Ok(v) = end[ix].to_integer() {
                            if v < 0 {
                                dim.clone() + v - 1
                            } else {
                                v.to_dim()
                            }
                        } else {
                            end[ix].clone()
                        };
                        s.equals(&outputs[0].shape[axis], (e - b).div_ceil(steps[ix].to_dim()))?;
                    }
                    for axis in 0..shape.len() {
                        if !axes.contains(&axis) {
                            s.equals(&outputs[0].shape[axis], &shape[axis])?;
                        }
                    }
                    Ok(())
                },
            )
        }

        s.given(&inputs[0].rank, move |s, rank| {
            let rank = rank as usize;
            match (self.optional_axes_input, self.optional_steps_input) {
                (Some(axes), Some(steps)) => {
                    s.given_2(&inputs[axes].value, &inputs[steps].value, move |s, axes, steps| {
                        let axes = axes
                            .cast_to::<i32>()?
                            .as_slice::<i32>()?
                            .iter()
                            .map(|&i| if i >= 0 { i } else { rank as i32 - 1 - i } as usize)
                            .collect::<TVec<usize>>();
                        let steps = steps
                            .cast_to::<i32>()?
                            .as_slice::<i32>()?
                            .iter()
                            .map(|&i| i as isize)
                            .collect::<TVec<isize>>();
                        output_shape_rules(s, inputs, outputs, axes, steps)
                    })
                }
                (Some(axes), None) => s.given(&inputs[axes].value, move |s, axes| {
                    let axes = axes
                        .cast_to::<i32>()?
                        .as_slice::<i32>()?
                        .iter()
                        .map(|&i| if i >= 0 { i } else { rank as i32 - 1 - i } as usize)
                        .collect::<TVec<usize>>();
                    let steps = tvec!(1; rank);
                    output_shape_rules(s, inputs, outputs, axes, steps)
                }),
                (None, Some(steps)) => s.given(&inputs[steps].value, move |s, steps| {
                    let axes = (0..rank).collect::<TVec<usize>>();
                    let steps = steps
                        .cast_to::<i32>()?
                        .as_slice::<i32>()?
                        .iter()
                        .map(|&i| i as isize)
                        .collect::<TVec<isize>>();
                    output_shape_rules(s, inputs, outputs, axes, steps)
                }),
                (None, None) => {
                    let axes = (0..rank).collect::<TVec<usize>>();
                    let steps = tvec!(1; rank);
                    output_shape_rules(s, inputs, outputs, axes, steps)
                }
            }
        })
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        unimplemented!();
        /*
        let input = target.outlet_fact(mapping[&node.inputs[0]])?.clone();
        let mut wire = mapping[&node.inputs[0]];
        for (ix, (&b, &e)) in self.starts.iter().zip(self.ends.iter()).enumerate() {
            let axis = self.axes.as_ref().map(|axes| axes[ix]).unwrap_or(ix);
            let dim = input.shape.dim(axis);
            if let Ok(dim) = dim.to_integer() {
                let b = (if b >= 0 { b.min(dim as isize) } else { dim as isize + b }) as usize;
                let e = (if e >= 0 { e.min(dim as isize) } else { dim as isize + e }) as usize;
                if b > 0 || e < dim as usize {
                    wire = target.wire_node(
                        format!("{}-axis-{}", node.name, axis),
                        tract_core::ops::array::Slice::new(axis, b, e),
                        [wire].as_ref(),
                    )?[0];
                }
            } else {
                bail!("Can't translate slice: axis={} dim={} b={} e={}", axis, dim, b, e)
            }
        }
        target.rename_node(wire.node, &*node.name)?;
        Ok(tvec!(wire))
        */
    }

    inference_op_as_op!();
}
*/
