use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn slice(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let v = ctx.onnx_operator_set_version;
    if v >= 1 && v < 10 {
        slice1(ctx, node)
    } else {
        slice10(ctx, node)
    }
}

fn slice1(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    let begin = node.get_attr_vec("starts")?;
    let end = node.get_attr_vec("ends")?;
    Ok((expand(Slice1::new(axes, begin, end)), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Slice1 {
    axes: Option<Vec<usize>>,
    starts: Vec<isize>,
    ends: Vec<isize>,
}

tract_data::impl_dyn_hash!(Slice1);

impl Expansion for Slice1 {
    fn name(&self) -> Cow<str> {
        "Slice1".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        if self.axes.is_none() {
            s.equals(&inputs[0].rank, self.starts.len() as i64)?;
            s.equals(&inputs[0].rank, self.ends.len() as i64)?;
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
                    if let Ok(d) = d.to_isize() {
                        if b > d {
                            b = d.into();
                        }
                        if e > d {
                            e = d.into();
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

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = target.outlet_fact(inputs[0])?.clone();
        let mut wire = inputs[0];
        for (ix, (&b, &e)) in self.starts.iter().zip(self.ends.iter()).enumerate() {
            let axis = self.axes.as_ref().map(|axes| axes[ix]).unwrap_or(ix);
            let dim = &input.shape[axis];
            if let Ok(dim) = dim.to_isize() {
                let b = (if b >= 0 { b.min(dim) } else { dim + b }) as usize;
                let e = (if e >= 0 { e.min(dim) } else { dim + e }) as usize;
                if b > 0 || e < dim as usize {
                    wire = target.wire_node(
                        format!("{}.axis-{}", prefix, axis),
                        tract_hir::ops::array::Slice::new(axis, b, e),
                        [wire].as_ref(),
                    )?[0];
                }
            } else {
                bail!("Can't translate slice: axis={} dim={} b={} e={}", axis, dim, b, e)
            }
        }
        target.rename_node(wire.node, &*prefix)?;
        Ok(tvec!(wire))
    }
}

fn slice10(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut optional_inputs = crate::model::optional_inputs(node).skip(3);
    Ok((
        expand(tract_hir::ops::array::StridedSlice {
            begin_mask: 0,
            end_mask: 0,
            shrink_axis_mask: 0,
            optional_axes_input: optional_inputs.next().unwrap(),
            optional_steps_input: optional_inputs.next().unwrap(),
        }),
        vec![],
    ))
}
