use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub(crate) fn reduce(
    ctx: &ParsingContext,
    node: &NodeProto,
    reducer: tract_hir::ops::nn::Reducer,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    // this is crazy. sum changed semantics at opset 13, other reducers switched at 18!
    if (ctx.onnx_operator_set_version >= 13 && "ReduceSum" == node.op_type)
        || ctx.onnx_operator_set_version >= 18
    {
        let have_axis_input = node.input.len() == 2;
        let keep_dims = node.get_attr_opt("keepdims")?.unwrap_or(1i64) == 1;
        let noop_with_empty_axes = node.get_attr_opt("noop_with_empty_axes")?.unwrap_or(0i64) == 1;
        Ok((
            expand(ReduceSum13 { have_axis_input, keep_dims, noop_with_empty_axes, reducer }),
            vec![],
        ))
    } else {
        let axes = node.get_attr_opt_vec("axes")?;
        let keep_dims = node.get_attr_opt("keepdims")?.unwrap_or(1i64) == 1;
        Ok((expand(tract_hir::ops::nn::Reduce::new(axes, keep_dims, reducer)), vec![]))
    }
}

#[derive(Debug, Clone, Hash)]
struct ReduceSum13 {
    pub have_axis_input: bool,
    pub keep_dims: bool,
    pub noop_with_empty_axes: bool,
    pub reducer: tract_hir::ops::nn::Reducer,
}



impl Expansion for ReduceSum13 {
    fn name(&self) -> Cow<str> {
        "Reduce13".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1 + self.have_axis_input as usize)?;
        check_output_arity(outputs, 1)?;
        if let tract_hir::ops::nn::Reducer::ArgMax(_) | tract_hir::ops::nn::Reducer::ArgMin(_) =
            self.reducer
        {
            s.equals(&outputs[0].datum_type, DatumType::I64)?;
        } else {
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        }
        if self.have_axis_input {
            s.given_2(&inputs[0].rank, &inputs[1].value, move |s, rank, axes| {
                let mut axes = axes.cast_to::<i64>()?.as_slice::<i64>()?.to_vec();
                if axes.len() == 0 && !self.noop_with_empty_axes {
                    axes = (0..rank).collect()
                };
                let op = tract_hir::ops::nn::Reduce::new(
                    Some(axes.clone()),
                    self.keep_dims,
                    self.reducer,
                );
                if self.keep_dims {
                    s.equals(&inputs[0].rank, &outputs[0].rank)?;
                } else {
                    s.equals(inputs[0].rank.bex() - axes.len() as i64, &outputs[0].rank)?;
                }
                s.given(&inputs[0].shape, move |s, shape| {
                    let out_shape = op.output_shape(&shape);
                    s.equals(&outputs[0].shape, out_shape)
                })
            })
        } else {
            s.given(&inputs[0].rank, move |s, rank| {
                let axes = if self.noop_with_empty_axes { vec![] } else { (0..rank).collect() };
                let op = tract_hir::ops::nn::Reduce::new(
                    Some(axes.clone()),
                    self.keep_dims,
                    self.reducer,
                );
                if self.keep_dims {
                    s.equals(&inputs[0].rank, &outputs[0].rank)?;
                } else {
                    s.equals(inputs[0].rank.bex() - axes.len() as i64, &outputs[0].rank)?;
                }
                s.given(&inputs[0].shape, move |s, shape| {
                    let out_shape = op.output_shape(&shape);
                    s.equals(&outputs[0].shape, out_shape)
                })
            })
        }
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let axes: Vec<i64> = if self.have_axis_input {
            model
                .outlet_fact(inputs[1])?
                .konst
                .as_ref()
                .context("expected axes as a constant")?
                .as_slice::<i64>()?
                .to_vec()
        } else {
            vec![]
        };
        let axes: Vec<i64> = if axes.len() == 0 {
            if self.noop_with_empty_axes {
                vec![]
            } else {
                (0..model.outlet_fact(inputs[0])?.rank()).map(|ax| ax as i64).collect()
            }
        } else {
            axes
        };
        let op = tract_hir::ops::nn::Reduce::new(Some(axes), self.keep_dims, self.reducer);
        op.wire(prefix, model, &inputs[0..1])
    }
}
