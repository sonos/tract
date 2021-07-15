use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub(crate) fn reduce(
    ctx: &ParsingContext,
    node: &NodeProto,
    reducer: tract_hir::ops::nn::Reducer,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    if ctx.onnx_operator_set_version < 13 {
        let axes = node.get_attr_opt_vec("axes")?;
        let keep_dims = node.get_attr_opt("keepdims")?.unwrap_or(1i64) == 1;
        Ok((expand(tract_hir::ops::nn::Reduce::new(axes, keep_dims, reducer)), vec![]))
    } else {
        let have_axis_input = node.input.len() == 2;
        let keep_dims = node.get_attr_opt("keepdims")?.unwrap_or(1i64) == 1;
        let noop_with_empty_axes = node.get_attr_opt("noop_with_empty_axes")?.unwrap_or(0i64) == 1;
        Ok((expand(Reduce13 { have_axis_input, keep_dims, noop_with_empty_axes, reducer }), vec![]))
    }
}

#[derive(Debug, Clone, Hash)]
struct Reduce13 {
    have_axis_input: bool,
    keep_dims: bool,
    noop_with_empty_axes: bool,
    reducer: tract_hir::ops::nn::Reducer,
}

impl_dyn_hash!(Reduce13);

impl Expansion for Reduce13 {
    fn name(&self) -> Cow<str> {
        "Reduce13".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1 + self.have_axis_input as usize)?;
        check_output_arity(&outputs, 1)?;
        s.given_2(&inputs[0].rank, &inputs[1].value, move |s, rank, axes| {
            let mut axes = axes.cast_to::<i64>()?.as_slice::<i64>()?.to_vec();
            if axes.len() == 0 && !self.noop_with_empty_axes {
                axes = (0..rank as i64).collect()
            };
            if let tract_hir::ops::nn::Reducer::ArgMax(_) | tract_hir::ops::nn::Reducer::ArgMin(_) =
                self.reducer
            {
                s.equals(&outputs[0].datum_type, DatumType::I64)?;
            } else {
                s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
            }
            if self.keep_dims {
                s.equals(&inputs[0].rank, &outputs[0].rank)?;
            } else {
                s.equals(inputs[0].rank.bex() - axes.len() as i64, &outputs[0].rank)?;
            }
            s.given(&inputs[0].shape, move |s, shape| {
                let op = tract_hir::ops::nn::Reduce::new(
                    Some(axes.clone()),
                    self.keep_dims,
                    self.reducer,
                );
                let out_shape = op.output_shape(&*shape);
                s.equals(&outputs[0].shape, out_shape)
            })
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(axes) = model.outlet_fact(inputs[1])?.konst.as_ref() {
            let mut axes = axes.cast_to::<i64>()?.as_slice::<i64>()?.to_vec();
            if axes.len() == 0 && !self.noop_with_empty_axes {
                axes = (0..model.outlet_fact(inputs[0])?.rank() as i64).collect()
            };
            let op = tract_hir::ops::nn::Reduce::new(Some(axes), self.keep_dims, self.reducer);
            op.wire(prefix, model, &inputs[0..1])
        } else {
            bail!("Need axes to be a constant")
        }
    }

    op_onnx!();
}
