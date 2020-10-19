use tract_ndarray::prelude::*;
use tract_nnef::internal::*;

#[derive(Debug, PartialEq, Clone, Hash)]
pub struct OneHot {
    pub axis: usize,
    pub dim: usize,
    pub off: Arc<Tensor>,
    pub on: Arc<Tensor>,
}

tract_data::impl_dyn_hash!(OneHot);

impl Op for OneHot {
    fn name(&self) -> Cow<str> {
        "Onehot".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl TypedOp for OneHot {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape = inputs[0].shape.to_tvec();
        shape.insert(self.axis, self.dim.to_dim());
        Ok(tvec!(TypedFact::dt_shape(self.off.datum_type(), &*shape)?))
    }

    as_op!();
}

impl EvalOp for OneHot {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        unsafe {
            let mut output = self.off.broadcast_scalar_to_shape(&mut shape)?;
            dispatch_datum_by_size!(Self::eval_t(self.off.datum_type())(
                self,
                &input,
                &mut output
            ))?;
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}

impl OneHot {
    unsafe fn eval_t<T: Datum + Clone>(
        &self,
        input: &Tensor,
        output: &mut Tensor,
    ) -> TractResult<()> {
        let on = self.on.to_scalar_unchecked::<T>();
        let mut shape: TVec<usize> = input.shape().into();
        shape.insert(self.axis, self.dim);
        let mut array = output.to_array_view_mut_unchecked::<T>();
        let input = input.cast_to::<i32>()?;
        let input = input.to_array_view::<i32>()?;
        for icoord in tract_ndarray::indices_of(&input) {
            let mut ocoord: Vec<usize> = icoord.slice().into();
            let coord = input[&icoord];
            let coord = if coord < 0 { coord + self.dim as i32 } else { coord } as usize;
            ocoord.insert(self.axis, coord);
            array[&*ocoord] = on.clone();
        }
        Ok(())
    }
}

pub fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Integer.named("axis"),
        TypeName::Integer.named("dim"),
        TypeName::Scalar.named("value_off").default(0.0),
        TypeName::Scalar.named("value_on").default(1.0),
    ]
}

pub fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let one_hot = node.op_as::<OneHot>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_onnx_one_hot",
        &[input],
        &[
            ("axis", numeric(one_hot.axis)),
            ("dim", numeric(one_hot.dim)),
            ("value_off", numeric(one_hot.off.cast_to_scalar::<f32>()?)),
            ("value_on", numeric(one_hot.on.cast_to_scalar::<f32>()?)),
        ],
    )))
}

pub fn load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let dim = invocation.named_arg_as(builder, "dim")?;
    let off = invocation.named_arg_as(builder, "value_off")?;
    let on = invocation.named_arg_as(builder, "value_on")?;
    let op = OneHot { axis, dim, on, off };
    builder.wire(op, &[input])
}
