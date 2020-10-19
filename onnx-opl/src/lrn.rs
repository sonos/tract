use tract_ndarray::prelude::*;
use tract_nnef::internal::*;

#[derive(Debug, Clone, Default, Educe)]
#[educe(Hash)]
pub struct Lrn {
    #[educe(Hash(method = "hash_f32"))]
    pub alpha: f32,
    #[educe(Hash(method = "hash_f32"))]
    pub beta: f32,
    #[educe(Hash(method = "hash_f32"))]
    pub bias: f32,
    pub size: usize,
}

tract_data::impl_dyn_hash!(Lrn);

impl Lrn {
    fn eval_t<
        T: Datum + tract_num_traits::Float + tract_num_traits::FromPrimitive + ::std::iter::Sum,
    >(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let input = input.to_array_view::<T>()?;
        let channels = input.shape()[1];
        let output = Array::from_shape_fn(input.shape(), |mut coords| {
            let c = coords[1];
            let x = input[&coords];
            let c_min = c.saturating_sub((self.size - 1) / 2);
            let c_max = (c + ((self.size - 1).div_ceil(2))).min(channels - 1);
            let square_sum: T = (c_min..=c_max)
                .map(|c| {
                    coords[1] = c;
                    input[&coords].powi(2)
                })
                .sum();
            x / (T::from(self.bias).unwrap()
                + T::from(self.alpha).unwrap() / T::from(self.size).unwrap() * square_sum)
                .powf(T::from(self.beta).unwrap())
        });
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl Op for Lrn {
    fn name(&self) -> Cow<str> {
        "Lrn".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl EvalOp for Lrn {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl TypedOp for Lrn {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

pub fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.named("alpha").default(0.0001),
        TypeName::Scalar.named("beta").default(0.75),
        TypeName::Scalar.named("bias").default(1.0),
        TypeName::Integer.named("size"),
    ]
}

pub fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let lrn = node.op_as::<Lrn>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_onnx_lrn",
        &[input],
        &[
            ("alpha", numeric(lrn.alpha)),
            ("beta", numeric(lrn.beta)),
            ("bias", numeric(lrn.bias)),
            ("size", numeric(lrn.size)),
        ],
    )))
}

pub fn load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let alpha = invocation.named_arg_as(builder, "alpha")?;
    let beta = invocation.named_arg_as(builder, "beta")?;
    let bias = invocation.named_arg_as(builder, "bias")?;
    let size = invocation.named_arg_as(builder, "size")?;
    let op = Lrn { alpha, beta, bias, size };
    builder.wire(op, &[input])
}
