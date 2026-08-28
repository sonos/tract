use tract_nnef::internal::*;
use tract_nnef::tract_ndarray::Dimension;
use tract_nnef::tract_num_traits::Zero;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_non_zero",
        &parameters(),
        &[("output", TypeName::Integer.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

/// Coordinates of the non-zero elements of the input, as an i64 tensor of shape
/// `[input_rank, count]`, column `i` holding the coordinates of the i-th non-zero
/// element in row-major order.
///
/// Accepts a non-quantized number or boolean input. `count` depends on the input values
/// rather than on its shape, so it is carried as a free symbol which the plan binds from
/// the actual output at eval time. Every node needs its own symbol: two NonZero outputs
/// have unrelated lengths. The symbol name is carried by the NNEF `count_symbol`
/// attribute, and a fresh one is minted when it is absent.
///
/// The binding happens once per turn, so a NonZero inside a Scan body keeps the count of
/// the first iteration for all of them: only a count that is constant across iterations
/// is representable there.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NonZero {
    pub count: Symbol,
}

impl NonZero {
    fn collect<T: Datum>(input: &Tensor, is_set: impl Fn(&T) -> bool) -> TractResult<Tensor> {
        let view = input.to_plain_array_view::<T>()?;
        let count = view.iter().filter(|v| is_set(v)).count();
        let mut output = Tensor::zero::<i64>(&[input.rank(), count])?;
        let mut view_mut: tract_ndarray::ArrayViewMut2<i64> =
            output.to_plain_array_view_mut::<i64>()?.into_dimensionality()?;
        for (i, (coords, _)) in view.indexed_iter().filter(|(_, v)| is_set(v)).enumerate() {
            view_mut
                .index_axis_mut(tract_ndarray::Axis(1), i)
                .assign(&coords.as_array_view().map(|d| *d as i64));
        }
        Ok(output)
    }

    fn eval_t<T: Datum + Zero>(input: &Tensor) -> TractResult<Tensor> {
        Self::collect(input, |v: &T| !v.is_zero())
    }
}

impl Op for NonZero {
    fn name(&self) -> StaticName {
        "NonZero".into()
    }

    op_as_typed_op!();
}

impl EvalOp for NonZero {
    fn is_pure_function(&self) -> bool {
        true
    }

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let output = if input.datum_type() == bool::datum_type() {
            Self::collect(&input, |v: &bool| *v)?
        } else {
            dispatch_numbers!(Self::eval_t(input.datum_type())(&input))?
        };
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for NonZero {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        ensure!(
            dt == bool::datum_type() || (dt.is_number() && !dt.is_quantized()),
            "NonZero expects a plain number or boolean input, got {dt:?}"
        );
        Ok(tvec!(i64::fact([inputs[0].rank().to_dim(), self.count.to_dim()])))
    }

    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![TypeName::Any.tensor().named("input"), TypeName::String.named("count_symbol")]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &NonZero) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_onnx_non_zero",
        &[input],
        &[("count_symbol", string(op.count.to_string()))],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let count = match invocation.get_named_arg_as::<String>(builder, "count_symbol")? {
        Some(name) => builder.model.symbols.sym(&name),
        None => builder.model.symbols.new_with_prefix("x"),
    };
    builder.wire(NonZero { count }, &[input])
}
