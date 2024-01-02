use std::hash::*;
use tract_itertools::Itertools;
use tract_nnef::internal::*;
use tract_smallvec::SmallVec;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_ml_direct_lookup",
        &parameters_direct_lookup(),
        &[("output", TypeName::Scalar.tensor())],
        load_direct_lookup,
    );
    registry.register_primitive(
        "tract_onnx_ml_reverse_lookup",
        &parameters_reverse_lookup(),
        &[("output", TypeName::Scalar.tensor())],
        load_reverse_lookup,
    );
    registry.register_dumper(dump_direct_lookup);
    registry.register_dumper(dump_reverse_lookup);
}

#[derive(Clone, Debug, Hash)]
pub struct DirectLookup {
    values: Arc<Tensor>,
    fallback_value: Arc<Tensor>,
}

impl DirectLookup {
    pub fn new(values: Arc<Tensor>, fallback_value: Arc<Tensor>) -> TractResult<DirectLookup> {
        Ok(DirectLookup { values, fallback_value })
    }

    fn eval_t<T: Datum>(&self, input: &Tensor) -> TractResult<Tensor> {
        let values = self.values.as_slice::<T>()?;
        let fallback_value = self.fallback_value.to_scalar::<T>()?;
        Ok(input
            .to_array_view::<i32>()?
            .mapv(|ix| values.get(ix as usize).unwrap_or(fallback_value).clone())
            .into_tensor())
    }
}

impl Op for DirectLookup {
    fn name(&self) -> Cow<str> {
        "DirectLookup".into()
    }

    op_as_typed_op!();
}

impl EvalOp for DirectLookup {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let output = dispatch_hash!(Self::eval_t(self.values.datum_type())(self, &input))?;
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for DirectLookup {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if self.values.datum_type() != self.fallback_value.datum_type() {
            bail!(
                "values and fallback value should be of the same type, got {:?}, {:?}",
                self.values,
                self.fallback_value
            )
        }
        Ok(tvec!(self.values.datum_type().fact(inputs[0].shape.iter())))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    as_op!();
}

#[derive(Clone, Debug)]
pub struct ReverseLookup {
    keys: Arc<Tensor>,
    index: HashMap<u64, SmallVec<[i32; 1]>>,
    fallback_value: i32,
}

#[allow(clippy::manual_hash_one)]
impl ReverseLookup {
    pub fn new(keys: Arc<Tensor>, fallback_value: i32) -> TractResult<ReverseLookup> {
        unsafe fn new_t<T: Datum + Hash>(keys: &Tensor) -> HashMap<u64, SmallVec<[i32; 1]>> {
            let keys = keys.as_slice_unchecked::<T>();
            let mut hashmap = HashMap::<u64, SmallVec<[i32; 1]>>::default();
            for (ix, k) in keys.iter().enumerate() {
                let mut hasher = hashmap.hasher().build_hasher();
                k.hash(&mut hasher);
                let u = hasher.finish();
                hashmap.entry(u).or_default().push(ix as i32);
            }
            hashmap
        }
        let index = unsafe { dispatch_hash!(new_t(keys.datum_type())(&keys)) };
        Ok(ReverseLookup { index, keys, fallback_value })
    }

    unsafe fn search_t<T: Datum + Hash>(&self, needle: &T) -> Option<i32> {
        let keys = self.keys.as_slice_unchecked::<T>();
        let mut hasher = self.index.hasher().build_hasher();
        needle.hash(&mut hasher);
        let u = hasher.finish();
        if let Some(candidates) = self.index.get(&u) {
            for candidate in candidates {
                if &keys[*candidate as usize] == needle {
                    return Some(*candidate);
                }
            }
        }
        None
    }

    fn eval_t<T: Datum + Hash>(&self, input: &Tensor) -> TractResult<Tensor> {
        unsafe {
            let mut output = Tensor::uninitialized_dt(i32::datum_type(), input.shape())?;
            for (i, o) in
                input.as_slice::<T>()?.iter().zip(output.as_slice_mut_unchecked::<i32>().iter_mut())
            {
                *o = self.search_t(i).unwrap_or(self.fallback_value);
            }
            Ok(output)
        }
    }
}

impl Hash for ReverseLookup {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.keys.hash(state);
        self.fallback_value.hash(state);
        self.index.iter().sorted().for_each(|v| Hash::hash(&v, state));
    }
}

impl Op for ReverseLookup {
    fn name(&self) -> Cow<str> {
        "ReverseLookup".into()
    }

    op_as_typed_op!();
}

impl EvalOp for ReverseLookup {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let output = dispatch_hash!(Self::eval_t(self.keys.datum_type())(self, &input))?;
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for ReverseLookup {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(i32::fact(inputs[0].shape.iter())))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    as_op!();
}

fn parameters_direct_lookup() -> Vec<Parameter> {
    vec![
        TypeName::String.tensor().named("input"),
        TypeName::Scalar.tensor().named("values"),
        TypeName::Scalar.tensor().named("fallback"),
    ]
}

fn parameters_reverse_lookup() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("keys"),
        TypeName::Scalar.named("fallback"),
    ]
}

fn dump_direct_lookup(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &DirectLookup,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let keys = ast.konst_variable(format!("{}.values", node.name), &op.values)?;
    let fallback = ast.konst_variable(format!("{}.fallback", node.name), &op.fallback_value)?;
    Ok(Some(invocation("tract_onnx_ml_direct_lookup", &[input, keys, fallback], &[])))
}

fn dump_reverse_lookup(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ReverseLookup,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let values = ast.konst_variable(format!("{}.keys", node.name), &op.keys)?;
    Ok(Some(invocation(
        "tract_onnx_ml_reverse_lookup",
        &[input, values],
        &[("fallback", numeric(op.fallback_value))],
    )))
}

fn load_direct_lookup(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let values: Arc<Tensor> = invocation.named_arg_as(builder, "values")?;
    let fallback_value = invocation.named_arg_as(builder, "fallback")?;
    let op = DirectLookup { fallback_value, values };
    builder.wire(op, &[input])
}

fn load_reverse_lookup(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let keys: isize = invocation.named_arg_as(builder, "keys")?;
    let fallback_value = invocation.named_arg_as(builder, "fallback")?;
    let op = ReverseLookup::new(fallback_value, keys as i32)?;
    builder.wire(op, &[input])
}
