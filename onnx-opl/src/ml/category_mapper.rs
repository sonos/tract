use std::hash::*;
use tract_nnef::internal::*;
use tract_nnef::tract_core::itertools::Itertools;
use tract_smallvec::SmallVec;

pub fn register(registry: &mut Registry) {
    /*
    registry.register_primitive(
    "tract_onnx_ml_category_mapper_to_int",
    &parameters_to_int(),
    load_to_int,
    );
    registry.register_primitive(
    "tract_onnx_ml_category_mapper_to_string",
    &parameters_to_string(),
    load_to_string,
    );
    registry.register_dumper(TypeId::of::<CategoryMapper<i64, String>>(), dump_to_string);
    registry.register_dumper(TypeId::of::<CategoryMapper<String, i64>>(), dump_to_int);
    */
}

#[derive(Clone, Debug, Hash)]
pub struct DirectLookup {
    values: Arc<Tensor>,
    fallback_value: Arc<Tensor>,
}

impl_dyn_hash!(DirectLookup);

impl DirectLookup {
    pub fn new(values: Arc<Tensor>, fallback_value: Arc<Tensor>) -> TractResult<DirectLookup> {
        Ok(DirectLookup { values, fallback_value })
    }

    fn eval_t<T: Datum>(&self, input: &Tensor) -> TractResult<Tensor> {
        let values = self.values.as_slice::<T>()?;
        let fallback_value = self.fallback_value.to_scalar::<T>()?;
        Ok(input
            .to_array_view::<i32>()?
            .mapv(|ix| values.get(ix as usize).unwrap_or(&fallback_value).clone())
            .into_tensor())
    }
}

impl Op for DirectLookup {
    fn name(&self) -> Cow<str> {
        "DirectLookup".into()
    }

    fn op_families(&self) -> &'static [&'static str] {
        &["onnx-ml"]
    }

    op_as_typed_op!();
}

impl EvalOp for DirectLookup {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let output = dispatch_hash!(Self::eval_t(self.values.datum_type())(self, &input))?;
        Ok(tvec!(output.into_arc_tensor()))
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
        Ok(tvec!(TypedFact::dt_shape(self.values.datum_type(), inputs[0].shape.iter())))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        Invariants::new_element_wise(model, node)
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

impl_dyn_hash!(ReverseLookup);

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

    fn op_families(&self) -> &'static [&'static str] {
        &["onnx-ml"]
    }

    op_as_typed_op!();
}

impl EvalOp for ReverseLookup {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let output = dispatch_hash!(Self::eval_t(self.keys.datum_type())(self, &input))?;
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl TypedOp for ReverseLookup {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(i32::datum_type(), inputs[0].shape.iter())))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        Invariants::new_element_wise(model, node)
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

/*
   fn parameters_to_int() -> Vec<Parameter> {
   vec![
   TypeName::String.tensor().named("input"),
   TypeName::String.tensor().named("keys"),
   TypeName::Scalar.tensor().named("values"),
   TypeName::Scalar.named("default").default(-1),
   ]
   }

   fn parameters_to_string() -> Vec<Parameter> {
   vec![
   TypeName::Scalar.tensor().named("input"),
   TypeName::Scalar.tensor().named("keys"),
   TypeName::String.tensor().named("values"),
   TypeName::String.named("default").default(""),
   ]
   }

   fn dump_to_string(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
   let input = ast.mapping[&node.inputs[0]].clone();
   let to_string = node.op_as::<CategoryMapper<i64, String>>().context("wrong op")?;
   let (keys, values) =
   to_string.hash.iter().map(|(k, v)| (*k, v.clone())).unzip::<i64, String, Vec<_>, Vec<_>>();
   let keys = ast.konst_variable(format!("{}_keys", node.name), &rctensor1(&keys));
   let values = ast.konst_variable(format!("{}_values", node.name), &rctensor1(&values));
   Ok(Some(invocation(
   "tract_onnx_ml_category_mapper_to_string",
   &[input, keys, values],
   &[("default", string(to_string.default.clone()))],
   )))
   }

   fn dump_to_int(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
   let input = ast.mapping[&node.inputs[0]].clone();
   let from_string = node.op_as::<CategoryMapper<String, i64>>().context("wrong op")?;
   let (keys, values) = from_string
   .hash
   .iter()
   .map(|(k, v)| (k.clone(), *v))
   .unzip::<String, i64, Vec<_>, Vec<_>>();
   let keys = ast.konst_variable(format!("{}_keys", node.name), &rctensor1(&keys));
   let values = ast.konst_variable(format!("{}_values", node.name), &rctensor1(&values));
   Ok(Some(invocation(
   "tract_onnx_ml_category_mapper_to_int",
   &[input, keys, values],
   &[("default", numeric(from_string.default))],
   )))
   }

   fn load_to_string(
   builder: &mut ModelBuilder,
   invocation: &ResolvedInvocation,
   ) -> TractResult<TVec<OutletId>> {
   let input = invocation.named_arg_as(builder, "input")?;
   let default = invocation.named_arg_as(builder, "default")?;
   let keys: Arc<Tensor> = invocation.named_arg_as(builder, "keys")?;
   let values: Arc<Tensor> = invocation.named_arg_as(builder, "values")?;
   let hash = keys
   .as_slice::<i64>()?
   .iter()
   .copied()
   .zip(values.as_slice::<String>()?.iter().cloned())
   .collect();
   let op = CategoryMapper::<i64, String> { hash, default };
   builder.wire(op, &[input])
   }

   fn load_to_int(
   builder: &mut ModelBuilder,
   invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let default = invocation.named_arg_as(builder, "default")?;
    let keys: Arc<Tensor> = invocation.named_arg_as(builder, "keys")?;
    let values: Arc<Tensor> = invocation.named_arg_as(builder, "values")?;
    let hash = keys
        .as_slice::<String>()?
        .iter()
        .cloned()
        .zip(values.as_slice::<i64>()?.iter().copied())
        .collect();
    let op = CategoryMapper::<String, i64> { hash, default };
    builder.wire(op, &[input])
}
*/
