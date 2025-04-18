use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::array::TypedConcat;

#[derive(Default, Clone, Debug, Hash)]
pub struct DynKeyValueCache;

impl Op for DynKeyValueCache {
    fn name(&self) -> Cow<str> {
        "DynamicKeyValueCache".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for DynKeyValueCache {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let _input = args_1!(inputs);

        Ok(tvec![])
    }
}

impl TypedOp for DynKeyValueCache {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!())
    }

    as_op!();
}

/// Search pattern => Input -> Concat -> Output
pub fn dynamic_kv_cache_rule(
    _ctx: &(),
    _model: &TypedModel,
    _node: &TypedNode,
    _node_name: &str,
    _op: &TypedConcat,
) -> TractResult<Option<TypedModelPatch>> {
    
    Ok(None)
}
