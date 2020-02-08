
/*
#[macro_export]
macro_rules! to_typed {
    () => {
        fn to_typed(
            &self,
            _source: &$crate::infer::InferenceModel,
            node: &$crate::infer::InferenceNode,
            target: &mut TypedModel,
            mapping: &HashMap<OutletId, OutletId>,
        ) -> TractResult<TVec<OutletId>> {
            let inputs = node.inputs.iter().map(|m| mapping[m]).collect::<TVec<_>>();
            target.wire_node(&*node.name, self.clone(), &*inputs)
        }
    }
}
*/
