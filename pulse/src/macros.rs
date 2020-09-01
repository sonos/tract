#[macro_export]
macro_rules! pulsed_op_to_typed_op {
    () => {
        fn to_typed(&self) -> Box<dyn TypedOp> {
            tract_core::dyn_clone::clone_box(self)
        }
    };
}

macro_rules! submit_op_pulsifier {
    ($op: ty, $func: expr) => {
        inventory::submit!(OpPulsifier {
            type_id: std::any::TypeId::of::<$op>(),
            func: |source: &TypedModel,
                   node: &TypedNode,
                   target: &mut PulsedModel,
                   mapping: &HashMap<OutletId, OutletId>,
                   pulse: usize|
             -> TractResult<TVec<OutletId>> {
                let op = node.op_as::<$op>().unwrap();
                ($func)(op, source, node, target, mapping, pulse)
            },
            name: stringify!($op)
        });
    };
}
