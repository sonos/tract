#[macro_export]
macro_rules! pulsed_op_to_typed_op {
    () => {
        fn to_typed(&self) -> Box<dyn TypedOp> {
            tract_core::dyn_clone::clone_box(self)
        }
    };
}

#[macro_export]
macro_rules! register_all_mod {
    ($($m: ident),*) => {
        pub fn register_all(inventory: &mut HashMap<TypeId, OpPulsifier>) {
            $( $m::register_all(inventory); )*
        }
    }
}

#[macro_export]
macro_rules! register_all {
    ($($op: ty: $func: expr),*) => {
        pub fn register_all(inventory: &mut HashMap<TypeId, OpPulsifier>) {
            $(
            inventory.insert(
                std::any::TypeId::of::<$op>(),
                OpPulsifier {
                    type_id: std::any::TypeId::of::<$op>(),
                    func: |source: &TypedModel,
                           node: &TypedNode,
                           target: &mut PulsedModel,
                           mapping: &HashMap<OutletId, OutletId>,
                           symbol: &Symbol,
                           pulse: &TDim|
                     -> TractResult<Option<TVec<OutletId>>> {
                        let op = node.op_as::<$op>().unwrap();
                        ($func)(op, source, node, target, mapping, symbol, pulse)
                    },
                    name: stringify!($op)
                }
            );)*
        }
    };
}
