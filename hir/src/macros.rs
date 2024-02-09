#[macro_export]
macro_rules! to_typed {
    () => {
        fn to_typed(
            &self,
            _source: &$crate::infer::InferenceModel,
            node: &$crate::infer::InferenceNode,
            target: &mut TypedModel,
            mapping: &std::collections::HashMap<OutletId, OutletId>,
        ) -> TractResult<TVec<OutletId>> {
            let inputs = node.inputs.iter().map(|m| mapping[m]).collect::<TVec<_>>();
            target.wire_node(&*node.name, self.clone(), &*inputs)
        }
    };
}

/// Constructs a type fact.
#[macro_export]
macro_rules! typefact {
    (_) => {
        $crate::infer::TypeFactoid::default()
    };
    ($arg:expr) => {{
        let fact: $crate::infer::TypeFactoid = $crate::infer::GenericFactoid::Only($arg);
        fact
    }};
}

/// Constructs a shape fact.
#[macro_export]
macro_rules! shapefactoid {
    () =>
        ($crate::infer::ShapeFactoid::closed(tvec![]));
    (..) =>
        ($crate::infer::ShapeFactoid::open(tvec![]));
    ($($arg:tt),+; ..) =>
        ($crate::infer::ShapeFactoid::open(tvec![$($crate::dimfact!($arg)),+]));
    ($($arg:tt),+) =>
        ($crate::infer::ShapeFactoid::closed(tvec![$($crate::dimfact!($arg)),+]));
}

/// Constructs a dimension fact.
#[macro_export]
macro_rules! dimfact {
    (_) => {
        $crate::infer::DimFact::default()
    };
    (S) => {
        $crate::infer::GenericFactoid::Only(tract_pulse::internal::stream_dim())
    };
    ($arg:expr) => {
        $crate::infer::GenericFactoid::Only($arg.to_dim())
    };
}

/// Constructs an value fact.
#[macro_export]
macro_rules! valuefact {
    (_) => {
        $crate::infer::ValueFact::default()
    };
    ($arg:expr) => {{
        let fact: $crate::infer::ValueFact = $crate::infer::GenericFactoid::Only($arg);
        fact
    }};
}
