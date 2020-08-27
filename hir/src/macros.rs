#[macro_export]
macro_rules! op_hir {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["core"]
        }
    };
}

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
        $crate::infer::GenericFactoid::Only($crate::tract_core::pulse::stream_dim())
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

/// Tries to unwrap an option, or returns Ok(None) otherwise.
#[macro_export]
macro_rules! unwrap_or_none {
    ($e:expr) => {{
        let e = $e;
        if e.is_none() {
            return Ok(None);
        } else {
            e.unwrap()
        }
    }};
}

#[cfg(tests)]
mod tests {
    #[test]
    fn shape_macro_closed_1() {
        assert_eq!(shapefactoid![], ShapeFactoid::closed(tvec![]));
    }

    #[test]
    fn shape_macro_closed_2() {
        assert_eq!(shapefactoid![1], ShapeFactoid::closed(tvec![GenericFactoid::Only(1)]));
    }

    #[test]
    fn shape_macro_closed_3() {
        assert_eq!(shapefactoid![(1 + 1)], ShapeFactoid::closed(vec![GenericFactoid::Only(2)]));
    }

    #[test]
    fn shape_macro_closed_4() {
        assert_eq!(
            shapefactoid![_, 2],
            ShapeFactoid::closed(vec![GenericFactoid::Any, GenericFactoid::Only(2)])
        );
    }

    #[test]
    fn shape_macro_closed_5() {
        assert_eq!(
            shapefactoid![(1 + 1), _, 2],
            ShapeFactoid::closed(vec![
                GenericFactoid::Only(2),
                GenericFactoid::Any,
                GenericFactoid::Only(2),
            ])
        );
    }

    #[test]
    fn shape_macro_open_1() {
        assert_eq!(shapefactoid![..], ShapeFactoid::open(tvec![]));
    }

    #[test]
    fn shape_macro_open_2() {
        assert_eq!(shapefactoid![1; ..], ShapeFactoid::open(vec![GenericFactoid::Only(1)]));
    }

    #[test]
    fn shape_macro_open_3() {
        assert_eq!(shapefactoid![(1 + 1); ..], ShapeFactoid::open(vec![GenericFactoid::Only(2)]));
    }

    #[test]
    fn shape_macro_open_4() {
        assert_eq!(
            shapefactoid![_, 2; ..],
            ShapeFactoid::open(vec![GenericFactoid::Any, GenericFactoid::Only(2)])
        );
    }

    #[test]
    fn shape_macro_open_5() {
        assert_eq!(
            shapefactoid![(1 + 1), _, 2; ..],
            ShapeFactoid::open(tvec![
                GenericFactoid::Only(2),
                GenericFactoid::Any,
                GenericFactoid::Only(2),
            ])
        );
    }
}
