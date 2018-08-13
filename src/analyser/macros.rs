/// Constructs a type fact.
#[macro_export]
macro_rules! typefact {
    (_) => {
        $crate::analyser::TypeFact::default()
    };
    ($arg:expr) => {{
        let fact: $crate::analyser::TypeFact = $crate::analyser::GenericFact::Only($arg);
        fact
    }};
}

/// Constructs a shape fact.
#[macro_export]
macro_rules! shapefact {
    () =>
        ($crate::analyser::ShapeFact::closed(vec![]));
    (..) =>
        ($crate::analyser::ShapeFact::open(vec![]));
    ($($arg:tt),+; ..) =>
        ($crate::analyser::ShapeFact::open(vec![$(dimfact!($arg)),+]));
    ($($arg:tt),+) =>
        ($crate::analyser::ShapeFact::closed(vec![$(dimfact!($arg)),+]));
}

/// Constructs a dimension fact.
#[macro_export]
macro_rules! dimfact {
    (_) => {
        $crate::analyser::DimFact::default()
    };
    (S) => {
        $crate::analyser::DimFact::Streamed
    };
    ($arg:expr) => {
        $crate::analyser::GenericFact::Only($arg.to_dim())
    };
}

/// Constructs an value fact.
#[macro_export]
macro_rules! valuefact {
    (_) => {
        $crate::analyser::ValueFact::default()
    };
    ($arg:expr) => {{
        let fact: $crate::analyser::ValueFact = $crate::analyser::GenericFact::Only($arg);
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
        assert_eq!(shapefact![], ShapeFact::closed(vec![]));
    }

    #[test]
    fn shape_macro_closed_2() {
        assert_eq!(shapefact![1], ShapeFact::closed(vec![GenericFact::Only(1)]));
    }

    #[test]
    fn shape_macro_closed_3() {
        assert_eq!(
            shapefact![(1 + 1)],
            ShapeFact::closed(vec![GenericFact::Only(2)])
        );
    }

    #[test]
    fn shape_macro_closed_4() {
        assert_eq!(
            shapefact![_, 2],
            ShapeFact::closed(vec![GenericFact::Any, GenericFact::Only(2)])
        );
    }

    #[test]
    fn shape_macro_closed_5() {
        assert_eq!(
            shapefact![(1 + 1), _, 2],
            ShapeFact::closed(vec![GenericFact::Only(2), GenericFact::Any, GenericFact::Only(2)])
        );
    }

    #[test]
    fn shape_macro_open_1() {
        assert_eq!(shapefact![..], ShapeFact::open(vec![]));
    }

    #[test]
    fn shape_macro_open_2() {
        assert_eq!(shapefact![1; ..], ShapeFact::open(vec![GenericFact::Only(1)]));
    }

    #[test]
    fn shape_macro_open_3() {
        assert_eq!(
            shapefact![(1 + 1); ..],
            ShapeFact::open(vec![GenericFact::Only(2)])
        );
    }

    #[test]
    fn shape_macro_open_4() {
        assert_eq!(
            shapefact![_, 2; ..],
            ShapeFact::open(vec![GenericFact::Any, GenericFact::Only(2)])
        );
    }

    #[test]
    fn shape_macro_open_5() {
        assert_eq!(
            shapefact![(1 + 1), _, 2; ..],
            ShapeFact::open(vec![GenericFact::Only(2), GenericFact::Any, GenericFact::Only(2)])
        );
    }
}
