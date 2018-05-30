/// Constructs a type fact.
#[macro_export]
macro_rules! typefact {
    (_) =>
        ($crate::analyser::TypeFact::Any);
    ($arg:expr) =>
        ($crate::analyser::TypeFact::Only($arg));
}


/// Constructs a shape fact.
#[macro_export]
macro_rules! shapefact {
    () =>
        ($crate::analyser::ShapeFact::Closed(vec![]));
    (..) =>
        ($crate::analyser::ShapeFact::Open(vec![]));
    ($($arg:tt),+; ..) =>
        ($crate::analyser::ShapeFact::Open(vec![$(dimfact!($arg)),+]));
    ($($arg:tt),+) =>
        ($crate::analyser::ShapeFact::Closed(vec![$(dimfact!($arg)),+]));
}

/// Constructs a dimension fact.
#[macro_export]
macro_rules! dimfact {
    (_) =>
        ($crate::analyser::DimFact::Any);
    ($arg:expr) =>
        ($crate::analyser::DimFact::Only($arg));
}

/// Constructs an value fact.
#[macro_export]
macro_rules! valuefact {
    (_) =>
        ($crate::analyser::ValueFact::Any);
    ($arg:expr) =>
        ($crate::analyser::ValueFact::Only($arg));
}

#[cfg(tests)]
mod tests {
    #[test]
    fn shape_macro_closed_1() {
        assert_eq!(shapefact![], ShapeFact::Closed(vec![]));
    }

    #[test]
    fn shape_macro_closed_2() {
        assert_eq!(shapefact![1], ShapeFact::Closed(vec![DimFact::Only(1)]));
    }

    #[test]
    fn shape_macro_closed_3() {
        assert_eq!(shapefact![(1 + 1)], ShapeFact::Closed(vec![DimFact::Only(2)]));
    }

    #[test]
    fn shape_macro_closed_4() {
        assert_eq!(
            shapefact![_, 2],
            ShapeFact::Closed(vec![
                DimFact::Any,
                DimFact::Only(2)
            ])
        );
    }

    #[test]
    fn shape_macro_closed_5() {
        assert_eq!(
            shapefact![(1 + 1), _, 2],
            ShapeFact::Closed(vec![
                DimFact::Only(2),
                DimFact::Any,
                DimFact::Only(2)
            ])
        );
    }

    #[test]
    fn shape_macro_open_1() {
        assert_eq!(shapefact![..], ShapeFact::Open(vec![]));
    }

    #[test]
    fn shape_macro_open_2() {
        assert_eq!(shapefact![1; ..], ShapeFact::Open(vec![DimFact::Only(1)]));
    }

    #[test]
    fn shape_macro_open_3() {
        assert_eq!(shapefact![(1 + 1); ..], ShapeFact::Open(vec![DimFact::Only(2)]));
    }

    #[test]
    fn shape_macro_open_4() {
        assert_eq!(
            shapefact![_, 2; ..],
            ShapeFact::Open(vec![
                DimFact::Any,
                DimFact::Only(2)
            ])
        );
    }

    #[test]
    fn shape_macro_open_5() {
        assert_eq!(
            shapefact![(1 + 1), _, 2; ..],
            ShapeFact::Open(vec![
                DimFact::Only(2),
                DimFact::Any,
                DimFact::Only(2)
            ])
        );
    }
}
