/// Constructs an abstract type.
#[macro_export]
macro_rules! atype {
    (_) =>
        ($crate::analyser::AType::Any);
    ($arg:expr) =>
        ($crate::analyser::AType::Only($arg));
}


/// Constructs an abstract shape.
#[macro_export]
macro_rules! ashape {
    () =>
        ($crate::analyser::AShape::Closed(vec![]));
    (..) =>
        ($crate::analyser::AShape::Open(vec![]));
    ($($arg:tt),+; ..) =>
        ($crate::analyser::AShape::Open(vec![$(adimension!($arg)),+]));
    ($($arg:tt),+) =>
        ($crate::analyser::AShape::Closed(vec![$(adimension!($arg)),+]));
}

/// Constructs an abstract dimension.
#[macro_export]
macro_rules! adimension {
    (_) =>
        ($crate::analyser::ADimension::Any);
    ($arg:expr) =>
        ($crate::analyser::ADimension::Only($arg));
}

/// Constructs an abstract value.
#[macro_export]
macro_rules! avalue {
    (_) =>
        ($crate::analyser::AValue::Any);
    ($arg:expr) =>
        ($crate::analyser::AValue::Only($arg));
}

#[cfg(tests)]
mod tests {
    #[test]
    fn shape_macro_closed_1() {
        assert_eq!(ashape![], AShape::Closed(vec![]));
    }

    #[test]
    fn shape_macro_closed_2() {
        assert_eq!(ashape![1], AShape::Closed(vec![ADimension::Only(1)]));
    }

    #[test]
    fn shape_macro_closed_3() {
        assert_eq!(ashape![(1 + 1)], AShape::Closed(vec![ADimension::Only(2)]));
    }

    #[test]
    fn shape_macro_closed_4() {
        assert_eq!(
            ashape![_, 2],
            AShape::Closed(vec![
                ADimension::Any,
                ADimension::Only(2)
            ])
        );
    }

    #[test]
    fn shape_macro_closed_5() {
        assert_eq!(
            ashape![(1 + 1), _, 2],
            AShape::Closed(vec![
                ADimension::Only(2),
                ADimension::Any,
                ADimension::Only(2)
            ])
        );
    }

    #[test]
    fn shape_macro_open_1() {
        assert_eq!(ashape![..], AShape::Open(vec![]));
    }

    #[test]
    fn shape_macro_open_2() {
        assert_eq!(ashape![1; ..], AShape::Open(vec![ADimension::Only(1)]));
    }

    #[test]
    fn shape_macro_open_3() {
        assert_eq!(ashape![(1 + 1); ..], AShape::Open(vec![ADimension::Only(2)]));
    }

    #[test]
    fn shape_macro_open_4() {
        assert_eq!(
            ashape![_, 2; ..],
            AShape::Open(vec![
                ADimension::Any,
                ADimension::Only(2)
            ])
        );
    }

    #[test]
    fn shape_macro_open_5() {
        assert_eq!(
            ashape![(1 + 1), _, 2; ..],
            AShape::Open(vec![
                ADimension::Only(2),
                ADimension::Any,
                ADimension::Only(2)
            ])
        );
    }
}
