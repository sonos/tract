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