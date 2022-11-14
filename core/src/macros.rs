#[macro_export]
macro_rules! dims {
    ($($dim:expr),*) => {
        ShapeFact::from(&[$(TDim::from($dim.clone())),*])
    }
}
