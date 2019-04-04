macro_rules! with_T {
    ($op:ty) => {
        |pb: &$crate::tfpb::node_def::NodeDef| -> $crate::tract_core::TractResult<Box<$crate::tract_core::ops::Op>> {
            let datum_type: $crate::tract_core::internal::TypeFact =
                pb.get_attr_datum_type("T")?.into();
            Ok(Box::new(<$op>::new(datum_type)) as _)
        }
    };
}
