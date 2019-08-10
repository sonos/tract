macro_rules! with_T {
    ($op:ty) => {
        |_, pb: &$crate::tfpb::node_def::NodeDef| -> $crate::tract_core::TractResult<Box<dyn $crate::tract_core::ops::InferenceOp>> {
            let datum_type: $crate::tract_core::internal::TypeFact =
                pb.get_attr_datum_type("T")?.into();
            Ok(Box::new(<$op>::new(datum_type)) as _)
        }
    };
}
