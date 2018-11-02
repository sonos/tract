macro_rules! with_T {
    ($op:ty) => {
        |pb: &NodeDef| -> $crate::tract_core::TfdResult<Box<$crate::tract_core::ops::Op>> {
            let datum_type: $crate::tract_core::ops::prelude::TypeFact =
                pb.get_attr_datum_type("T")?.into();
            Ok(Box::new(<$op>::new(datum_type)) as _)
        }
    };
}
