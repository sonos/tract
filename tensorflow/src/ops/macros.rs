macro_rules! with_T {
    ($op:ty) => {
        |pb: &NodeDef| -> $crate::tract::TfdResult<Box<$crate::tract::ops::Op>> {
            let datum_type: $crate::tract::ops::prelude::TypeFact =
                pb.get_attr_datum_type("T")?.into();
            Ok(Box::new(<$op>::new(datum_type)) as _)
        }
    };
}
