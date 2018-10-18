macro_rules! with_T {
    ($op:ty) => {
        |pb: &NodeDef| -> $crate::tfdeploy::TfdResult<Box<$crate::tfdeploy::ops::Op>> {
            let datum_type: $crate::tfdeploy::analyser::TypeFact =
                pb.get_attr_datum_type("T")?.into();
            Ok(Box::new(<$op>::new(datum_type)) as _)
        }
    };
}
