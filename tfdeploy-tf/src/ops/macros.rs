
macro_rules! with_T {
    ($op:ty) => { |pb:&NodeDef| -> $crate::tfdeploy::Result<Box<$crate::tfdeploy::ops::Op>> {
        let datum_type = pb.get_attr_datum_type("T")?;
        Ok(Box::new(<$op>::new(datum_type)) as _)
    } }
}

