macro_rules! element_map {
    ($Struct:ident, $expr:expr) => {
        #[derive(Debug)]
        pub struct $Struct {}

        impl $Struct {
            pub fn build(_pb: &::tfpb::node_def::NodeDef) -> $crate::Result<$Struct> {
                Ok($Struct {})
            }
        }

        impl ::ops::Op for $Struct {
            fn eval(&self, mut inputs: Vec<$crate::matrix::Matrix>) -> $crate::Result<Vec<$crate::matrix::Matrix>> {
                let mut input = inputs.remove(0).take_f32s().ok_or(
                    "Expect input #0 to be f32",
                )?;
                input.mapv_inplace($expr);
                Ok(vec![$crate::matrix::Matrix::F32(input)])
            }
        }
    }
}
