use matrix::Matrix;

macro_rules! element_map {
    ($Struct:ident, $expr:expr) => {
        #[derive(Debug)]
        pub struct $Struct {}

        impl $Struct {
            pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<$Struct> {
                Ok($Struct {})
            }
        }

        impl ::ops::Op for $Struct {
            fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
                let mut input = inputs.remove(0).take_f32s().ok_or(
                    "Expect input #0 to be f32",
                )?;
                input.mapv_inplace($expr);
                Ok(vec![Matrix::F32(input)])
            }
        }
    }
}
