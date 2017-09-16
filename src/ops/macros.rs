macro_rules! element_map {
    ($Struct:ident, $expr:expr) => {
        #[derive(Debug)]
        pub struct $Struct;

        impl $Struct {
            pub fn build(_pb: &::tfpb::node_def::NodeDef) -> $crate::Result<Box<Op>> {
                Ok(Box::new($Struct))
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

macro_rules! element_bin {
    ($Name:ident, $expr: expr) =>
    {
        #[derive(Debug)]
        pub struct $Name;

        impl $Name {
            pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
                Ok(Box::new($Name))
            }
        }

        impl Op for $Name {
            fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
                let input1 = inputs.remove(0).take_f32s().ok_or(
                    "Expect input #0 to be f32",
                )?;
                let input2 = inputs.remove(0).take_f32s().ok_or(
                    "Expect input #1 to be f32",
                )?;
                Ok(vec!($expr(input1, input2).into()))
            }
        }
    }
}
