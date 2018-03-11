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
            fn eval(&self, mut inputs: Vec<$crate::ops::Input>) -> $crate::Result<Vec<$crate::ops::Input>> {
                let a = args_1!(inputs);
                let mut a = a.into_matrix().take_f32s().ok_or(
                    "Expect input #0 to be f32",
                )?;
                a.mapv_inplace($expr);
                Ok(vec![$crate::matrix::Matrix::F32(a).into()])
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
            fn eval(&self, mut inputs: Vec<$crate::ops::Input>) -> Result<Vec<$crate::ops::Input>> {
                let (a, b) = args_2!(inputs);
                let a = a.into_matrix().take_f32s().ok_or(
                    "Expect input #0 to be f32",
                )?;
                let b = b.as_f32s().ok_or(
                    "Expect input #1 to be f32",
                )?;
                Ok(vec!(Matrix::from($expr(a,b.view())).into()))
            }
        }
    }
}

macro_rules! args_1 {
    ($inputs:expr) => { {
        if $inputs.len() != 1 {
            Err("Expected 1 arg")?
        }
        $inputs.pop().unwrap()
    } }
}

macro_rules! args_2 {
    ($inputs:expr) => { {
        if $inputs.len() != 2 {
            Err("Expected 2 args")?
        }
        $inputs.reverse();
        ($inputs.pop().unwrap(), $inputs.pop().unwrap())
    } }
}
