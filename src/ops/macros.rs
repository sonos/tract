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
            /// Evaluates the operation given the input tensors.
            fn eval(&self, mut inputs: Vec<$crate::ops::TensorView>) -> $crate::Result<Vec<$crate::ops::TensorView>> {
                let a = args_1!(inputs);
                let mut a = a.into_tensor().take_f32s().ok_or(
                    "Expect input #0 to be f32",
                )?;
                a.mapv_inplace($expr);
                Ok(vec![$crate::tensor::Tensor::F32(a).into()])
            }

            /// Infers properties about the output tensors from the input tensors.
            fn infer_forward(&self, inputs: Vec<&$crate::analyser::TensorFact>) -> Result<Option<Vec<$crate::analyser::TensorFact>>> {
                if inputs.len() != 1 {
                    bail!("Unary operations only supports one input.");
                }

                $crate::analyser::helpers::infer_forward_basic(self, inputs)
            }

            /// Infers properties about the input tensors from the output tensors.
            fn infer_backward(&self, outputs: Vec<&$crate::analyser::TensorFact>) -> Result<Option<Vec<$crate::analyser::TensorFact>>> {
                if outputs.len() < 1 {
                    bail!("Unary operations need at least one output.");
                }

                let input = $crate::analyser::TensorFact {
                    datatype: outputs[0].datatype,
                    shape: outputs[0].shape.clone(),
                    value: valuefact!(_)
                };

                Ok(Some(vec![input]))
            }
        }
    }
}

macro_rules! element_bin {
    ($Name:ident, $name:ident, $expr: expr) =>
    {
        #[derive(Debug,new)]
        pub struct $Name<T: ::tensor::Datum>(::std::marker::PhantomData<T>);

        pub fn $name(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
            let dtype = pb.get_attr_datatype("T")?;
            Ok(boxed_new!($Name(dtype)()))
        }

        impl<T: ::tensor::Datum> Op for $Name<T> {
            /// Evaluates the operation given the input tensors.
            fn eval(&self, mut inputs: Vec<$crate::ops::TensorView>) -> Result<Vec<$crate::ops::TensorView>> {
                let (a, b) = args_2!(inputs);
                let a = T::mat_into_array(a.into_tensor())?;
                let b = T::mat_to_view(&*b)?;
                Ok(vec!(T::array_into_tensor($expr(a,b)).into()))
            }

            /// Infers properties about the output tensors from the input tensors.
            fn infer_forward(&self, inputs: Vec<&$crate::analyser::TensorFact>) -> Result<Option<Vec<$crate::analyser::TensorFact>>> {
                use $crate::analyser::TypeFact::*;

                if inputs.len() != 2 {
                    bail!("Binary operations only supports two inputs.");
                }

                if let (Only(i), Only(j)) = (inputs[0].datatype, inputs[1].datatype) {
                    if i != j {
                        bail!("Binary operations don't support inputs of different types.");
                    }
                }

                $crate::analyser::helpers::infer_forward_basic(self, inputs)
            }

            /// Infers properties about the input tensors from the output tensors.
            fn infer_backward(&self, outputs: Vec<&$crate::analyser::TensorFact>) -> Result<Option<Vec<$crate::analyser::TensorFact>>> {
                if outputs.len() < 1 {
                    bail!("Binary operations need at least one output.");
                }

                let input = $crate::analyser::TensorFact {
                    datatype: outputs[0].datatype,
                    shape: shapefact![..],
                    value: valuefact!(_)
                };

                Ok(Some(vec![input.clone(), input]))
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

#[allow(unused_macros)]
macro_rules! args_3 {
    ($inputs:expr) => { {
        if $inputs.len() != 3 {
            Err("Expected 3 args")?
        }
        $inputs.reverse();
        ($inputs.pop().unwrap(), $inputs.pop().unwrap(), $inputs.pop().unwrap())
    } }
}

macro_rules! args_4 {
    ($inputs:expr) => { {
        if $inputs.len() != 4 {
            Err("Expected 4 args")?
        }
        $inputs.reverse();
        ($inputs.pop().unwrap(), $inputs.pop().unwrap(),
        $inputs.pop().unwrap(), $inputs.pop().unwrap())
    } }
}

macro_rules! boxed_new {
    ($op:tt($dtype:expr)($($arg:expr),*)) => { {
        use tfpb::types::DataType;
        match $dtype {
            DataType::DT_INT32 => Box::new($op::<i32>::new($($arg),*)) as Box<Op>,
            DataType::DT_FLOAT => Box::new($op::<f32>::new($($arg),*)) as Box<Op>,
            DataType::DT_DOUBLE => Box::new($op::<f64>::new($($arg),*)) as Box<Op>,
            _ => unimplemented!()
        }
    } }
}
