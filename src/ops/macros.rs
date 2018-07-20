macro_rules! element_map {
    ($Struct:ident, $expr:expr) => {
        #[derive(Debug, Clone)]
        pub struct $Struct;

        impl $Struct {
            pub fn build(_pb: &::tfpb::node_def::NodeDef) -> $crate::Result<Box<Op>> {
                Ok(Box::new($Struct))
            }
        }

        impl ::ops::Op for $Struct {
            /// Returns the attributes of the operation and their values.
            fn get_attributes(&self) -> ::std::collections::HashMap<&'static str, ::ops::Attr> {
                hashmap!{}
            }

            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                mut inputs: Vec<$crate::ops::TensorView>,
            ) -> $crate::Result<Vec<$crate::ops::TensorView>> {
                let a = args_1!(inputs);
                let mut a = a.into_tensor()
                    .take_f32s()
                    .ok_or("Expect input #0 to be f32")?;
                a.mapv_inplace($expr);
                Ok(vec![$crate::tensor::Tensor::F32(a).into()])
            }

            /// Evaluates one step of the operation on the given input tensors.
            fn step(
                &self,
                mut inputs: Vec<(Option<usize>, Option<$crate::ops::TensorView>)>,
                _buffer: &mut Box<$crate::ops::OpBuffer>,
            ) -> Result<Option<Vec<$crate::ops::TensorView>>> {
                let a = args_1!(inputs);
                match a.1 {
                    None => Ok(None),
                    Some(tv) => Ok(Some(self.eval(vec![tv])?))
                }
            }
        }

        impl ::ops::InferenceRulesOp for $Struct {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r>(
                &self,
                solver: &mut $crate::analyser::interface::Solver<'r>,
                inputs: &'p $crate::analyser::interface::TensorsProxy,
                outputs: &'p $crate::analyser::interface::TensorsProxy,
            ) {
                solver
                    .equals(&inputs.len, 1)
                    .equals(&outputs.len, 1)
                    .equals(&inputs[0].datatype, &outputs[0].datatype)
                    .equals(&inputs[0].shape, &outputs[0].shape);
            }
        }
    };
}

macro_rules! element_bin {
    ($Name:ident, $name:ident, $expr:expr) => {
        #[derive(Debug, Clone, new)]
        pub struct $Name<T: ::tensor::Datum>(::std::marker::PhantomData<T>);

        pub fn $name(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
            let dtype = pb.get_attr_datatype("T")?;
            Ok(boxed_new!($Name(dtype)()))
        }

        impl<T: ::tensor::Datum> Op for $Name<T> {
            /// Returns the attributes of the operation and their values.
            fn get_attributes(&self) -> ::std::collections::HashMap<&'static str, ::ops::Attr> {
                hashmap!{ "T" => ::ops::Attr::DataType(T::datatype()) }
            }

            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                mut inputs: Vec<$crate::ops::TensorView>,
            ) -> Result<Vec<$crate::ops::TensorView>> {
                let (a, b) = args_2!(inputs);
                let a = T::tensor_into_array(a.into_tensor())?;
                let b = T::tensor_to_view(&*b)?;
                Ok(vec![T::array_into_tensor($expr(a, b)).into()])
            }

            /// Returns a new streaming buffer for the operation.
            fn new_buffer(&self) -> Box<$crate::ops::OpBuffer> {
                Box::new($crate::ops::QueuesBuffer::new(2))
            }

            /// Evaluates one step of the operation on the given input tensors.
            fn step(
                &self,
                mut inputs: Vec<(Option<usize>, Option<$crate::ops::TensorView>)>,
                buffer: &mut Box<$crate::ops::OpBuffer>,
            ) -> Result<Option<Vec<$crate::ops::TensorView>>> {
                let buffer = buffer.downcast_mut::<$crate::ops::QueuesBuffer>()
                    .ok_or("The buffer can't be downcasted to QueuesBuffer.")?;

                // If we don't have a value for some of the inputs yet, we buffer
                // the current values to reuse them on the next call.
                buffer.append(&mut inputs)?;

                if buffer[0].is_empty() || buffer[1].is_empty() {
                    Ok(None)
                } else {
                    let a = buffer[0].pop_front().unwrap();
                    let b = buffer[1].pop_front().unwrap();
                    Ok(Some(self.eval(vec![a, b])?))
                }
            }
        }

        impl<T: ::tensor::Datum> ::ops::InferenceRulesOp for $Name<T> {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r>(
                &self,
                solver: &mut $crate::analyser::interface::Solver<'r>,
                inputs: &'p $crate::analyser::interface::TensorsProxy,
                outputs: &'p $crate::analyser::interface::TensorsProxy,
            ) {
                let a = &inputs[0];
                let b = &inputs[1];
                let c = &outputs[0];

                solver
                    .equals(&outputs.len, 1)
                    .equals_all(wrap![&a.datatype, &b.datatype, &c.datatype])
                    .equals_all(wrap![&a.shape, &b.shape, &c.shape]);
            }
        }
    };
}

macro_rules! args_1 {
    ($inputs:expr) => {{
        if $inputs.len() != 1 {
            Err("Expected 1 arg")?
        }
        $inputs.pop().unwrap()
    }};
}

macro_rules! args_2 {
    ($inputs:expr) => {{
        if $inputs.len() != 2 {
            Err("Expected 2 args")?
        }
        $inputs.reverse();
        ($inputs.pop().unwrap(), $inputs.pop().unwrap())
    }};
}

#[allow(unused_macros)]
macro_rules! args_3 {
    ($inputs:expr) => {{
        if $inputs.len() != 3 {
            Err("Expected 3 args")?
        }
        $inputs.reverse();
        (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        )
    }};
}

macro_rules! args_4 {
    ($inputs:expr) => {{
        if $inputs.len() != 4 {
            Err("Expected 4 args")?
        }
        $inputs.reverse();
        (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        )
    }};
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

/// Asserts that forward inference results work as expected.
#[allow(unused_macros)]
macro_rules! assert_forward {
    ($op:expr, $input:ident, $output:ident) => (
        assert_eq!(
            $op.infer(
                vec![$input.clone()],
                vec![TensorFact::new()],
            ).unwrap(), (
                vec![$input.clone()],
                vec![$output],
            )
        )
    );
}

/// Asserts that backward inference results work as expected.
#[allow(unused_macros)]
macro_rules! assert_backward {
    ($op:expr, $input:ident, $output:ident) => (
        assert_eq!(
            $op.infer(
                vec![TensorFact::new()],
                vec![$output.clone()],
            ).unwrap(), (
                vec![$input],
                vec![$output.clone()],
            )
        )
    );
}
