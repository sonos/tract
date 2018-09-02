macro_rules! element_map_float {
    ($Name:ident, $name:ident, $expr:expr) => {
        pub fn $name(pb: &$crate::tfpb::node_def::NodeDef) -> $crate::Result<Box<Op>> {
            let datum_type = pb.get_attr_datum_type("T")?;
            let it = match datum_type {
                $crate::DatumType::F32 => Box::new($Name::<f32>::new()) as Box<Op>,
                $crate::DatumType::F64 => Box::new($Name::<f64>::new()) as Box<Op>,
                _ => unimplemented!("missing type"),
            };
            Ok(it)
        }

        #[derive(Debug, Clone, new)]
        pub struct $Name<T: $crate::tensor::Datum + ::num::Float>(::std::marker::PhantomData<T>);

        impl<T: $crate::tensor::Datum + ::num::Float> ::ops::Op for $Name<T> {
            /// Returns the attributes of the operation and their values.
            fn get_attributes(&self) -> ::std::collections::HashMap<&'static str, ::ops::Attr> {
                hashmap!{ "T" => $crate::ops::Attr::DatumType(T::datum_type()) }
            }

            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                mut inputs: TVec<$crate::ops::Value>,
            ) -> $crate::Result<TVec<$crate::ops::Value>> {
                let a = args_1!(inputs);
                let mut a = a.into_array::<T>()?;
                a.mapv_inplace($expr);
                Ok(tvec![a.into()])
            }

            /// Evaluates one step of the operation on the given input tensors.
            fn step(
                &self,
                mut inputs: TVec<$crate::ops::StepValue>,
                _buffer: &mut Box<$crate::ops::OpBuffer>,
            ) -> Result<Option<TVec<$crate::ops::Value>>> {
                let a = args_1!(inputs);
                match a.into_value() {
                    None => Ok(None),
                    Some(tv) => Ok(Some(self.eval(tvec![tv])?)),
                }
            }
        }

        impl<T: $crate::tensor::Datum + ::num::Float> $crate::analyser::rules::InferenceRulesOp for $Name<T> {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                solver: &mut $crate::analyser::rules::prelude::Solver<'r>,
                inputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
                outputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
            ) {
                solver
                    .equals(&inputs.len, 1)
                    .equals(&outputs.len, 1)
                    .equals_all(wrap![
                        &inputs[0].datum_type,
                        &outputs[0].datum_type,
                        &T::datum_type()
                    ])
                    .equals(&inputs[0].shape, &outputs[0].shape);
            }
        }
    };
}

macro_rules! element_map {
    ($Name:ident, $name:ident, [$($type:tt),*], $expr:expr) => {
        pub fn $name(pb: &$crate::tfpb::node_def::NodeDef) -> $crate::Result<Box<$crate::ops::Op>> {
            let datum_type = pb.get_attr_datum_type("T")?;
            Ok(Box::new($Name::new(datum_type)) as _)
        }

        #[derive(Debug, Clone, new)]
        pub struct $Name($crate::tensor::DatumType);

        impl ::ops::Op for $Name {
            /// Returns the attributes of the operation and their values.
            fn get_attributes(&self) -> ::std::collections::HashMap<&'static str, ::ops::Attr> {
                hashmap!{ "T" => $crate::ops::Attr::DatumType(self.0) }
            }

            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                mut inputs: TVec<$crate::ops::Value>,
            ) -> $crate::Result<TVec<$crate::ops::Value>> {
                use $crate::tensor::Datum;
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $(if dt == $type::datum_type() {
                    let mut a = a.into_array::<$type>()?;
                    a.mapv_inplace($expr);
                    return Ok(tvec![a.into()])
                })*
                bail!("{} not covering {:?}", stringify!($Name), dt)
            }

            /// Evaluates one step of the operation on the given input tensors.
            fn step(
                &self,
                mut inputs: TVec<$crate::ops::StepValue>,
                _buffer: &mut Box<$crate::ops::OpBuffer>,
            ) -> $crate::Result<Option<TVec<$crate::ops::Value>>> {
                let a = args_1!(inputs);
                match a.into_value() {
                    None => Ok(None),
                    Some(tv) => Ok(Some(self.eval(tvec![tv])?)),
                }
            }
        }

        impl $crate::analyser::rules::InferenceRulesOp for $Name {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                solver: &mut $crate::analyser::rules::prelude::Solver<'r>,
                inputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
                outputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
            ) {
                solver
                    .equals(&inputs.len, 1)
                    .equals(&outputs.len, 1)
                    .equals_all(wrap![
                        &inputs[0].datum_type,
                        &outputs[0].datum_type,
                    ])
                    .equals(&inputs[0].shape, &outputs[0].shape);
            }
        }
    };
}

macro_rules! element_map_signed {
    ($Name:ident, $name:ident, $expr:expr) => {
        pub fn $name(pb: &$crate::tfpb::node_def::NodeDef) -> $crate::Result<Box<Op>> {
            let datum_type = pb.get_attr_datum_type("T")?;
            let it = match datum_type {
                $crate::DatumType::I32 => Box::new($Name::<i32>::new()) as Box<Op>,
                $crate::DatumType::F32 => Box::new($Name::<f32>::new()) as Box<Op>,
                $crate::DatumType::F64 => Box::new($Name::<f64>::new()) as Box<Op>,
                _ => unimplemented!("missing type"),
            };
            Ok(it)
        }

        #[derive(Debug, Clone, new)]
        pub struct $Name<T: $crate::tensor::Datum + ::num::Signed>(::std::marker::PhantomData<T>);

        impl<T: $crate::tensor::Datum + ::num::Signed> ::ops::Op for $Name<T> {
            /// Returns the attributes of the operation and their values.
            fn get_attributes(&self) -> ::std::collections::HashMap<&'static str, ::ops::Attr> {
                hashmap!{ "T" => $crate::ops::Attr::DatumType(T::datum_type()) }
            }

            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                mut inputs: TVec<$crate::ops::Value>,
            ) -> $crate::Result<TVec<$crate::ops::Value>> {
                let a = args_1!(inputs);
                let mut a = a.into_array::<T>()?;
                a.mapv_inplace($expr);
                Ok(tvec![a.into()])
            }

            /// Evaluates one step of the operation on the given input tensors.
            fn step(
                &self,
                mut inputs: TVec<$crate::ops::StepValue>,
                _buffer: &mut Box<$crate::ops::OpBuffer>,
            ) -> Result<Option<TVec<$crate::ops::Value>>> {
                let a = args_1!(inputs);
                match a.into_value() {
                    None => Ok(None),
                    Some(tv) => Ok(Some(self.eval(tvec![tv])?)),
                }
            }
        }

        impl<T: $crate::tensor::Datum + ::num::Signed> $crate::analyser::rules::InferenceRulesOp for $Name<T> {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                solver: &mut $crate::analyser::rules::prelude::Solver<'r>,
                inputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
                outputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
            ) {
                solver
                    .equals(&inputs.len, 1)
                    .equals(&outputs.len, 1)
                    .equals_all(wrap![
                        &inputs[0].datum_type,
                        &outputs[0].datum_type,
                        &T::datum_type()
                    ])
                    .equals(&inputs[0].shape, &outputs[0].shape);
            }
        }
    };
}

macro_rules! element_bin {
    ($Name:ident, $name:ident, [$($type:tt),*], $expr:expr) => {
        #[derive(Debug, Clone, new)]
        pub struct $Name($crate::tensor::DatumType);

        pub fn $name(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
            let dtype = pb.get_attr_datum_type("T")?;
            Ok(Box::new($Name::new(dtype)))
        }

        impl $Name {
            /// Evaluates the operation given the input tensors.
            fn eval_t<T: ::tensor::Datum>(
                &self,
                mut inputs: TVec<$crate::ops::Value>,
            ) -> Result<TVec<$crate::ops::Value>> {
                let (a, b) = args_2!(inputs);
                let a = a.cast_to_array::<T>()?.into_owned();
                let b = b.cast_to_array::<T>()?;
                Ok(tvec![$expr(a, b.view()).into()])
            }
        }

        impl Op for $Name {
            /// Returns the attributes of the operation and their values.
            fn get_attributes(&self) -> ::std::collections::HashMap<&'static str, ::ops::Attr> {
                hashmap!{ "T" => $crate::ops::Attr::DatumType(self.0) }
            }

            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                inputs: TVec<$crate::ops::Value>,
            ) -> Result<TVec<$crate::ops::Value>> {
                use $crate::tensor::Datum;
                if let Some(dt) = inputs[0].datum_type().common_super_type(inputs[1].datum_type()) {
                    $(if dt == $type::datum_type() {
                        return self.eval_t::<$type>(inputs);
                    })*
                    bail!("{} not covering {:?}", stringify!($Name), dt)
                } else {
                    bail!("Could not find a supertype accomodating {:?} and {:?}",
                          inputs[0].datum_type(), inputs[1].datum_type());
                }
            }

            /// Returns a new streaming buffer for the operation.
            fn new_buffer(&self) -> Box<$crate::ops::OpBuffer> {
                Box::new($crate::ops::QueuesBuffer::new(2))
            }

            /// Evaluates one step of the operation on the given input tensors.
            fn step(
                &self,
                inputs: TVec<$crate::ops::StepValue>,
                buffer: &mut Box<$crate::ops::OpBuffer>,
            ) -> Result<Option<TVec<$crate::ops::Value>>> {
                let buffer = buffer.downcast_mut::<$crate::ops::QueuesBuffer>()
                    .ok_or("The buffer can't be downcasted to QueuesBuffer.")?;

                // If we don't have a value for some of the inputs yet, we buffer
                // the current values to reuse them on the next call.
                buffer.append(inputs)?;

                if buffer[0].is_empty() || buffer[1].is_empty() {
                    Ok(None)
                } else {
                    let a = buffer[0].pop_front().unwrap();
                    let b = buffer[1].pop_front().unwrap();
                    Ok(Some(self.eval(tvec![a, b])?))
                }
            }
        }

        impl $crate::analyser::rules::InferenceRulesOp for $Name {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                solver: &mut $crate::analyser::rules::prelude::Solver<'r>,
                inputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
                outputs: &'p $crate::analyser::rules::prelude::TensorsProxy,
            ) {
                let a = &inputs[0];
                let b = &inputs[1];
                let c = &outputs[0];

                solver.given(&inputs[0].datum_type, move |solver, dta| {
                    solver.given(&inputs[1].datum_type, move |solver, dtb| {
                        if let Some(dt) = dta.common_super_type(dtb) {
                            solver.equals(&outputs[0].datum_type, dt);
                        }
                    });
                });
                solver
                    .equals(&outputs.len, 1)
                    .with(&a.shape, move |solver, a_shape| {
                        solver.with(&b.shape, move |solver, b_shape| {
                            if let Ok(Some(c_shape)) = ::analyser::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape]) {
                                solver.equals(&c.shape, c_shape);
                            }
                        });
                    });
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
        use $crate::DatumType;
        match $dtype {
            DatumType::I32 => Box::new($op::<i32>::new($($arg),*)) as Box<Op>,
            DatumType::F32 => Box::new($op::<f32>::new($($arg),*)) as Box<Op>,
            DatumType::F64 => Box::new($op::<f64>::new($($arg),*)) as Box<Op>,
            _ => unimplemented!("missing type")
        }
    } }
}

/// Asserts that forward inference results work as expected.
#[allow(unused_macros)]
macro_rules! assert_forward {
    ($op:expr, $input:ident, $output:ident) => {
        assert_eq!(
            $op.infer(tvec![$input.clone()], tvec![TensorFact::new()])
                .unwrap(),
            (tvec![$input.clone()], tvec![$output])
        )
    };
}

/// Asserts that backward inference results work as expected.
#[allow(unused_macros)]
macro_rules! assert_backward {
    ($op:expr, $input:ident, $output:ident) => {
        assert_eq!(
            $op.infer(tvec![TensorFact::new()], tvec![$output.clone()])
                .unwrap(),
            (tvec![$input], tvec![$output.clone()])
        )
    };
}
