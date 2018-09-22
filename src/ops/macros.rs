macro_rules! element_map {
    ($Name:ident, [$($type:ty),*], $expr:expr) => {
        element_map!($Name, match $($type => $expr),*);
    };
    ($Name:ident, match $($type:ty => $expr:expr),*) => {
        #[derive(Debug, Clone, new, Default)]
        pub struct $Name($crate::analyser::TypeFact);

        impl ::ops::Op for $Name {
            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                mut inputs: $crate::TVec<$crate::ops::Value>,
            ) -> $crate::TfdResult<$crate::TVec<$crate::ops::Value>> {
                use $crate::tensor::Datum;
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $(if dt == <$type>::datum_type() {
                    let mut a = a.into_array::<$type>()?;
                    a.mapv_inplace($expr);
                    return Ok(tvec![a.into()])
                })*
                bail!("{} not covering {:?}", stringify!($Name), dt)
            }

            /// Evaluates one step of the operation on the given input tensors.
            fn step(
                &self,
                mut inputs: $crate::TVec<$crate::ops::StepValue>,
                _buffer: &mut Box<$crate::ops::OpBuffer>,
            ) -> $crate::TfdResult<Option<$crate::TVec<$crate::ops::Value>>> {
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

macro_rules! element_bin {
    ($Name:ident, [$($type:ty),*] => $to:ty { $expr:expr }) => {
        element_bin!($Name, match $($type => $to { $expr } ),*);
    };
    ($Name:ident, [$($type:ty),*] { $expr:expr }) => {
        element_bin!($Name, match $($type => $type { $expr } ),*);
    };
    ($Name:ident, match $($type:ty => $to:ty { $expr:expr }),*) => {
        #[derive(Debug, Clone, Default, new)]
        pub struct $Name($crate::analyser::TypeFact);

        impl Op for $Name {

            /// Evaluates the operation given the input tensors.
            fn eval(
                &self,
                mut inputs: TVec<$crate::ops::Value>,
            ) -> $crate::TfdResult<TVec<$crate::ops::Value>> {
                use $crate::tensor::Datum;
                let (a, b) = args_2!(inputs);
                if let Some(dt) = a.datum_type().common_super_type(b.datum_type()) {
                    $(if dt == <$type>::datum_type() {
                        let a = a.cast_to_array::<$type>()?.into_owned();
                        let b = b.cast_to_array::<$type>()?;
                        let shape = $crate::broadcast::multi_broadcast(&[a.shape(), b.view().shape()])
                            .ok_or_else(|| format!("Incompatible shapes {:?} and{:?}",
                                                   a.shape(), b.view().shape()))?;
                        let mut c = $crate::ndarray::ArrayD::<$to>::default(shape);
                        $crate::ndarray::Zip::from(&mut c)
                            .and_broadcast(&a)
                            .and_broadcast(&b.view())
                            .apply(|c,&a:&$type,&b:&$type| *c = $expr(a,b));
                        return Ok(tvec![c.into()])

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
            ) -> $crate::TfdResult<Option<TVec<$crate::ops::Value>>> {
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

#[macro_export]
macro_rules! args_1 {
    ($inputs:expr) => {{
        if $inputs.len() != 1 {
            Err("Expected 1 arg")?
        }
        $inputs.pop().unwrap()
    }};
}

#[macro_export]
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
#[macro_export]
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

#[allow(unused_macros)]
#[macro_export]
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

#[allow(unused_macros)]
#[macro_export]
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
#[macro_export]
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
#[macro_export]
macro_rules! assert_backward {
    ($op:expr, $input:ident, $output:ident) => {
        assert_eq!(
            $op.infer(tvec![TensorFact::new()], tvec![$output.clone()])
                .unwrap(),
            (tvec![$input], tvec![$output.clone()])
        )
    };
}
