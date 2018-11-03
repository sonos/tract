macro_rules! element_map {
    ($Name:ident, [$($type:ty),*], $expr:expr) => {
        element_map!($Name, match $($type => $expr),*);
    };
    ($Name:ident, match $($type:ty => $expr:expr),*) => {
        #[allow(unused_imports)]
        use $crate::ops::prelude::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name(TypeFact);

        impl StatelessOp for $Name {
            fn eval(&self, mut inputs: TVec<Value>,) -> TractResult<TVec<Value>> {
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $(if dt == <$type>::datum_type() {
                    let mut a = a.into_array::<$type>()?;
                    a.mapv_inplace($expr);
                    return Ok(tvec![a.into()])
                })*
                bail!("{} not covering {:?}", stringify!($Name), dt)
            }
        }

        impl Op for $Name {
            fn name(&self) -> &str {
                stringify!($Name)
            }

            fn pulsify( &self, inputs: TVec<&PulsedTensorFact>,) -> TractResult<Vec<PulsifiedOp>> {
                Ok(vec!(PulsifiedOp::new(Box::new(self.clone()), tvec!(inputs[0].clone()))))
            }

        }

        impl InferenceRulesOp for $Name {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                s: &mut Solver<'r>,
                inputs: &'p TensorsProxy,
                outputs: &'p TensorsProxy,
            ) -> InferenceResult {
                s.equals(&inputs.len, 1)?;
                s.equals(&outputs.len, 1)?;
                s.equals_all(wrap![
                    &inputs[0].datum_type,
                    &outputs[0].datum_type,
                ])?;
                s.equals(&inputs[0].shape, &outputs[0].shape)
            }
        }
    };
}

macro_rules! element_map_with_params {
    ($Name:ident, [$($type:ty),*], {$($pname:ident : $pty:ty),*}, $expr:item) => {
        #[allow(unused_imports)]
        use $crate::ops::prelude::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name {
            $( $pname: $pty ),*
        }

        impl StatelessOp for $Name {
            fn eval(&self, mut inputs: TVec<Value>,) -> TractResult<TVec<Value>> {
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $expr;
                $(if dt == <$type>::datum_type() {
                    let mut a = a.into_array::<$type>()?;
                    a.mapv_inplace(|x| eval_one(self,x));
                    return Ok(tvec![a.into()])
                })*
                bail!("{} not covering {:?}", stringify!($Name), dt)
            }
        }

        impl Op for $Name {
            fn name(&self) -> &str {
                stringify!($Name)
            }
        }

        impl InferenceRulesOp for $Name {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                s: &mut Solver<'r>,
                inputs: &'p TensorsProxy,
                outputs: &'p TensorsProxy,
            ) -> InferenceResult {
                s.equals(&inputs.len, 1)?;
                s.equals(&outputs.len, 1)?;
                s.equals_all(wrap![
                    &inputs[0].datum_type,
                    &outputs[0].datum_type,
                ])?;
                s.equals(&inputs[0].shape, &outputs[0].shape)
            }
        }
    };
}

macro_rules! element_bin {
    ($name:ident, [$($type:ty),*] => $to:ty { $expr:expr }) => {
        element_bin!($name, match $($type => $to { $expr } ),*);
    };
    ($name:ident, [$($type:ty),*] { $expr:expr }) => {
        element_bin!($name, match $($type => $type { $expr } ),*);
    };
    ($name:ident, match $($type:ty => $to:ty { $expr:expr }),*) => {
        #[allow(non_snake_case)]
        pub mod $name {
            #[allow(unused_imports)]
            use $crate::ops::prelude::*;
            #[allow(unused_imports)]
            use num::Float;

            pub fn default() -> Bin {
                Bin::default()
            }

            fn eval_bin(a: Value, b: &Value) -> TractResult<Value> {
                let shape:TVec<usize> = $crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])
                    .ok_or_else(|| format!("Incompatible shapes {:?} and{:?}",
                                           a.shape(), b.shape()))?;
                let dt = a.datum_type().common_super_type(b.datum_type())
                    .ok_or_else(|| format!("Incompatible types {:?} and{:?}",
                                           a.datum_type(), b.datum_type()))?;
                $(if dt == <$type>::datum_type() {
                    let a = a.cast_to_array::<$type>()?.into_owned();
                    let b = b.cast_to_array::<$type>()?;
                    let mut c = $crate::ndarray::ArrayD::<$to>::default(&*shape);
                    $crate::ndarray::Zip::from(&mut c)
                        .and_broadcast(&a)
                        .and_broadcast(&b.view())
                        .apply(|c,&a:&$type,&b:&$type| *c = $expr(a,b));
                    return Ok(c.into())
                })*
                bail!("{} not covering {:?}", stringify!($name), dt)
            }

            #[derive(Debug, Clone, Default, new)]
            pub struct Bin(TypeFact);

            impl StatelessOp for Bin {
                fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
                    let (a, b) = args_2!(inputs);
                    Ok(tvec!(eval_bin(a, &b)?))
                }

            }

            impl Op for Bin {
                fn name(&self) -> &str {
                    stringify!($name)
                }

                fn reduce(&self, inputs: TVec<&TensorFact>, _outputs: TVec<&TensorFact>,
                ) -> TractResult<Option<ReducedOpRewire>> {
                    if let Some(b) = inputs[1].value.concretize() {
                        return Ok(Some(ReducedOpRewire {
                            new_op: Box::new(UnaryA {
                                dt: self.0,
                                b: b.into(),
                            }),
                            rewired: tvec!(0)
                        }))
                    }
                    Ok(None)
                }

                fn pulsify(&self, inputs: TVec<&PulsedTensorFact>,) -> TractResult<Vec<PulsifiedOp>> {
                    let mut fact = inputs[0].clone();
                    $(if fact.dt == <$type>::datum_type() {
                        fact.dt = <$to>::datum_type().into();
                    })*
                    Ok(vec!(PulsifiedOp::new(Box::new(self.clone()), tvec!(fact))))
                }
            }

            impl InferenceRulesOp for Bin {
                /// Infers properties about the input and output tensors.
                fn rules<'r, 'p: 'r, 's: 'r>(
                    &'s self,
                    s: &mut Solver<'r>,
                    inputs: &'p TensorsProxy,
                    outputs: &'p TensorsProxy,
                ) -> InferenceResult {
                    let a = &inputs[0];
                    let b = &inputs[1];
                    let c = &outputs[0];

                    s.given(&inputs[0].datum_type, move |s, dt| {
                        $(if dt == <$type>::datum_type() {
                            return s.equals(&outputs[0].datum_type, <$to>::datum_type());
                        })*
                        bail!("{} not covering {:?}", stringify!($name), dt)
                    })?;
                    s.equals(&inputs.len, 2)?;
                    s.equals(&outputs.len, 1)?;
                    s.with(&a.shape, move |s, a_shape| {
                        s.with(&b.shape, move |s, b_shape| {
                            if let Ok(Some(c_shape)) = $crate::analyser::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape]) {
                                s.equals(&c.shape, c_shape)?;
                            }
                            Ok(())
                        })
                    })
                }
            }

            #[derive(Debug, Clone, new)]
            pub struct UnaryA {
                dt: TypeFact,
                b: Value,
            }

            impl StatelessOp for UnaryA {
                fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
                    let a = args_1!(inputs);
                    Ok(tvec!(eval_bin(a, &self.b)?))
                }
            }

            impl Op for UnaryA {
                fn name(&self) -> &str {
                    stringify!($name)
                }

                fn pulsify(&self, inputs: TVec<&PulsedTensorFact>,) -> TractResult<Vec<PulsifiedOp>> {
                    let mut fact = inputs[0].clone();
                    $(if fact.dt == <$type>::datum_type() {
                        fact.dt = <$to>::datum_type().into();
                    })*
                    Ok(vec!(PulsifiedOp::new(Box::new(self.clone()), tvec!(fact))))
                }
            }

            impl InferenceRulesOp for UnaryA {
                /// Infers properties about the input and output tensors.
                fn rules<'r, 'p: 'r, 's: 'r>(
                    &'s self,
                    s: &mut Solver<'r>,
                    inputs: &'p TensorsProxy,
                    outputs: &'p TensorsProxy,
                ) -> InferenceResult {
                    let a = &inputs[0];
                    let c = &outputs[0];

                    s.given(&inputs[0].datum_type, move |s, dt| {
                        $(if dt == <$type>::datum_type() {
                            return s.equals(&outputs[0].datum_type, <$to>::datum_type());
                        })*
                        bail!("{} not covering {:?}", stringify!($name), dt)
                    })?;
                    s.equals(&inputs.len, 1)?;
                    s.equals(&outputs.len, 1)?;
                    s.with(&a.shape, move |s, a_shape| {
                        let b_shape = self.b.shape();
                        if let Ok(Some(c_shape)) = $crate::analyser::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape.into()]) {
                            s.equals(&c.shape, c_shape)?;
                        }
                        Ok(())
                    })
                }
            }
        }
    };
}

macro_rules! element_nary {
    ($Name:ident, [$($type:ty),*] => $to:ty { $expr:expr }) => {
        element_nary!($Name, match $($type => $to { $expr } ),*);
    };
    ($Name:ident, [$($type:ty),*] { $expr:expr }) => {
        element_nary!($Name, match $($type => $type { $expr } ),*);
    };
    ($Name:ident, match $($type:ty => $to:ty { $expr:expr }),*) => {
        #[allow(unused_imports)]
        use $crate::ops::prelude::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name {
            datum: TypeFact,
            n: Option<usize>,
        }

        impl Op for $Name {
            fn name(&self) -> &str {
                stringify!($Name)
            }

            fn pulsify(
                &self,
                inputs: TVec<&PulsedTensorFact>,
            ) -> TractResult<Vec<PulsifiedOp>> {
                let mut fact = inputs[0].clone();
                $(if fact.dt == <$type>::datum_type() {
                    fact.dt = <$to>::datum_type().into();
                })*
                Ok(vec!(PulsifiedOp::new(Box::new(self.clone()), tvec!(fact))))
            }
        }

        impl StatelessOp for $Name {
            /// Evaluates the operation given the input tensors.
            fn eval(&self, inputs: TVec<Value>) -> TractResult<TVec<Value>> {
                use $crate::ndarray::{ ArrayD, ArrayViewD };
                if let Some(n) = self.n {
                    if inputs.len() != n {
                        bail!("Expected {} inputs, got {}", n, inputs.len());
                    }
                }
                let dt = DatumType::super_type_for(inputs.iter().map(|i| i.datum_type()))
                    .ok_or("Could not find a supertype")?;
                let shapes:TVec<&[usize]> = inputs.iter().map(|i| i.shape()).collect();
                let shape:TVec<usize> = $crate::broadcast::multi_broadcast(&shapes)
                    .ok_or("Could not find a shape")?;
                $(if dt == <$type>::datum_type() {
                    let casts:Vec<_> = inputs.iter()
                        .map(|a| a.as_tensor().cast_to_array::<$type>().unwrap())
                        .collect();
                    let views:Vec<ArrayViewD<$type>> = casts.iter()
                        .map(|a| a.view())
                        .collect();
                    let broadcasted:Vec<_> = views.iter()
                        .map(|a| a.broadcast(&*shape).unwrap())
                        .collect();
                    let c = ArrayD::<$to>::from_shape_fn(&*shape, |dims| {
                        let values:Vec<$type> = broadcasted.iter().map(|i| i[&dims]).collect();
                        $expr(&values)
                    });
                    return Ok(tvec![c.into()])
                })*
                bail!("{} not covering {:?}", stringify!($Name), dt)
            }
        }

        impl InferenceRulesOp for $Name {
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                s: &mut Solver<'r>,
                inputs: &'p TensorsProxy,
                outputs: &'p TensorsProxy,
            ) -> InferenceResult {
                if let Some(n) = self.n {
                    s.equals(&inputs.len, n as i32)?;
                }
                s.equals(&outputs.len, 1)?;
                s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
                s.equals(&inputs[0].rank, &outputs[0].rank)?;
                s.given(&inputs.len, move |s, n| {
                    let n = n as usize;
                    s.equals_all((0..n).map(|i| (&inputs[i].datum_type).bex()).collect())?;
                    s.equals_all((0..n).map(|i| inputs[i].rank.bex()).collect())?;
                    s.given(&inputs[0].rank, move |s, rank: i32| {
                        for dim in 0..(rank as usize) {
                            s.equals(&inputs[0].shape[dim], &outputs[0].shape[dim])?;
                            s.equals_all(
                                (0..n as usize)
                                    .map(|i| inputs[i].shape[dim].bex())
                                    .collect(),
                            )?;
                        }
                        Ok(())
                    })
                })
            }
        }
    }
}

#[macro_export]
macro_rules! args_1 {
    ($inputs:expr) => {{
        if $inputs.len() != 1 {
            Err("Expected 1 arg")?
        }
        let result = $inputs.pop().unwrap();
        ::std::mem::drop($inputs);
        result
    }};
}

#[macro_export]
macro_rules! args_2 {
    ($inputs:expr) => {{
        if $inputs.len() != 2 {
            Err("Expected 2 args")?
        }
        $inputs.reverse();
        let result = ($inputs.pop().unwrap(), $inputs.pop().unwrap());
        ::std::mem::drop($inputs);
        result
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
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
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
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
    }};
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! args_5 {
    ($inputs:expr) => {{
        if $inputs.len() != 5 {
            Err("Expected 5 args")?
        }
        $inputs.reverse();
        let result = (
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
            $inputs.pop().unwrap(),
        );
        ::std::mem::drop($inputs);
        result
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
        let any = TensorFact::new();
        assert_eq!(
            $op.infer_facts(tvec![&$input], tvec![&any]).unwrap(),
            (tvec![$input.clone()], tvec![$output])
        )
    };
}

/// Asserts that backward inference results work as expected.
#[allow(unused_macros)]
#[macro_export]
macro_rules! assert_backward {
    ($op:expr, $input:ident, $output:ident) => {
        let any = TensorFact::new();
        assert_eq!(
            $op.infer_facts(tvec![&any], tvec![&$output]).unwrap(),
            (tvec![$input], tvec![$output.clone()])
        )
    };
}

#[macro_export]
macro_rules! dispatch_datum {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => {
        match $dt {
            DatumType::Bool => $($path)::*::<bool>($($args),*),
            DatumType::U8   => $($path)::*::<u8>($($args),*),
            DatumType::U16  => $($path)::*::<u16>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            DatumType::TDim => $($path)::*::<TDim>($($args),*),
            DatumType::String => bail!("String is not a Datum")
        }
    }
}

#[macro_export]
macro_rules! dispatch_numbers {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => {
        match $dt {
            DatumType::U8   => $($path)::*::<u8>($($args),*),
            DatumType::U16  => $($path)::*::<u16>($($args),*),
            DatumType::I8   => $($path)::*::<i8>($($args),*),
            DatumType::I16  => $($path)::*::<i16>($($args),*),
            DatumType::I32  => $($path)::*::<i32>($($args),*),
            DatumType::I64  => $($path)::*::<i64>($($args),*),
            DatumType::F16  => $($path)::*::<f16>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            _ => bail!("{:?} is not a number", $dt)
        }
    }
}

#[macro_export]
macro_rules! dispatch_floatlike {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => {
        match $dt {
            DatumType::F16  => $($path)::*::<f32>($($args),*),
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            _ => bail!("{:?} is not float-like", $dt)
        }
    }
}
