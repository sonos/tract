macro_rules! element_map {
    ($Name:ident, [$($type:ty),*], $expr:expr) => {
        element_map!($Name, match $($type => { $expr } ),*);
    };
    ($Name:ident, match $($type:ty => { $expr:expr }),*) => {
        element_map!($Name, match $($type => $type { $expr }),*);
    };
    ($Name:ident, match $($type:ty => $to:ty { $expr:expr }),*) => {
        #[allow(unused_imports)]
        use $crate::ops::prelude::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name(TypeFact);

        impl StatelessOp for $Name {
            fn eval(&self, mut inputs: TVec<SharedTensor>,) -> TractResult<TVec<SharedTensor>> {
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $(if dt == <$type>::datum_type() {
                    let a = a.to_array::<$type>()?;
                    return Ok(tvec!(a.mapv($expr).into()));
                })*
                bail!("{} not covering {:?}", stringify!($Name), dt)
            }
        }

        impl Op for $Name {
            fn name(&self) -> Cow<str> {
                stringify!($Name).into()
            }

            fn pulsify(
                &self,
                _source: &NormalizedModel,
                node: &NormalizedNode,
                target: &mut PulsedModel,
                mapping: &HashMap<OutletId, OutletId>,
            ) -> TractResult<TVec<OutletId>> {
                let input = mapping[&node.inputs[0]];
                let fact = target.fact(input)?.clone();
                let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
                Ok(tvec!(OutletId::new(id, 0)))
            }

        }

        impl InferenceRulesOp for $Name {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                s: &mut Solver<'r>,
                inputs: &'p [TensorProxy],
                outputs: &'p [TensorProxy],
            ) -> InferenceResult {
                check_input_arity(&inputs, 1)?;
                check_output_arity(&outputs, 1)?;
                s.given(&inputs[0].datum_type, move |s, dt| {
                    $(if dt == <$type>::datum_type() {
                        s.equals(&outputs[0].datum_type, <$to>::datum_type())?;
                    })*
                    Ok(())
                })?;
                s.equals(&inputs[0].shape, &outputs[0].shape)
            }
        }
    };
}

#[macro_export]
macro_rules! element_map_with_params {
    ($Name:ident, [$($type:ty),*], {$($pname:ident : $pty:ty),*}, $expr:item) => {
        #[allow(unused_imports)]
        use $crate::ops::prelude::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name {
            $( $pname: $pty ),*
        }

        impl StatelessOp for $Name {
            fn eval(&self, mut inputs: TVec<SharedTensor>,) -> TractResult<TVec<SharedTensor>> {
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $expr;
                $(if dt == <$type>::datum_type() {
                    let mut a = a.to_array::<$type>()?;
                    a.mapv_inplace(|x| eval_one(self,x));
                    return Ok(tvec![a.into()])
                })*
                bail!("{} not covering {:?}", stringify!($Name), dt)
            }
        }

        impl Op for $Name {
            fn name(&self) -> Cow<str> {
                stringify!($Name).into()
            }
        }

        impl InferenceRulesOp for $Name {
            /// Infers properties about the input and output tensors.
            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                s: &mut Solver<'r>,
                inputs: &'p [TensorProxy],
                outputs: &'p [TensorProxy],
            ) -> InferenceResult {
                check_input_arity(&inputs, 1)?;
                check_output_arity(&outputs, 1)?;
                s.equals_all(wrap![
                    &inputs[0].datum_type,
                    &outputs[0].datum_type,
                ])?;
                s.equals(&inputs[0].shape, &outputs[0].shape)
            }
        }
    };
}

#[macro_export]
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
            use num_traits::Float;

            pub fn default() -> Bin {
                Bin::default()
            }

            fn eval_bin(a: SharedTensor, b: &SharedTensor) -> TractResult<SharedTensor> {
                let shape:TVec<usize> = $crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])
                    .ok_or_else(|| format!("Incompatible shapes {:?} and{:?}",
                                           a.shape(), b.shape()))?;
                let dt = a.datum_type().common_super_type(b.datum_type())
                    .ok_or_else(|| format!("Incompatible types {:?} and{:?}",
                                           a.datum_type(), b.datum_type()))?;
                $(if dt == <$type>::datum_type() {
                    let a = a.cast_to::<$type>()?.into_owned().into_array::<$type>()?;
                    let b = b.cast_to::<$type>()?;
                    let mut c = $crate::ndarray::ArrayD::<$to>::default(&*shape);
                    $crate::ndarray::Zip::from(&mut c)
                        .and_broadcast(&a)
                        .and_broadcast(&b.to_array_view::<$type>()?)
                        .apply(|c,&a:&$type,&b:&$type| *c = $expr(a,b));
                    return Ok(c.into())
                })*
                bail!("{} not covering {:?}", stringify!($name), dt)
            }

            #[derive(Debug, Clone, Default, new)]
            pub struct Bin(TypeFact);

            impl StatelessOp for Bin {
                fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
                    let (a, b) = args_2!(inputs);
                    Ok(tvec!(eval_bin(a, &b)?))
                }

            }

            impl Op for Bin {
                fn name(&self) -> Cow<str> {
                    concat!(stringify!($name), "::Binary").into()
                }

                fn declutter(&self, model: &$crate::model::TypedModel, node: &$crate::model::TypedNode)
                 -> TractResult<Option<TypedModelPatch>> {
                     let inputs = model.node_input_facts(node.id)?;
                    if let Some(b) = inputs[1].konst.clone() {
                        let op = UnaryA { dt: self.0, b };
                        return Ok(Some(TypedModelPatch::single_unary_op(&model, &node, op)?));
                    }
                    Ok(None)
                }

                fn pulsify(
                    &self,
                    _source: &NormalizedModel,
                    node: &NormalizedNode,
                    target: &mut PulsedModel,
                    mapping: &HashMap<OutletId, OutletId>,
                ) -> TractResult<TVec<OutletId>> {
                    use $crate::pulse::delay::Delay;
                    let a = mapping[&node.inputs[0]];
                    let b = mapping[&node.inputs[1]];
                    let a_fact = target.fact(a)?.clone();
                    let b_fact = target.fact(b)?.clone();
                    let delay = a_fact.delay.max(b_fact.delay);
                    let mut fact = target.fact(a)?.clone();
                    fact.delay = delay;
                    $(if fact.dt == <$type>::datum_type() {
                        fact.dt = <$to>::datum_type().into();
                    })*
                    let a_source = if a_fact.delay < delay {
                        let add_delay = delay - a_fact.delay;
                        let mut fixed_fact = a_fact.clone();
                        fixed_fact.delay += add_delay;
                        let id = target.chain_after(a, 
                            format!("{}/Delay", &*node.name), Delay::new(a_fact.clone(), add_delay, 0), tvec!(fixed_fact))?;
                        OutletId::new(id, 0)
                    } else {
                        a
                    };
                    let b_source = if b_fact.delay < delay {
                        let add_delay = delay - b_fact.delay;
                        let mut fixed_fact = b_fact.clone();
                        fixed_fact.delay += add_delay;
                        let id = target.chain_after(b,
                            format!("{}/Delay", &*node.name), Delay::new(b_fact.clone(), add_delay, 0), tvec!(fixed_fact))?;
                        OutletId::new(id, 0)
                    } else {
                        b
                    };
                    let id = target.add_node(&*node.name, self.clone(), tvec!(fact))?;
                    target.add_edge(a_source, InletId::new(id, 0))?;
                    target.add_edge(b_source, InletId::new(id, 1))?;
                    Ok(tvec!(OutletId::new(id, 0)))
                }
            }

            impl InferenceRulesOp for Bin {
                /// Infers properties about the input and output tensors.
                fn rules<'r, 'p: 'r, 's: 'r>(
                    &'s self,
                    s: &mut Solver<'r>,
                    inputs: &'p [TensorProxy],
                    outputs: &'p [TensorProxy],
                ) -> InferenceResult {
                    let a = &inputs[0];
                    let b = &inputs[1];
                    let c = &outputs[0];

                    let n = inputs.len();
                    s.given_all((0..n).map(|i| &inputs[i as usize].datum_type), move |s, dts| {
                        let dt:DatumType = DatumType::super_type_for(dts.iter().cloned())
                            .ok_or_else(|| format!("No supertype for {:?}", dts))?;
                        $(if dt == <$type>::datum_type() {
                            return s.equals(&outputs[0].datum_type, <$to>::datum_type());
                        })*
                        bail!("{} not covering {:?}", stringify!($name), dt)
                    })?;
                    check_input_arity(&inputs, 2)?;
                    check_output_arity(&outputs, 1)?;
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
                b: SharedTensor,
            }

            impl StatelessOp for UnaryA {
                fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
                    let a = args_1!(inputs);
                    Ok(tvec!(eval_bin(a, &self.b)?))
                }
            }

            impl Op for UnaryA {
                fn name(&self) -> Cow<str> {
                    concat!(stringify!($name), "::UnaryA").into()
                }

                fn pulsify(
                    &self,
                    _source: &NormalizedModel,
                    node: &NormalizedNode,
                    target: &mut PulsedModel,
                    mapping: &HashMap<OutletId, OutletId>,
                ) -> TractResult<TVec<OutletId>> {
                    let input = mapping[&node.inputs[0]];
                    let mut fact = target.fact(input)?.clone();
                    $(if fact.dt == <$type>::datum_type() {
                        fact.dt = <$to>::datum_type().into();
                    })*
                    let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
                    Ok(tvec!(OutletId::new(id, 0)))
                }
            }

            impl InferenceRulesOp for UnaryA {
                /// Infers properties about the input and output tensors.
                fn rules<'r, 'p: 'r, 's: 'r>(
                    &'s self,
                    s: &mut Solver<'r>,
                    inputs: &'p [TensorProxy],
                    outputs: &'p [TensorProxy],
                ) -> InferenceResult {
                    let a = &inputs[0];
                    let c = &outputs[0];

                    s.given(&inputs[0].datum_type, move |s, dt| {
                        $(if dt == <$type>::datum_type() {
                            return s.equals(&outputs[0].datum_type, <$to>::datum_type());
                        })*
                        bail!("{} not covering {:?}", stringify!($name), dt)
                    })?;
                    check_input_arity(&inputs, 1)?;
                    check_output_arity(&outputs, 1)?;
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
            fn name(&self) -> Cow<str> {
                stringify!($Name).into()
            }

            fn pulsify(
                &self,
                _source: &NormalizedModel,
                node: &NormalizedNode,
                target: &mut PulsedModel,
                mapping: &HashMap<OutletId, OutletId>,
            ) -> TractResult<TVec<OutletId>> {
                let input = mapping[&node.inputs[0]];
                let mut fact = target.fact(input)?.clone();
                $(if fact.dt == <$type>::datum_type() {
                    fact.dt = <$to>::datum_type().into();
                })*
                let id = target.add_node(&*node.name, self.clone(), tvec!(fact))?;
                for (ix, i) in node.inputs.iter().enumerate() {
                    target.add_edge(mapping[i], InletId::new(id, ix))?;
                }
                Ok(tvec!(OutletId::new(id, 0)))
            }

        }

        impl StatelessOp for $Name {
            /// Evaluates the operation given the input tensors.
            fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
                        .map(|a| a.cast_to::<$type>().unwrap())
                        .collect();
                    let views:Vec<ArrayViewD<$type>> = casts.iter()
                        .map(|a| a.to_array_view::<$type>())
                        .collect::<TractResult<_>>()?;
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
                inputs: &'p [TensorProxy],
                outputs: &'p [TensorProxy],
            ) -> InferenceResult {
                if let Some(n) = self.n {
                    check_input_arity(&inputs, n)?;
                }
                check_output_arity(&outputs, 1)?;
                s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
                s.equals(&inputs[0].rank, &outputs[0].rank)?;
                let n = inputs.len();
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
        let result = ($inputs.pop().unwrap(), $inputs.pop().unwrap(), $inputs.pop().unwrap());
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
            DatumType::String => $($path)::*::<String>($($args),*),
        }
    }
}

#[macro_export]
macro_rules! dispatch_copy {
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
            _ => bail!("{:?} is not Copy", $dt)
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
            DatumType::F16  => $($path)::*::<f32>($($args),*), // FIXME !!!
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            _ => bail!("{:?} is not float-like", $dt)
        }
    }
}

#[macro_export]
macro_rules! impl_op_same_as {
    () => {
        fn same_as(&self, other: &Op) -> bool {
            if let Some(other) = other.downcast_ref::<Self>() {
                self == other
            } else {
                false
            }
        }
    }
}

#[macro_export]
macro_rules! assert_close {
    ($left:expr, $right:expr) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !(left_val.close_enough(right_val, true)) {
                    panic!(r#"assertion failed: `(left ~ right)`
  left: `{:?}`,
 right: `{:?}`"#, left_val, right_val)
                }
            }
        }
    });
    ($left:expr, $right:expr,) => ({
        assert_eq!($left, $right)
    });
    ($left:expr, $right:expr, $($arg:tt)+) => ({
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if !(left_val.close_enough(right_val, true)) {
                    panic!(r#"assertion failed: `(left ~ right)`
  left: `{:?}`,
 right: `{:?}`: {}"#, left_val, right_val,
                           format_args!($($arg)+))
                }
            }
        }
    });
}
