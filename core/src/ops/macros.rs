#[macro_export]
macro_rules! inference_op_as_op {
    () => {
        fn as_op(&self) -> &dyn Op {
            self
        }

        fn as_op_mut(&mut self) -> &mut dyn Op {
            self
        }
    }
}

#[macro_export]
macro_rules! element_map {
    ($Name:ident, [$($type:ty),*], $expr:expr) => {
        element_map!($Name, match $($type => { $expr } ),*);
    };
    ($Name:ident, match $($type:ty => { $expr:expr }),*) => {
        #[allow(unused_imports)]
        use $crate::internal::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name(TypeFact);

        impl StatelessOp for $Name {
            fn eval(&self, mut inputs: TVec<Arc<Tensor>>,) -> TractResult<TVec<Arc<Tensor>>> {
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $(if dt == <$type>::datum_type() {
                    let mut a = a.into_tensor();
                    let f: fn($type) -> $type = $expr;
                    for x in a.as_slice_mut::<$type>()? {
                        *x = f(x.clone());
                    }
                    return Ok(tvec!(a.into_arc_tensor()))
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
                let fact = target.outlet_fact(input)?.clone();
                let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
                Ok(tvec!(OutletId::new(id, 0)))
            }

            fn translation_invariants(&self,
                _model: &TypedModel,
                node: &TypedNode,
            ) -> TractResult<Vec<TranslationInvariant>> {
                let rank = node.outputs[0].fact.shape.rank();
                Ok((0..rank).map(|axis| TranslationInvariant { axis, period: 1 }).collect())
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
                s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
                s.equals(&inputs[0].shape, &outputs[0].shape)
            }

            inference_op_as_op!();
        }
    };
}

#[macro_export]
macro_rules! element_map_move {
    ($Name:ident, [$($type:ty),*], $expr:expr) => {
        element_map!($Name, match $($type => { $expr } ),*);
    };
    ($Name:ident, match $($type:ty => { $expr:expr }),*) => {
        element_map!($Name, match $($type => $type { $expr }),*);
    };
    ($Name:ident, match $($type:ty => $to:ty { $expr:expr }),*) => {
        #[allow(unused_imports)]
        use $crate::internal::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name(TypeFact);

        impl StatelessOp for $Name {
            fn eval(&self, mut inputs: TVec<Arc<Tensor>>,) -> TractResult<TVec<Arc<Tensor>>> {
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $(if dt == <$type>::datum_type() {
                    let a = a.into_tensor().into_array::<$type>()?;
                    return Ok(tvec!(a.mapv($expr).into_arc_tensor()));
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
                let fact = target.outlet_fact(input)?.clone();
                let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
                Ok(tvec!(OutletId::new(id, 0)))
            }

            fn translation_invariants(&self,
                _model: &TypedModel,
                node: &TypedNode,
            ) -> TractResult<Vec<TranslationInvariant>> {
                let rank = node.outputs[0].fact.shape.rank();
                Ok((0..rank).map(|axis| TranslationInvariant { axis, period: 1 }).collect())
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

            inference_op_as_op!();
        }
    };
}

#[macro_export]
macro_rules! element_map_with_params {
    ($Name:ident, [$($type:ty),*], {$($pname:ident : $pty:ty),*}, $eval_one:item) => {
        #[allow(unused_imports)]
        use $crate::internal::*;

        #[derive(Debug, Clone, new, Default)]
        pub struct $Name {
            $( pub $pname: $pty ),*
        }

        impl StatelessOp for $Name {
            fn eval(&self, mut inputs: TVec<Arc<Tensor>>,) -> TractResult<TVec<Arc<Tensor>>> {
                let a = args_1!(inputs);
                let dt = a.datum_type();
                $eval_one;
                $(if dt == <$type>::datum_type() {
                    let mut a = a.into_tensor();
                    for x in a.as_slice_mut::<$type>()? {
                        *x = eval_one(self, x.clone());
                    }
                    return Ok(tvec!(a.into_arc_tensor()))
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
                let fact = target.outlet_fact(input)?.clone();
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
                s.equals_all(wrap![
                    &inputs[0].datum_type,
                    &outputs[0].datum_type,
                ])?;
                s.equals(&inputs[0].shape, &outputs[0].shape)
            }

            inference_op_as_op!();
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
        use $crate::internal::*;

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
                let mut fact = target.outlet_fact(input)?.clone();
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
            fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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
                    return Ok(tvec![c.into_arc_tensor()])
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

            inference_op_as_op!();
        }
    }
}

#[macro_export]
macro_rules! args_1 {
    ($inputs:expr) => {{
        if $inputs.len() != 1 {
            $crate::error_chain::bail!("Expected 1 arg, got {:?}", $inputs)
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
            $crate::error_chain::bail!("Expected 2 arg, got {:?}", $inputs)
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
            $crate::error_chain::bail!("Expected 3 arg, got {:?}", $inputs)
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
            $crate::error_chain::bail!("Expected 4 arg, got {:?}", $inputs)
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
            $crate::error_chain::bail!("Expected 5 arg, got {:?}", $inputs)
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
        use $crate::datum::DatumType;
        match $dtype {
            DatumType::I32 => Box::new($op::<i32>::new($($arg),*)) as _,
            DatumType::F32 => Box::new($op::<f32>::new($($arg),*)) as _,
            DatumType::F64 => Box::new($op::<f64>::new($($arg),*)) as _,
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
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::datum::DatumType;
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
    } }
}

#[macro_export]
macro_rules! dispatch_copy {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::datum::DatumType;
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
            _ => bail!("{:?} is not Copy", $dt)
        }
    } }
}

#[macro_export]
macro_rules! dispatch_numbers {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::datum::DatumType;
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
    } }
}

#[macro_export]
macro_rules! dispatch_floatlike {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        use $crate::datum::DatumType;
        match $dt {
            DatumType::F16  => $($path)::*::<f32>($($args),*), // FIXME !!!
            DatumType::F32  => $($path)::*::<f32>($($args),*),
            DatumType::F64  => $($path)::*::<f64>($($args),*),
            _ => bail!("{:?} is not float-like", $dt)
        }
    } }
}

#[macro_export]
macro_rules! impl_op_same_as {
    () => {
        fn same_as(&self, other: &dyn Op) -> bool {
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
                if let Err(e) = left_val.close_enough(right_val, true) {
                    panic!(r#"assertion failed: `(left ~ right)`
  left: `{:?}`,
 right: `{:?}`
 {:?}"#, left_val, right_val, e)
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
                if let Err(e) = left_val.close_enough(right_val, true) {
                    panic!(r#"assertion failed: `(left ~ right)`
  left: `{:?}`,
 right: `{:?}`: {}
 {:?}"#, left_val, right_val,
                           format_args!($($arg)+), e)
                }
            }
        }
    });
}
