use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;

pub trait ElementWiseMiniOp:
    fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + DynHash
{
    fn name(&self) -> String;
    fn prefix(&self) -> &'static str {
        ""
    }
    fn validation(&self) -> Validation {
        Validation::Accurate
    }
    #[allow(unused_variables)]
    fn output_type(&self, input_type: DatumType) -> Option<DatumType> {
        None
    }
    #[allow(unused_variables)]
    fn eval_in_place(&self, t: &mut Tensor) -> TractResult<()> {
        bail!("Element wise eval in-place not defined");
    }
    #[allow(unused_variables)]
    fn eval_out_of_place(&self, t: &Tensor) -> TractResult<Tensor> {
        bail!("Element wise eval out-of-place place not defined");
    }
    #[allow(unused_variables)]
    fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
        tvec!()
    }
    #[allow(unused_variables)]
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    #[allow(unused_variables)]
    fn quantize(
        &self,
        dt: DatumType,
        scale: f32,
        zero_point: i32,
    ) -> TractResult<Option<Box<dyn ElementWiseMiniOp>>> {
        Ok(None)
    }
    #[allow(unused_variables)]
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![])
    }
}

impl Hash for Box<dyn ElementWiseMiniOp> {
    fn hash<H: std::hash::Hasher>(&self, mut state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(&mut state)
    }
}

dyn_clone::clone_trait_object!(ElementWiseMiniOp);
downcast_rs::impl_downcast!(ElementWiseMiniOp);

#[derive(Debug, Clone, Hash)]
pub struct ElementWiseOp(pub Box<dyn ElementWiseMiniOp>);

impl Op for ElementWiseOp {
    fn name(&self) -> Cow<str> {
        format!("{}", self.0.name()).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.0.info()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    op_core_lir_mir!();
    op_as_typed_op!();
}

impl_dyn_hash!(ElementWiseOp);

impl EvalOp for ElementWiseOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if let Some(_dt) = self.0.output_type(inputs[0].datum_type()) {
            Ok(tvec!(self.0.eval_out_of_place(&inputs[0])?.into_arc_tensor()))
        } else {
            if let Some(m) = Arc::get_mut(&mut inputs[0]) {
                self.0.eval_in_place(m)?;
            } else {
                let mut t = inputs.remove(0).into_tensor();
                self.0.eval_in_place(&mut t)?;
                inputs.push(t.into_arc_tensor());
            }
            Ok(inputs)
        }
    }
}

impl TypedOp for ElementWiseOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        if let Some(dt) = self.0.output_type(fact.datum_type) {
            fact.datum_type = dt;
        }
        Ok(tvec!(fact))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.0.declutter(model, node)
    }

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        Invariants::new_element_wise(inputs, outputs)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = inputs[0].shape.iter().product();
        Ok(self
            .0
            .cost_per_element(inputs[0].datum_type)
            .into_iter()
            .map(|(c, n)| (c, count.clone() * n))
            .collect())
    }

    fn quantize(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        dt: DatumType,
        scale: f32,
        zero_point: i32,
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        if let Some(mini) = self.0.quantize(dt, scale, zero_point)? {
            Ok(Some(Box::new(ElementWiseOp(mini))))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

#[macro_export]
macro_rules! element_wise {
    ($func:ident, $Op:ident $({$( $(#[$meta: meta])? $var: ident : $var_typ: path),*})?,
        $([$($typ:ident),*] => $f:expr ),*
        $(; q: $( [$($typ_dt:ident),*] => $f_f32:expr),*)?
        $(; cost: $cost:expr )?
        $(; declutter: $declutter:expr )?
        $(; prefix: $prefix:expr )?
        $(; quantize: $quantize:expr )?
        $(; validation: $validation:expr )?
    ) => {
        #[derive(Debug, Clone, Educe)]
        #[educe(Hash)]
        pub struct $Op { $( $( $(#[$meta])? pub $var: $var_typ),* )? }
        $crate::impl_dyn_hash!($Op);
        impl $crate::ops::element_wise::ElementWiseMiniOp for $Op {
            fn name(&self) -> String {
                format!("{}{}", self.prefix(), stringify!($Op))
            }
            fn eval_in_place(&self, t: &mut Tensor) -> TractResult<()> {
                $(
                    $(if t.datum_type() == $typ::datum_type() {
                        let t: &mut[$typ] = t.as_slice_mut::<$typ>()?;
                        let f: fn(&Self, &mut[$typ]) -> TractResult<()> = $f;
                        f(self, t)?;
                        return Ok(())
                    }
                    )*
                )*
                $(
                    $(
                       $(if t.datum_type().unquantized() == <$typ_dt>::datum_type().unquantized() {
                           let dt = t.datum_type();
                           let t: &mut[$typ_dt] = t.as_slice_mut::<$typ_dt>()?;
                           let f: fn(&Self, &mut[$typ_dt], DatumType) -> TractResult<()> = |_, xs, dt| {
                            let (zp, scale) = dt.zp_scale();
                            xs.iter_mut().for_each(|x| {
                                let x_f32 = (*x as f32 - zp as f32) * scale;
                                *x = (($f_f32(x_f32) / scale) + zp as f32).as_()
                            });
                            Ok(())
                        };
                           f(self, t, dt)?;
                           return Ok(())
                       }
                       )*
                   )*
                )?
                bail!("{} does not support {:?}", self.name(), t.datum_type());
            }
            $(
            fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
                $cost(dt)
            }
            )?
            $(
                fn declutter(
                    &self,
                    model: &TypedModel,
                    node: &TypedNode,
                ) -> TractResult<Option<TypedModelPatch>> {
                    $declutter(model, node)
                }
            )?
            $(
            fn prefix(&self) -> &'static str {
                $prefix
            }
            )?
            $(
            fn quantize(
                &self,
                dt: DatumType,
                scale: f32,
                zero_point: i32) -> TractResult<Option<Box<dyn ElementWiseMiniOp>>> {
                    $quantize(&self, dt, scale, zero_point)
            }
            )?
            $(
            fn validation(&self) -> Validation {
                $validation
            }
            )?
        }
        pub fn $func($( $($var: $var_typ),* )?) -> $crate::ops::element_wise::ElementWiseOp {
            $crate::ops::element_wise::ElementWiseOp(Box::new($Op { $( $($var),* )? } ))
        }
    }
}

#[macro_export]
macro_rules! element_wise_oop {
    ($(#[$fmeta:meta])* $func:ident, $Op:ident $({$( $(#[$meta: meta])? $var: ident : $var_typ: path),*})?,
        $( [$($typ:ident),*] => $typ_dst:ident $f:expr ),*
        $(; cost: $cost:expr )?
        $(; info: $info:expr )?
        $(; prefix: $prefix:expr )?
        $(; quantize: $quantize:expr )?
        $(; validation: $validation:expr )?
    ) => {
        #[derive(Debug, Clone, Educe)]
        #[educe(Hash)]
        pub struct $Op { $( $($(#[$meta])? pub $var: $var_typ),* )? }
        $crate::impl_dyn_hash!($Op);
        impl $crate::ops::element_wise::ElementWiseMiniOp for $Op {
            fn name(&self) -> String {
                format!("{}{}", self.prefix(), stringify!($Op))
            }
            fn output_type(&self, input_type: DatumType) -> Option<DatumType> {
                $(
                    $(if input_type == $typ::datum_type() {
                        return Some(<$typ_dst>::datum_type())
                    }
                    )*
                )*
                None
            }
            fn eval_out_of_place(&self, t: &Tensor) -> TractResult<Tensor> {
                $(
                    let mut dst = unsafe { Tensor::uninitialized_dt(<$typ_dst>::datum_type(), &t.shape())? };
                    $(if t.datum_type() == $typ::datum_type() {
                        let f: fn(&Self, &[$typ], &mut[$typ_dst]) -> TractResult<()> = $f;
                        f(self, t.as_slice::<$typ>()?, dst.as_slice_mut::<$typ_dst>()?)?;
                        return Ok(dst)
                    }
                    )*
                )*
                bail!("{} does not support {:?}", self.name(), t.datum_type());
            }
            $(
            fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
                $cost(dt)
            }
            )?
            $(
            fn info(&self) -> TractResult<Vec<String>> {
                $info(self)
            }
            )?
            $(
            fn prefix(&self) -> &'static str {
                $prefix
            }
            )?
            $(
            fn quantize(
                &self,
                dt: DatumType,
                scale: f32,
                zero_point: i32) -> TractResult<Option<Box<dyn ElementWiseMiniOp>>> {
                    $quantize(ft, scale, zero_point)
            }
            )?
            $(
            fn validation(&self) -> Validation {
                $validation
            }
            )?
        }
        $(#[$fmeta])*
        pub fn $func($( $($var: $var_typ),* )?) -> $crate::ops::element_wise::ElementWiseOp {
            $crate::ops::element_wise::ElementWiseOp(Box::new($Op { $( $($var),* )? } ))
        }
    }
}
