use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;

pub trait ElementWiseMiniOp:
    fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast
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
        unreachable!()
    }
    #[allow(unused_variables)]
    fn eval_out_of_place(&self, t: &Tensor) -> TractResult<Tensor> {
        unreachable!()
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

dyn_clone::clone_trait_object!(ElementWiseMiniOp);
downcast_rs::impl_downcast!(ElementWiseMiniOp);

#[derive(Debug, Clone)]
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

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for ElementWiseOp {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if let Some(_dt) = self.0.output_type(inputs[0].datum_type()) {
            Ok(tvec!(self.0.eval_out_of_place(&inputs[0])?.into_arc_tensor()))
        } else {
            let mut t = args_1!(inputs).into_tensor();
            self.0.eval_in_place(&mut t)?;
            Ok(tvec!(t.into_arc_tensor()))
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

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        Invariants::new_element_wise(model, node)
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

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        target.wire_node(&*node.name, self.clone(), &[input])
    }

    as_op!();
}

impl PulsedOp for ElementWiseOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        if let Some(dt) = self.0.output_type(fact.datum_type) {
            fact.datum_type = dt;
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

#[macro_export]
macro_rules! element_wise {
    ($func:ident, $Op:ident $({$($var: ident : $var_typ: path),*})?,
        $( [$($typ:ident),*] => $f:expr ),*
        $(; cost: $cost:expr )?
        $(; prefix: $prefix:expr )?
        $(; quantize: $quantize:expr )?
        $(; validation: $validation:expr )?
    ) => {
        #[derive(Debug, Clone)]
        pub struct $Op { $( $(pub $var: $var_typ),* )? }
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
                bail!("{} does not support {:?}", self.name(), t.datum_type());
            }
            $(
            fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
                $cost(dt)
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
    ($func:ident, $Op:ident $({$($var: ident : $var_typ: path),*})?,
        $( [$($typ:ident),*] => $typ_dst:ident $f:expr ),*
        $(; cost: $cost:expr )?
        $(; info: $info:expr )?
        $(; prefix: $prefix:expr )?
        $(; quantize: $quantize:expr )?
        $(; validation: $validation:expr )?
    ) => {
        #[derive(Debug, Clone)]
        pub struct $Op { $( $(pub $var: $var_typ),* )? }
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
        pub fn $func($( $($var: $var_typ),* )?) -> $crate::ops::element_wise::ElementWiseOp {
            $crate::ops::element_wise::ElementWiseOp(Box::new($Op { $( $($var),* )? } ))
        }
    }
}
