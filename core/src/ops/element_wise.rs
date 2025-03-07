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
    fn eval_in_place(&self, t: &mut Tensor, out_dt: Option<DatumType>) -> TractResult<()> {
        bail!("Element wise eval in-place not defined");
    }
    #[allow(unused_variables)]
    fn eval_out_of_place(&self, t: &Tensor, out_dt: Option<DatumType>) -> TractResult<Tensor> {
        bail!("Element wise eval out-of-place place not defined");
    }
    #[allow(unused_variables)]
    fn cost_per_element(&self, dt: DatumType) -> TVec<(Cost, usize)> {
        tvec!()
    }
    #[allow(unused_variables)]
    fn operating_datum_type(&self, dt: DatumType) -> DatumType {
        dt
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

    #[allow(unused_variables)]
    fn same_as(&self, other: &dyn ElementWiseMiniOp) -> bool {
        false
    }
}

dyn_clone::clone_trait_object!(ElementWiseMiniOp);
downcast_rs::impl_downcast!(ElementWiseMiniOp);

#[derive(Debug, Clone)]
pub struct ElementWiseOp(pub Box<dyn ElementWiseMiniOp>, pub Option<DatumType>);

impl ElementWiseOp {
    fn output_datum_type(&self, input_dt: DatumType) -> DatumType {
        self.1.unwrap_or(self.0.operating_datum_type(input_dt))
    }
}

impl Op for ElementWiseOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.0.info()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<ElementWiseOp>() else { return false };
        self.1 == other.1 && self.0.same_as(&*other.0)
    }

    op_as_typed_op!();
}

impl EvalOp for ElementWiseOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        if let Some(_dt) = self.0.output_type(inputs[0].datum_type()) {
            Ok(tvec!(self.0.eval_out_of_place(&inputs[0], self.1)?.into_tvalue()))
        } else {
            let mut m = inputs.remove(0).into_tensor();
            self.0.eval_in_place(&mut m, self.1)?;
            Ok(tvec!(m.into()))
        }
    }
}

impl TypedOp for ElementWiseOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone().without_value();
        let dt = self.output_datum_type(fact.datum_type);
        if let Some(dt) = self.1 {
            fact.datum_type = dt;
        } else if let Some(dt) = self.0.output_type(dt) {
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
        if let Some(prec) = model.single_prec(node.id)? {
            if prec.op_is::<AxisOp>() || prec.op_is::<IntoShape>() {
                let mut patch = TypedModelPatch::default();
                let mut wire = tvec!(patch.tap_model(model, prec.inputs[0])?);
                wire = patch.wire_node(&node.name, &node.op, &wire)?;
                wire = patch.wire_node(&prec.name, &prec.op, &wire)?;
                patch.shunt_outside(model, node.id.into(), wire[0])?;
                return Ok(Some(patch));
            }
        }
        self.0.declutter(model, node)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
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
            Ok(Some(Box::new(ElementWiseOp(mini, self.1))))
        } else {
            Ok(None)
        }
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
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
        $(; operating_datum_type: $operating_datum_type:expr )?
        $(; prefix: $prefix:expr )?
        $(; quantize: $quantize:expr )?
        $(; validation: $validation:expr )?
    ) => {
        #[derive(Debug, Clone)]
        pub struct $Op { $( $( $(#[$meta])? pub $var: $var_typ),* )? }
        impl $crate::ops::element_wise::ElementWiseMiniOp for $Op {
            fn name(&self) -> String {
                format!("{}{}", self.prefix(), stringify!($Op))
            }
            #[allow(unused_variables)]
            fn same_as(&self, other: &dyn ElementWiseMiniOp) -> bool {
                let Some(other) = other.downcast_ref::<$Op>() else { return false };
                $( $( if self.$var != other.$var { return false; })* )?
                true
            }
            fn eval_in_place(&self, t: &mut Tensor, out_dt: Option<DatumType>) -> TractResult<()> {
                $(
                    $(if out_dt.unwrap_or(t.datum_type()) == $typ::datum_type() {
                        let t: &mut[$typ] = t.as_slice_mut::<$typ>()?;
                        let f: fn(&Self, &mut[$typ]) -> TractResult<()> = $f;
                        f(self, t)?;
                        return Ok(())
                    }
                    )*
                )*
                $(
                    $(
                       $(
                        let mut input_dt = t.datum_type();
                        let sout_dt = out_dt.unwrap_or(input_dt);
                        if sout_dt.unquantized() == <$typ_dt>::datum_type().unquantized() {
                           if input_dt.unquantized() != sout_dt.unquantized() {
                               // align unquantized input type to unquantized output type
                               *t = match input_dt.unquantized() {
                                   DatumType::U8 => t.clone().into_arc_tensor().offset_u8_as_i8(),
                                   DatumType::I8 => t.clone().into_arc_tensor().offset_i8_as_u8(),
                                   unknown_dt => bail!("unexpected quantization input dt {:?}", unknown_dt)
                               }.into_tensor();
                               input_dt = t.datum_type(); // because zero_point change
                           }
                           unsafe { t.set_datum_type(sout_dt) } // force cast
                           let t: &mut[$typ_dt] = t.as_slice_mut::<$typ_dt>()?;
                           let f: fn(&Self, &mut[$typ_dt], DatumType, DatumType) -> TractResult<()> = |_, xs, input_dt, out_dt| {
                               let (izp, iscale) = input_dt.zp_scale();
                               let (ozp, oscale) = out_dt.zp_scale();
                               xs.iter_mut().for_each(|x| {
                                   let x_f32 = (*x as f32 - izp as f32) * iscale;
                                   *x = (($f_f32(x_f32) / oscale) + ozp as f32).as_()
                               });
                               Ok(())
                           };
                           f(self, t, input_dt, sout_dt)?;
                           return Ok(())
                       }
                       )*
                   )*
                )?
                bail!("{} does not support {:?}", self.name(), out_dt.unwrap_or(t.datum_type()));
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
            $(
            fn operating_datum_type(&self, dt: DatumType) -> DatumType {
                ($operating_datum_type)(dt)
            }
            )?
        }
        pub fn $func($( $($var: $var_typ),* )?) -> $crate::ops::element_wise::ElementWiseOp {
            $crate::ops::element_wise::ElementWiseOp(Box::new($Op { $( $($var),* )? }), None)
        }
    }
}

#[macro_export]
macro_rules! element_wise_oop {
    ($(#[$fmeta:meta])* $func:ident, $Op:ident $({$( $(#[$meta: meta])? $var: ident : $var_typ: path),*})?,
        $( [$($typ:ident),*] => $typ_dst:ident $f:expr ),*
        $(; cost: $cost:expr )?
        $(; info: $info:expr )?
        $(; operating_datum_type: $operating_datum_type:expr )?
        $(; prefix: $prefix:expr )?
        $(; quantize: $quantize:expr )?
        $(; validation: $validation:expr )?
    ) => {
        #[derive(Debug, Clone)]
        pub struct $Op { $( $($(#[$meta])? pub $var: $var_typ),* )? }
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
            fn eval_out_of_place(&self, t: &Tensor, _out_dt: Option<DatumType>) -> TractResult<Tensor> {
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
            $(
            fn operating_datum_type(&self, dt: DatumType) -> DatumType {
                ($operating_datum_type)(dt)
            }
            )?
        }
        $(#[$fmeta])*
        pub fn $func($( $($var: $var_typ),* )?) -> $crate::ops::element_wise::ElementWiseOp {
            $crate::ops::element_wise::ElementWiseOp(Box::new($Op { $( $($var),* )? }), None)
        }
    }
}
