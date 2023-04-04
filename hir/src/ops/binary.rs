use crate::infer::*;
use crate::internal::*;

use tract_core::ops as mir;
use tract_core::ops::binary::wire_bin;
pub use tract_core::ops::binary::wire_rank_broadcast;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::math::Div;

#[derive(Debug, Clone)]
pub struct InferenceBinOp(pub Box<dyn BinMiniOp>);

impl Expansion for InferenceBinOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(s, inputs, outputs, move |typa, typb| self.0.result_datum_type(typa, typb))
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let operating_datum_type = self.0.operating_datum_type(
            target.outlet_fact(inputs[0])?.datum_type,
            target.outlet_fact(inputs[1])?.datum_type,
        )?;
        let wires = wire_cast(prefix, target, &inputs, operating_datum_type)?;
        wire_bin(prefix, target, self.0.clone(), &wires)
    }
}

pub fn rules<'r, 'p: 'r, 's: 'r, DT: Fn(DatumType, DatumType) -> TractResult<DatumType> + 'p>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
    dt: DT,
) -> InferenceResult {
    check_input_arity(inputs, 2)?;
    check_output_arity(outputs, 1)?;

    s.with(&inputs[0].shape, move |s, a_shape| {
        s.with(&inputs[1].shape, move |s, b_shape| {
            if let Ok(Some(c_shape)) =
                crate::infer::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape])
            {
                s.equals(&outputs[0].shape, c_shape)?;
            }
            Ok(())
        })
    })?;
    s.given_2(&inputs[0].datum_type, &inputs[1].datum_type, move |s, typa, typb| {
        s.equals(&outputs[0].datum_type, dt(typa, typb)?)
    })?;
    Ok(())
}

pub fn wire_cast(
    prefix: &str,
    target: &mut TypedModel,
    inputs: &[OutletId],
    operating_datum_type: DatumType,
) -> TractResult<TVec<OutletId>> {
    let mut wires = tvec!();
    for (ix, mut wire) in inputs.iter().copied().enumerate() {
        if target.outlet_fact(wire)?.datum_type != operating_datum_type {
            wire = target.wire_node(
                format!("{prefix}.cast-{ix}"),
                mir::cast::cast(operating_datum_type),
                &[wire],
            )?[0];
        }
        wires.push(wire);
    }
    Ok(wires)
}

pub trait BinIntoHir {
    fn into_hir(self) -> Box<dyn InferenceOp>;
}

impl<B: BinMiniOp> BinIntoHir for B {
    fn into_hir(self) -> Box<dyn InferenceOp> {
        expand(InferenceBinOp(Box::new(self) as _))
    }
}

#[derive(Debug, Clone)]
pub struct Nary(pub Box<dyn mir::binary::BinMiniOp>, pub bool);

impl Expansion for Nary {
    fn name(&self) -> Cow<str> {
        format!("{}Nary", self.0.name()).into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 1)?;
        let n = inputs.len();
        s.given_all(
            (0..n).map(|i| (&inputs[i].datum_type).bex()),
            move |s, types: Vec<DatumType>| {
                let dt = DatumType::super_type_for(&types)
                    .with_context(|| format!("No super type for {types:?}"))?;
                let dt = self.0.operating_datum_type(dt, dt)?;
                let result = self.0.result_datum_type(dt, dt)?;
                s.equals(&outputs[0].datum_type, result)
            },
        )?;
        s.given_all(inputs.iter().map(|i| &i.shape), move |s, shapes: Vec<TVec<TDim>>| {
            let out = tract_core::broadcast::multi_broadcast(&shapes)
                .with_context(|| format!("Failed to broadcast {:?}", &shapes))?;
            s.equals(&outputs[0].shape, ShapeFactoid::from(out))
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let types = inputs
            .iter()
            .map(|i| Ok(model.outlet_fact(*i)?.datum_type))
            .collect::<TractResult<Vec<_>>>()?;
        let dt = DatumType::super_type_for(&types)
            .with_context(|| format!("No super type for {types:?}"))?;
        let operating = self.0.operating_datum_type(dt, dt)?;
        let inputs = wire_cast(prefix, model, &inputs, operating)?;
        let mut wire = inputs[0];
        for (ix, i) in inputs[1..].iter().enumerate() {
            wire = wire_bin(format!("{prefix}.{ix}"), model, self.0.clone(), &[wire, *i])?[0];
        }
        if self.1 {
            let n = tensor0(inputs.len() as i32).cast_to_dt(dt)?.into_owned();
            let n = model.add_const(format!("{prefix}.n"), n)?;
            wire = wire_bin(format!("{prefix}.norm"), model, Div, &[wire, n])?[0];
        }
        Ok(tvec!(wire))
    }
}
