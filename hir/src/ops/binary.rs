use crate::infer::*;
use crate::internal::*;

use tract_core::broadcast::multi_broadcast;
use tract_core::ops as mir;
pub use tract_core::ops::cast::wire_cast;
pub use tract_core::ops::change_axes::wire_rank_broadcast;
use tract_core::ops::binary::BinMiniOp;

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
        let wires = wire_rank_broadcast(prefix, target, inputs)?;
        let wires = wire_cast(prefix, target, &wires, operating_datum_type)?;
        target.wire_node(prefix, mir::binary::TypedBinOp(self.0.clone(), None), &wires)
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

    /*
    s.with(&inputs[0].shape, move |s, a_shape| {
        s.with(&inputs[1].shape, move |s, b_shape| {
            /*
            if let Some(c_shape) =
                crate::infer::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape])
                    .with_context(|| {
                        format!(
                            "Matching {a_shape:?} and {b_shape:?} with numpy/onnx broadcast rules"
                        )
                    })?
            {
                s.equals(&outputs[0].shape, c_shape)?;
            }
            Ok(())
        })
        */
    })?;
    */
    s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, a, b| {
        s.equals(&outputs[0].shape, multi_broadcast(&[a, b])?)
    })?;
    s.given_2(&inputs[0].datum_type, &inputs[1].datum_type, move |s, typa, typb| {
        s.equals(&outputs[0].datum_type, dt(typa, typb)?)
    })?;
    Ok(())
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

impl Nary {
    fn normalize_t<T>(t: &mut Tensor, n: usize) -> TractResult<()>
    where
        T: Datum + std::ops::DivAssign<T> + Copy,
        usize: tract_num_traits::AsPrimitive<T>,
    {
        use tract_num_traits::AsPrimitive;
        let mut t = t.to_array_view_mut::<T>()?;
        let n: T = n.as_();
        t /= &tract_ndarray::arr0(n);
        Ok(())
    }
}

impl Op for Nary {
    fn name(&self) -> Cow<str> {
        format!("{}Nary", self.0.name()).into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    not_a_typed_op!();
}

impl EvalOp for Nary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut t = inputs[0].clone().into_tensor();
        for i in inputs[1..].iter() {
            let mut i = i.clone().into_tensor();
            let operating_datum_type =
                self.0.operating_datum_type(t.datum_type(), i.datum_type())?;
            if i.datum_type() != operating_datum_type {
                i = i.cast_to_dt(operating_datum_type)?.into_owned();
            }
            if t.datum_type() != operating_datum_type {
                t = t.cast_to_dt(operating_datum_type)?.into_owned();
            }
            t = self.0.eval(t.into_tvalue(), i.into_tvalue(), operating_datum_type)?;
        }
        if self.1 {
            dispatch_numbers!(Self::normalize_t(t.datum_type())(&mut t, inputs.len()))?;
        }
        Ok(tvec!(t.into_tvalue()))
    }
}

impl InferenceRulesOp for Nary {
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
            let out = tract_core::broadcast::multi_broadcast(&shapes)?;
            s.equals(&outputs[0].shape, ShapeFactoid::from(out))
        })
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<Vec<_>>();
        let types = inputs
            .iter()
            .map(|i| Ok(target.outlet_fact(*i)?.datum_type))
            .collect::<TractResult<Vec<_>>>()?;
        let dt = DatumType::super_type_for(&types)
            .with_context(|| format!("No super type for {types:?}"))?;
        let operating = self.0.operating_datum_type(dt, dt)?;
        let inputs = wire_cast(&node.name, target, &inputs, operating)?;
        let mut wire = inputs[0];
        for (ix, i) in inputs[1..].iter().enumerate() {
            let wires = wire_rank_broadcast(format!("{}.{}", node.name, ix), target, &[wire, *i])?;
            wire = target.wire_node(
                format!("{}.{}", node.name, ix),
                mir::binary::TypedBinOp(self.0.clone(), None),
                &wires,
            )?[0];
        }
        if self.1 {
            let n = tensor0(inputs.len() as i32)
                .cast_to_dt(node.outputs[0].fact.datum_type.concretize().unwrap())?
                .into_owned()
                .broadcast_into_rank(target.outlet_fact(inputs[0])?.rank())?;
            let n = target.add_const(format!("{}.n", node.name), n.into_arc_tensor())?;
            wire = target.wire_node(
                format!("{}.norm", node.name),
                crate::ops::math::div(),
                &[wire, n],
            )?[0];
        }
        Ok(tvec!(wire))
    }

    as_op!();
}
