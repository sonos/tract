use crate::infer::*;
use crate::internal::*;

use tract_core::ops as mir;
use tract_core::ops::binary::BinMiniOp;

#[derive(Debug, Clone, Hash)]
pub struct InferenceBinOp(pub Box<dyn BinMiniOp>);
tract_data::impl_dyn_hash!(InferenceBinOp);

impl Expansion for InferenceBinOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    op_hir!();

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
        target.wire_node(prefix, mir::binary::TypedBinOp(self.0.clone()), &wires)
    }
}

pub fn rules<'r, 'p: 'r, 's: 'r, DT: Fn(DatumType, DatumType) -> TractResult<DatumType> + 'p>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
    dt: DT,
) -> InferenceResult {
    check_input_arity(&inputs, 2)?;
    check_output_arity(&outputs, 1)?;

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

pub fn wire_rank_broadcast(
    prefix: &str,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let facts = [target.outlet_fact(inputs[0])?.clone(), target.outlet_fact(inputs[1])?.clone()];
    let max_rank = facts[0].rank().max(facts[1].rank());
    let mut wires = tvec!();
    for i in 0..2 {
        let mut wire = inputs[i];
        for j in facts[i].rank()..max_rank {
            wire = target.wire_node(
                format!("{}.fix-rank-{}-{}", prefix, i, j),
                AxisOp::Add(0),
                &[wire],
            )?[0];
        }
        wires.push(wire);
    }
    Ok(wires)
}

pub fn wire_cast(
    prefix: &str,
    target: &mut TypedModel,
    inputs: &[OutletId],
    operating_datum_type: DatumType,
) -> TractResult<TVec<OutletId>> {
    let facts = [target.outlet_fact(inputs[0])?.clone(), target.outlet_fact(inputs[1])?.clone()];
    let mut wires = tvec!();
    for i in 0..inputs.len() {
        let mut wire = inputs[i];
        if facts[i].datum_type != operating_datum_type {
            wire = target.wire_node(
                format!("{}.cast-{}", prefix, i),
                mir::cast::cast(operating_datum_type),
                &[wire],
            )?[0];
        }
        wires.push(wire);
    }
    Ok(wires)
}

pub trait IntoHir {
    fn into_hir(self) -> Box<dyn InferenceOp>;
}

impl<B: BinMiniOp> IntoHir for B {
    fn into_hir(self) -> Box<dyn InferenceOp> {
        expand(InferenceBinOp(Box::new(self) as _))
    }
}

#[derive(Debug, Clone, Hash)]
pub struct Nary(pub Box<dyn mir::binary::BinMiniOp>, pub bool);
tract_data::impl_dyn_hash!(Nary);

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

    op_hir!();
    not_a_typed_op!();
}

impl EvalOp for Nary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut t = inputs[0].clone();
        for i in inputs[1..].into_iter() {
            t = self.0.eval_broadcast_and_typecast(tvec!(t.clone(), i.clone()))?.remove(0);
        }
        if self.1 {
            let mut t = t.into_tensor();
            dispatch_numbers!(Self::normalize_t(t.datum_type())(&mut t, inputs.len()))?;
            Ok(tvec!(t.into_arc_tensor()))
        } else {
            Ok(tvec!(t))
        }
    }
}

impl InferenceRulesOp for Nary {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        let n = inputs.len();
        s.equals_all((0..n).map(|i| (&inputs[i].datum_type).bex()).collect())?;
        s.given_all(inputs.iter().map(|i| &i.shape), move |s, shapes: Vec<TVec<TDim>>| {
            let out = tract_core::broadcast::multi_broadcast(&*shapes)
                .with_context(|| format!("Failed to broadcast {:?}", &shapes))?;
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
        let mut wire = inputs[0];
        for (ix, i) in inputs[1..].iter().enumerate() {
            wire = target.wire_node(
                format!("{}-{}", node.name, ix),
                mir::binary::TypedBinOp(self.0.clone()),
                [wire, *i].as_ref(),
            )?[0];
        }
        if self.1 {
            let n = tensor0(inputs.len() as i32)
                .cast_to_dt(node.outputs[0].fact.datum_type.concretize().unwrap())?
                .into_owned()
                .broadcast_into_rank(target.outlet_fact(inputs[0])?.rank())?;
            let n = target.add_const(format!("{}-n", node.name), n.into_arc_tensor())?;
            wire = target.wire_node(
                format!("{}-norm", node.name),
                crate::ops::math::div::bin_typed(),
                [wire, n.into()].as_ref(),
            )?[0];
        }
        Ok(tvec!(wire))
    }

    as_op!();
}
