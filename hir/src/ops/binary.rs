use crate::infer::*;
use crate::internal::*;

use tract_core::ops as mir;

impl InferenceRulesOp for mir::binary::InferenceBinOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
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
            s.equals(&outputs[0].datum_type, self.0.result_datum_type(typa, typb)?)
        })?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let facts = node
            .inputs
            .iter()
            .map(|i| target.outlet_fact(mapping[i]).map(|c| c.clone()))
            .collect::<TractResult<TVec<_>>>()?;
        let operating_datum_type =
            self.0.operating_datum_type(facts[0].datum_type, facts[1].datum_type)?;
        let max_rank = facts[0].rank().max(facts[1].rank());
        let mut inputs = tvec!();
        for i in 0..2 {
            let mut wire = mapping[&node.inputs[i]];
            if facts[i].datum_type != operating_datum_type {
                wire = target.wire_node(
                    format!("{}Cast{}", &*node.name, i),
                    mir::element_wise::ElementWiseOp(Box::new(mir::cast::Cast::new(
                        operating_datum_type,
                    ))),
                    &[wire],
                )?[0];
            }
            for i in facts[i].rank()..max_rank {
                wire = target.wire_node(
                    format!("{}-BroadcastToRank-{}", &*node.name, i),
                    AxisOp::Add(0),
                    &[wire],
                )?[0];
            }
            inputs.push(wire);
        }
        target.wire_node(&*node.name, mir::binary::TypedBinOp(self.0.clone()), &*inputs)
    }

    as_op!();
}

#[derive(Debug, Clone, Hash)]
pub struct Nary(pub Box<dyn mir::binary::BinMiniOp>, pub bool);
tract_linalg::impl_dyn_hash!(Nary);

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
    not_a_pulsed_op!();
}

impl StatelessOp for Nary {
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
                .ok_or_else(|| format!("Failed to broadcast {:?}", &shapes))?;
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
            let n = target.add_const(
                format!("{}-n", node.name),
                tensor0(inputs.len() as i32)
                    .cast_to_dt(node.outputs[0].fact.datum_type.concretize().unwrap())?
                    .into_owned(),
            )?;
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
