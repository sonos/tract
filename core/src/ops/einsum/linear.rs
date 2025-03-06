use std::ops::Deref;

use crate::internal::*;
use tract_linalg::block_quant::BlockQuantValue;
use tract_linalg::mmm::{MMMInputFormat, MMMInputValue};
use tract_linalg::WeightType;

use super::as_matmul::EinSumAnnotatedAsMatMul;

#[derive(Debug, Clone, Hash)]
pub struct LinearEinsum {
    pub op: EinSumAnnotatedAsMatMul,
    pub act_is_left: bool,
    pub act_dt: DatumType,
    pub weight_type: WeightType,
}

impl Op for LinearEinsum {
    fn name(&self) -> Cow<str> {
        "LinearEinsum".into()
    }

    op_as_typed_op!();
}

impl LinearEinsum {
    pub fn replace(
        model: &TypedModel,
        node: &TypedNode,
        op: &EinSumAnnotatedAsMatMul,
    ) -> TractResult<Option<TypedModelPatch>> {
        if node.inputs.len() != 2 {
            return Ok(None);
        }
        let facts = model.node_input_facts(node.id)?;
        if facts[0].konst.is_none() && facts[1].konst.is_none() {
            return Ok(None);
        }
        let act_is_left = facts[0].konst.is_none();
        let (w_fact, x_fact) = (facts[act_is_left as usize], facts[1 - act_is_left as usize]);

        let act_dt = x_fact.datum_type;
        let bqv = w_fact
            .konst
            .as_ref()
            .unwrap()
            .to_scalar::<Opaque>()
            .ok()
            .and_then(|a| a.downcast_ref::<BlockQuantValue>());
        let weight_type = if let Some(a_payload) = bqv {
            WeightType::BlockQuant(a_payload.fact.format.clone())
        } else {
            w_fact.datum_type.into()
        };
        TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs,
            LinearEinsum { act_is_left: false, op: op.clone(), act_dt, weight_type },
        )
        .map(Some)
    }

    pub fn weight_mn(&self) -> usize {
        if self.act_is_left {
            self.op.n.to_usize().unwrap()
        } else {
            self.op.m.to_usize().unwrap()
        }
    }

    pub fn k(&self) -> usize {
        self.op.k.to_usize().unwrap()
    }

    pub fn weight_mn_axis(&self) -> usize {
        if self.act_is_left {
            self.n_axis().inputs[1][0]
        } else {
            self.m_axis().inputs[0][0]
        }
    }

    pub fn weight_k_axis(&self) -> usize {
        self.k_axis().inputs[self.act_is_left as usize][0]
    }

    pub fn input_k_axis(&self) -> usize {
        self.k_axis().inputs[1 - self.act_is_left as usize][0]
    }

    // pub fn output_m_axis(&self) -> usize {
    //     self.m_axis().outputs[0][0]
    // }

    pub fn need_mmv(&self) -> bool {
        !self.op.n.as_i64().is_some_and(|n| n > 1)
    }

    pub fn need_mmm(&self) -> bool {
        !self.op.n.as_i64().is_some_and(|n| n == 1)
    }

    pub fn cost_for_weights(&self, format: &dyn MMMInputFormat) -> usize {
        let ops = tract_linalg::ops();
        let acc = self.op.acceptable_accumulators();
        let mut cost = 0;
        if self.need_mmv() {
            cost += ops
                .filter_impls(format, &acc, self.act_dt)
                .map(|(mmm, _, _, pe, _)| {
                    1_000_000 + mmm.quality().cost() * 1000 + mmm.nr() * 10 - mmm.mr() * 10
                        + pe.is_some() as usize
                })
                .min()
                .unwrap_or(usize::MAX / 2);
        };
        if self.need_mmm() {
            cost += ops
                .filter_impls(format, &acc, self.act_dt)
                .map(|(mmm, _, _, pe, _)| {
                    1_000_000 + mmm.quality().cost() * 1000 - mmm.nr() * 10 - mmm.mr() * 10
                        + pe.is_some() as usize
                })
                .min()
                .unwrap_or(usize::MAX / 2);
        };
        cost
    }

    pub fn preferred_packing(&self) -> Box<dyn MMMInputFormat> {
        let (k, mn) = (self.k(), self.weight_mn());
        if self.act_dt == self.op.acceptable_accumulators()[0]
            && self.weight_type == self.act_dt.into()
        {
            if let Ok(n) = self.op.n.to_usize() {
                let mmm = tract_linalg::ops()
                    .mmm(self.op.acceptable_accumulators()[0], Some(mn), Some(k), Some(n))
                    .unwrap();
                return mmm.packings()[0].0.clone();
            }
        }
        if self.act_dt.is_integer() && self.weight_type == self.act_dt.into() {
            if let Ok(n) = self.op.n.to_usize() {
                let mmm =
                    tract_linalg::ops().mmm(i32::datum_type(), Some(mn), Some(k), Some(n)).unwrap();
                if let Some(packing) =
                    mmm.packings().iter().find(|(a, _)| a.precursor() == self.weight_type)
                {
                    return packing.0.clone();
                }
            }
        }
        dyn_clone::clone_box(
            tract_linalg::ops()
                .all_possible_packing(self.weight_type.clone())
                .min_by_key(|p| self.cost_for_weights(&**p))
                .unwrap(),
        )
    }

    pub fn transpose_weights(&mut self) {
        self.op
            .op
            .axes
            .iter_all_axes_mut()
            .for_each(|axes| axes.inputs[0].iter_mut().for_each(|pos| *pos = 1 - *pos));
    }

    fn act_mn(&self) -> _ {
        todo!()
    }
}

impl Deref for LinearEinsum {
    type Target = EinSumAnnotatedAsMatMul;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl EvalOp for LinearEinsum {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.op.eval(inputs)
    }
}

impl TypedOp for LinearEinsum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.op.output_facts(inputs)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let facts = model.node_input_facts(node.id)?;
        let (w_fact, act_fact) =
            (facts[self.act_is_left as usize], facts[1 - self.act_is_left as usize]);
        let Some(packed_weights) = w_fact
            .konst
            .as_ref()
            .unwrap()
            .to_scalar::<Opaque>()
            .ok()
            .and_then(|op| op.downcast_ref::<Box<dyn MMMInputValue>>())
        else {
            return Ok(None);
        };

        let (mode_picker, configs) = if self.act_is_left {
            todo!()
        } else if let Some((mmm, pack)) = act_mn
            .as_i64()
            .and_then(|n| {
                tract_linalg::ops().mmm(
                    op.operating_dt,
                    Some(prepack.mn()),
                    Some(prepack.k()),
                    Some(n as usize),
                )
            })
            .and_then(|mmm| {
                mmm.packings()
                    .iter()
                    .enumerate()
                    .find(|(_, (a, _b))| a.same_as(prepack.format()))
                    .map(|(ix, _)| (mmm.clone(), ix))
            })
        {
            (ModePicker::Single, vec![(mmm, pack, None)])
        } else {
            let mmms = tract_linalg::ops()
                .mmm_impls()
                .iter()
                .filter(|mmm| op.acceptable_accumulators().contains(&mmm.internal_type()))
                .flat_map(move |mmm| {
                    mmm.packings().iter().enumerate().map(|(ix, p)| (mmm.clone(), ix, &p.0, &p.1))
                })
                .filter_map(|(mmm, packing, pa, pb)| {
                    if pb.precursor().as_dt().is_some_and(|dt| dt != b_dt) {
                        None
                    } else if prepack.format().same_as(&**pa) {
                        Some((mmm, packing, None))
                    } else {
                        tract_linalg::ops()
                            .panel_extractors()
                            .iter()
                            .find(|pe| prepack.format().same_as(&*pe.from) && pe.to.same_as(&**pa))
                            .map(|pe| (mmm, packing, Some(pe)))
                    }
                })
                .collect_vec();

            let mmv = mmms
                .iter()
                .min_by_key(|(mmm, _packing, pe)| {
                    mmm.quality().cost() * 10000 + 100 * mmm.nr() + pe.is_some() as usize
                })
                .unwrap();
            let mmm = mmms
                .iter()
                .min_by_key(|(mmm, _packing, pe)| {
                    1_000_000 + mmm.quality().cost() * 10000 + pe.is_some() as usize
                        - mmm.nr() * 100
                })
                .unwrap();

            (
                ModePicker::VecVsMat,
                [mmv, mmm]
                    .iter()
                    .map(|(mmm, packing, pe)| (mmm.clone(), *packing, pe.cloned()))
                    .collect_vec(),
            )
        };
        dbg!(w_fact);
        dbg!(act_fact);
        todo!()
    }

    as_op!();
}
