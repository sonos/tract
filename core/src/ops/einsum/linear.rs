use crate::internal::*;
use tract_linalg::block_quant::BlockQuantValue;
use tract_linalg::mmm::MMMInputFormat;
use tract_linalg::WeightType;

use super::as_matmul::EinSumAnnotatedAsMatMul;
use super::{block_quant_aware_input_shape, EinSum};

#[derive(Debug, Clone, Hash)]
pub struct LinearEinsum {
    pub op: EinSum,
    pub m_axis: char,
    pub k_axis: char,
    pub n_axis: char,
    pub m: usize,
    pub k: usize,
    pub n: TDim,
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
    pub fn from(
        model: &TypedModel,
        node: &TypedNode,
        op: &EinSumAnnotatedAsMatMul,
    ) -> TractResult<Option<Self>> {
        if node.inputs.len() != 2 {
            return Ok(None);
        }
        let input_facts = model.node_input_facts(node.id)?;
        if input_facts[0].konst.is_none() {
            return Ok(None);
        }
        let act_dt = input_facts[1].datum_type;
        let bqv = input_facts[0]
            .konst
            .as_ref()
            .unwrap()
            .to_scalar::<Opaque>()
            .ok()
            .and_then(|a| a.downcast_ref::<BlockQuantValue>());
        let weight_type = if let Some(a_payload) = bqv {
            WeightType::BlockQuant(a_payload.fact.format.clone())
        } else {
            input_facts[0].datum_type.into()
        };
        let weight_shape = block_quant_aware_input_shape(input_facts[0])?;
        let m = weight_shape[op.m_axis.inputs[0][0]].to_usize()?;
        let k = weight_shape[op.k_axis.inputs[0][0]].to_usize()?;
        Ok(Some(LinearEinsum {
            op: op.op.clone(),
            m_axis: op.m_axis.repr,
            k_axis: op.k_axis.repr,
            n_axis: op.n_axis.repr,
            m,
            k,
            n: op.n.clone(),
            act_dt,
            weight_type,
        }))
    }

    pub fn m_axis(&self) -> &Axis {
        self.op.axes.axis(self.m_axis).unwrap()
    }

    pub fn k_axis(&self) -> &Axis {
        self.op.axes.axis(self.k_axis).unwrap()
    }

    pub fn n_axis(&self) -> &Axis {
        self.op.axes.axis(self.k_axis).unwrap()
    }

    pub fn weight_m_axis(&self) -> usize {
        self.m_axis().inputs[0][0]
    }

    pub fn weight_k_axis(&self) -> usize {
        self.k_axis().inputs[0][0]
    }

    pub fn input_k_axis(&self) -> usize {
        self.k_axis().inputs[1][0]
    }

    pub fn output_m_axis(&self) -> usize {
        self.m_axis().outputs[0][0]
    }

    pub fn need_mmv(&self) -> bool {
        !self.n.as_i64().is_some_and(|n| n > 1)
    }

    pub fn need_mmm(&self) -> bool {
        !self.n.as_i64().is_some_and(|n| n == 1)
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
        if self.act_dt == self.op.acceptable_accumulators()[0]
            && self.weight_type == self.act_dt.into()
        {
            if let Ok(n) = self.n.to_usize() {
                let mmm = tract_linalg::ops()
                    .mmm(self.op.acceptable_accumulators()[0], Some(self.m), Some(self.k), Some(n))
                    .unwrap();
                return mmm.packings()[0].0.clone();
            }
        }
        if self.act_dt.is_integer() && self.weight_type == self.act_dt.into() {
            if let Ok(n) = self.n.to_usize() {
                let mmm = tract_linalg::ops()
                    .mmm(i32::datum_type(), Some(self.m), Some(self.k), Some(n))
                    .unwrap();
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
            .axes
            .iter_all_axes_mut()
            .for_each(|axes| axes.inputs[0].iter_mut().for_each(|pos| *pos = 1 - *pos));
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

    as_op!();
}
