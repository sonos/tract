use std::ops::Deref;

use crate::internal::*;
use tract_linalg::block_quant::BlockQuantValue;
use tract_linalg::mmm::MMMInputFormat;
use tract_linalg::WeightType;

use super::{block_quant_aware_input_shape, EinSum};

#[derive(Debug)]
pub struct EinSumAnnotatedAsLinear<'a> {
    pub op: &'a EinSum,
    pub m_axis: &'a Axis,
    pub k_axis: &'a Axis,
    pub n_axes: Vec<&'a Axis>,
    pub m: usize,
    pub k: usize,
    pub ns: Vec<&'a TDim>,
    pub act_dt: DatumType,
    pub weight_type: WeightType,
}

impl<'a> EinSumAnnotatedAsLinear<'a> {
    pub fn from(
        model: &'a TypedModel,
        node: &'a TypedNode,
        op: &'a EinSum,
    ) -> TractResult<Option<Self>> {
        if node.inputs.len() != 2 {
            return Ok(None);
        }
        let input_facts = model.node_input_facts(node.id)?;
        if input_facts[0].konst.is_none() {
            return Ok(None);
        }
        let mut n_axes = vec![];
        let mut ns = Vec::<&'a TDim>::new();

        let Some(m_axis) = op.axes.iter_all_axes().find(|axis| {
            axis.inputs[0].len() == 1 && axis.inputs[1].len() == 0 && axis.outputs[0].len() == 1
        }) else {
            return Ok(None);
        };
        let Some(k_axis) = op.axes.iter_all_axes().find(|axis| {
            axis.inputs[0].len() == 1 && axis.inputs[1].len() == 1 && axis.outputs[0].len() == 0
        }) else {
            return Ok(None);
        };
        for axis in op.axes.iter_all_axes() {
            if axis != k_axis
                && axis != m_axis
                && axis.inputs[0].len() == 0
                && axis.inputs[1].len() == 1
                && axis.outputs[0].len() == 1
            {
                n_axes.push(axis);
                ns.push(&node.outputs[0].fact.shape[axis.outputs[0][0]]);
            }
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
        let m = weight_shape[m_axis.inputs[0][0]].to_usize()?;
        let k = weight_shape[k_axis.inputs[0][0]].to_usize()?;
        Ok(Some(EinSumAnnotatedAsLinear {
            op,
            m_axis,
            k_axis,
            n_axes,
            m,
            k,
            ns,
            act_dt,
            weight_type,
        }))
    }

    pub fn weight_m_axis(&self) -> usize {
        self.m_axis.inputs[0][0]
    }

    pub fn weight_k_axis(&self) -> usize {
        self.k_axis.inputs[0][0]
    }

    pub fn input_k_axis(&self) -> usize {
        self.k_axis.inputs[1][0]
    }

    pub fn output_m_axis(&self) -> usize {
        self.m_axis.outputs[0][0]
    }

    pub fn need_mmv(&self) -> bool {
        self.ns.iter().any(|n| n.as_i64().map(|n| n == 1).unwrap_or(true))
    }

    pub fn need_mmm(&self) -> bool {
        self.ns.iter().any(|n| n.as_i64().map(|n| n > 1).unwrap_or(true))
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
        if self.act_dt == self.acceptable_accumulators()[0]
            && self.weight_type == self.act_dt.into()
        {
            if let Ok(n) = self.ns.iter().cloned().product::<TDim>().to_usize() {
                let mmm = tract_linalg::ops()
                    .mmm(self.acceptable_accumulators()[0], Some(self.m), Some(self.k), Some(n))
                    .unwrap();
                return mmm.packings()[0].0.clone();
            }
        }
        if self.act_dt.is_integer() && self.weight_type == self.act_dt.into() {
            if let Ok(n) = self.ns.iter().cloned().product::<TDim>().to_usize() {
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
}

impl Deref for EinSumAnnotatedAsLinear<'_> {
    type Target = EinSum;
    fn deref(&self) -> &Self::Target {
        self.op
    }
}
