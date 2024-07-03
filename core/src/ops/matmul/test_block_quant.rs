use tract_itertools::Itertools;

use crate::internal::*;
use crate::ops::einsum::EinSum;

struct Problem {
    weights: Tensor,
    activation: Tensor,
}

impl Problem {
    fn check(&self) -> TractResult<()> {
        let mut model = TypedModel::default();
        let w = model.add_const("w", self.weights.clone())?;
        let x = model.add_source("x", TypedFact::shape_and_dt_of(&self.activation))?;
        let y =
            model.wire_node("y", EinSum::new("mk,kn->mn".parse()?, f32::datum_type()), &[w, x])?[0];
        model.set_output_outlets(&[y])?;

        let refer = model
            .clone()
            .into_optimized()?
            .into_runnable()?
            .run(tvec!(self.activation.clone().into_tvalue()))?
            .remove(0);

        let q4ed =
            crate::transform::get_transform("block-quant").unwrap().transform_into(&model)?;

        let q4 = q4ed
            .into_optimized()?
            .into_runnable()?
            .run(tvec!(self.activation.clone().into_tvalue()))?
            .remove(0);

        q4.close_enough(&refer, Approximation::SuperApproximate)
    }
}

fn for_m_k_n(m: usize, k: usize, n: usize) -> TractResult<()> {
    let weights = tensor1(&(0..m * k).map(|x| x as f32).collect_vec()).into_shape(&[m, k])?;
    let activation = tensor1(&(0..k * n).map(|x| x as f32).collect_vec()).into_shape(&[k, n])?;
    Problem { weights, activation }.check()
}

#[test]
fn ex_4x32x4() -> TractResult<()> {
    for_m_k_n(4, 32, 4)
}

#[test]
fn ex_8x32x8() -> TractResult<()> {
    for_m_k_n(8, 32, 8)
}

#[test]
fn ex_4x64x4() -> TractResult<()> {
    for_m_k_n(4, 64, 4)
}

#[test]
fn ex_8x64x8() -> TractResult<()> {
    for_m_k_n(8, 64, 8)
}
