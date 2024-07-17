use tract_core::ops::einsum::*;

use super::*;

#[test]
fn einsum_pulsedmm() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let x = model.add_source("x", f32::fact(dims!(s, 8, 2))).unwrap();
    let w = model.add_const("w", Tensor::zero::<f32>(&[8, 2, 4]).unwrap()).unwrap();

    let expr = "sij,ijk->sik".parse().unwrap();
    let einsum = EinSum { axes: expr, operating_dt: f32::datum_type(), q_params: None };

    let einsum = model.wire_node("einsum", einsum, &[x, w]).unwrap();
    model.set_output_outlets(&einsum).unwrap();
    model.declutter().unwrap();

    let mut input = Tensor::zero::<f32>(&[5, 8, 2]).unwrap();
    input.as_slice_mut::<f32>().unwrap().iter_mut().enumerate().for_each(|(ix, x)| *x = ix as f32);
    proptest_regular_against_pulse(model, 1, input.into_array().unwrap(), 0).unwrap()
}
