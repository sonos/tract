use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_ndarray::*;

pub fn qtensor(shape: Vec<usize>, dt: DatumType) -> BoxedStrategy<Tensor> {
    assert!(dt.unquantized() == dt);
    let len = shape.iter().product::<usize>();
    let range = if dt.is_signed() { -100..100i32 } else { 0..100i32 };
    vec(range, len..=len)
        .prop_map(move |v| {
            ArrayD::from_shape_vec(shape.clone(), v)
                .unwrap()
                .into_tensor()
                .cast_to_dt(dt.unquantized())
                .unwrap()
                .into_owned()
        })
        .boxed()
}

pub fn build_qtensor(values: Tensor, dt: DatumType, zp: Tensor, scale: f32) -> Tensor {
    let mut values = values;
    let zp = zp.cast_to_scalar::<i32>().unwrap();
    let dt = dt.quantize(QParams::ZpScale { zero_point: zp, scale });
    unsafe {
        values.set_datum_type(dt);
    }
    values
}

pub fn pick_signed_datum(signed: bool) -> DatumType {
    if signed {
        DatumType::I8
    } else {
        DatumType::U8
    }
}

pub fn qu8_dt(zp: i32, scale: f32) -> DatumType {
    u8::datum_type().quantize(QParams::ZpScale { zero_point: zp, scale })
}

pub fn qu8_tensor(tensor: Tensor, zp: i32, scale: f32) -> TractResult<Tensor> {
    Ok(tensor.cast_to_dt(qu8_dt(zp, scale))?.into_owned())
}

pub fn qu8_tensor0(value: u8, zp: i32, scale: f32) -> TractResult<Tensor> {
    qu8_tensor(tensor0(value), zp, scale)
}

pub fn qu8_tensor1(values: &[u8], zp: i32, scale: f32) -> TractResult<Tensor> {
    qu8_tensor(tensor1(values), zp, scale)
}
