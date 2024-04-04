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
    u8::datum_type().with_zp_scale(zp, scale)
}

pub fn qi8_dt(zp: i32, scale: f32) -> DatumType {
    i8::datum_type().with_zp_scale(zp, scale)
}

pub fn qu8_tensor(tensor: Tensor, zp: i32, scale: f32) -> TractResult<Tensor> {
    Ok(tensor.cast_to_dt(qu8_dt(zp, scale))?.into_owned())
}

pub fn qi8_tensor(tensor: Tensor, zp: i32, scale: f32) -> TractResult<Tensor> {
    Ok(tensor.cast_to_dt(qi8_dt(zp, scale))?.into_owned())
}

pub fn qu8_tensor0(value: u8, zp: i32, scale: f32) -> TractResult<Tensor> {
    qu8_tensor(tensor0(value), zp, scale)
}

pub fn qu8_tensor1(values: &[u8], zp: i32, scale: f32) -> TractResult<Tensor> {
    qu8_tensor(tensor1(values), zp, scale)
}

pub fn qi8_tensor1(values: &[i8], zp: i32, scale: f32) -> TractResult<Tensor> {
    qi8_tensor(tensor1(values), zp, scale)
}

pub trait QOpProblem {
    fn reference_float_ops(&self) -> TractResult<Tensor>;

    fn check_ref_with_approx(&self, result: Tensor, approx: Approximation) -> infra::TestResult {
        let mut reference = self.reference_float_ops()?;
        let out_dt = result.datum_type();
        let (zero_point, scale) = out_dt.zp_scale();
        let min_repr_val =
            (out_dt.unquantized().min_value().cast_to_scalar::<f32>()? - zero_point as f32) * scale;
        let max_repr_val =
            (out_dt.unquantized().max_value().cast_to_scalar::<f32>()? - zero_point as f32) * scale;

        reference
            .to_array_view_mut()?
            .iter_mut()
            .for_each(|x: &mut f32| *x = (*x).clamp(min_repr_val, max_repr_val));

        let mut fp_results = result.cast_to::<f32>()?.into_owned();

        let acceptable_scale_error_ratio = match approx {
            Approximation::Exact => 0.,
            Approximation::Approximate => 2.,
            _ => 3.,
        };
        assert!(tract_core::ndarray::Zip::from(fp_results.to_array_view_mut()?)
            .and(reference.to_array_view()?)
            .all(|x: &mut f32, xref: &f32| {
                let closest_x = (*x).clamp(min_repr_val, max_repr_val);
                // core maximal accepted distance by default
                (xref - closest_x).abs() <= scale * acceptable_scale_error_ratio
            }));
        Ok(())
    }
}
