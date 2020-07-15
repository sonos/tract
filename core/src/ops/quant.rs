use crate::internal::*;
use num_traits::Zero;
use tract_linalg::lut::Lut;

#[derive(Clone, Debug, Educe)]
#[educe(Hash)]
pub struct QParams {
    pub c_datum_type: DatumType,
    pub zero_point_a: Option<Arc<Tensor>>,
    pub zero_point_b: Option<Arc<Tensor>>,
    pub zero_point_c: Option<Arc<Tensor>>,
    #[educe(Hash(method = "hash_scale"))]
    pub scale_factor: Option<f32>,
}

fn hash_scale<H: std::hash::Hasher>(it: &Option<f32>, state: &mut H) {
    Hash::hash(&it.clone().unwrap_or(1.0).to_bits(), state)
}

fn cleanup_zeropoint(zp: &Arc<Tensor>) -> Option<Arc<Tensor>> {
    match zp.datum_type() {
        DatumType::U8 => cleanup_zeropoint_t::<u8>(zp),
        DatumType::I8 => cleanup_zeropoint_t::<i8>(zp),
        _ => Some(zp.clone()),
    }
}

fn cleanup_zeropoint_t<T: Datum + Zero + Copy>(zp: &Arc<Tensor>) -> Option<Arc<Tensor>> {
    let mut zp = zp.clone();
    if zp.rank() == 1 {
        let slice = zp.as_slice::<T>().unwrap();
        if slice[1..].iter().all(|&x| x == slice[0]) {
            zp = rctensor0(slice[0]);
        }
    }
    if zp.rank() == 0 && *zp.to_scalar::<T>().unwrap() == T::zero() {
        None
    } else {
        Some(zp.into_arc_tensor())
    }
}

impl QParams {
    pub fn new(dt: DatumType) -> QParams {
        QParams {
            c_datum_type: dt,
            zero_point_a: None,
            zero_point_b: None,
            zero_point_c: None,
            scale_factor: None,
        }
    }

    pub fn with_zero_point_a(self, zero_point: &Arc<Tensor>) -> QParams {
        QParams { zero_point_a: cleanup_zeropoint(zero_point), ..self }
    }

    pub fn with_zero_point_b(self, zero_point: &Arc<Tensor>) -> QParams {
        QParams { zero_point_b: cleanup_zeropoint(zero_point), ..self }
    }

    pub fn with_zero_point_c(self, zero_point: &Arc<Tensor>) -> QParams {
        QParams { zero_point_c: cleanup_zeropoint(zero_point), ..self }
    }

    pub fn with_scale_factor(self, scale_factor: f32) -> QParams {
        QParams { scale_factor: Some(scale_factor), ..self }
    }

    pub fn set_zero_point_a(&mut self, zero_point: &Arc<Tensor>) {
        self.zero_point_a = cleanup_zeropoint(zero_point);
    }

    pub fn set_zero_point_b(&mut self, zero_point: &Arc<Tensor>) {
        self.zero_point_b = cleanup_zeropoint(zero_point);
    }

    pub fn set_zero_point_c(&mut self, zero_point: &Arc<Tensor>) {
        self.zero_point_c = cleanup_zeropoint(zero_point);
    }

    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.scale_factor = Some(scale_factor)
    }
}

pub fn quantize_linear_f32_u8(x: f32, scale: f32, zero_point: i32) -> u8 {
    (((x * scale).round() as i32) + zero_point as i32)
        .max(u8::min_value() as i32)
        .min(u8::max_value() as i32) as u8
}

pub fn quantize_linear_f32_i8(x: f32, scale: f32, zero_point: i32) -> i8 {
    (((x * scale).round() as i32) + zero_point as i32)
        .max(i8::min_value() as i32)
        .min(i8::max_value() as i32) as i8
}

pub fn wire_quant_pipeline(
    prefix: &str,
    model: &mut TypedModel,
    scale: f32,
    zero_point: i32,
    dt: DatumType,
    wires: &[OutletId],
) -> TractResult<TVec<OutletId>> {
    let fact = model.outlet_fact(wires[0])?.clone();
    let rank = fact.rank();
    let mut wire: TVec<OutletId> = wires.into();
    if fact.datum_type != f32::datum_type() {
        wire = model.wire_node(
            format!("{}.cast-to-f32", prefix),
            crate::ops::cast::cast(f32::datum_type()),
            &wire,
        )?;
    }
    if scale != 1.0 {
        let scale = tensor0(scale).broadcast_into_rank(rank)?;
        wire = model.wire_node(
            format!("{}.scale", prefix),
            crate::ops::math::mul::unary(scale.into_arc_tensor()),
            &wire,
        )?;
    }
    wire = model.wire_node(format!("{}.round", prefix), crate::ops::math::round(), &wire)?;
    wire = model.wire_node(
        format!("{}.cast-to-i32", prefix),
        crate::ops::cast::cast(i32::datum_type()),
        &wire,
    )?;
    if zero_point != 0 {
        let zero_point = tensor0(zero_point).broadcast_into_rank(rank)?;
        wire = model.wire_node(
            format!("{}.zero_point", prefix),
            crate::ops::math::add::unary(zero_point.into_arc_tensor()),
            &wire,
        )?;
    }
    let (min, max) = match dt {
        DatumType::I8 => (i8::min_value() as i32, i8::max_value() as i32),
        DatumType::U8 => (u8::min_value() as i32, u8::max_value() as i32),
        _ => bail!("QuantizeLinear only support i8 and u8 as output"),
    };
    let min = tensor0(min).broadcast_into_rank(rank)?.into_arc_tensor();
    let max = tensor0(max).broadcast_into_rank(rank)?.into_arc_tensor();
    wire = model.wire_node(format!("{}.max", prefix), crate::ops::math::max::unary(min), &wire)?;
    wire = model.wire_node(format!("{}.min", prefix), crate::ops::math::min::unary(max), &wire)?;
    wire = model.wire_node(format!("{}.cast", prefix), crate::ops::cast::cast(dt), &wire)?;
    Ok(wire)
}

element_wise_oop!(lookup_table,
 LookupTable {
     #[educe(Hash(method="hash_lookup_table"))]
     table: Box<dyn Lut>
 },
 [i8] => i8 |op, xs, ys| {
     ys.copy_from_slice(xs);
     unsafe {
         let casted = std::slice::from_raw_parts_mut(ys.as_mut_ptr() as *mut u8, ys.len());
         op.table.run(casted);
     }
     Ok(())
 },
 [u8] => u8 |op, xs, ys| {
     ys.copy_from_slice(xs);
     op.table.run(ys);
     Ok(())
 }
);

fn hash_lookup_table<H: std::hash::Hasher>(lut: &Box<dyn Lut>, h: &mut H) {
    Hash::hash_slice(lut.table(), h)
}
