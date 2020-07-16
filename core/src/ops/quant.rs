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

element_wise_oop!(q_scale_i32,
 QScaleInt32 {
     #[educe(Hash(method="hash_f32"))]
     scale: f32
 },
 [i32] => i32 |op, xs, ys| { ys.iter_mut().zip(xs.iter()).for_each(|(y,&x)| *y = (x as f32 * op.scale).round() as i32); Ok(()) }
);

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
    if fact.datum_type != i32::datum_type() && fact.datum_type != f32::datum_type() {
        bail!("Only i32 and f32 allowed for quant pipeline");
    }
    if scale != 1.0 {
        if fact.datum_type == i32::datum_type() {
            wire = model.wire_node(format!("{}.scale", prefix), q_scale_i32(scale), &wire)?;
        } else if fact.datum_type == f32::datum_type() {
            let scale = tensor0(scale).broadcast_into_rank(rank)?;
            wire = model.wire_node(
                format!("{}.scale", prefix),
                crate::ops::math::mul::unary(scale.into_arc_tensor()),
                &wire,
            )?;
        }
    }

    if model.outlet_fact(wire[0])?.datum_type == f32::datum_type() {
        wire = model.wire_node(format!("{}.round", prefix), crate::ops::math::round(), &wire)?;
        wire = model.wire_node(
            format!("{}.cast-to-i32", prefix),
            crate::ops::cast::cast(i32::datum_type()),
            &wire,
        )?;
    }

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

pub fn quantize_section(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let fact = model.outlet_fact(node.inputs[0])?;
    let incoming_dt = fact.datum_type;

    let mut current = node.id;
    let mut stop = None;
    while let Some(next) = model.single_succ(current)? {
        if next.op.invariants(model, next)?.element_wise() {
            let next_fact = model.outlet_fact(next.id.into())?;
            if next_fact.datum_type == u8::datum_type() || next_fact.datum_type == i8::datum_type()
            {
                stop = Some(next)
            }
            current = next.id;
        } else {
            break;
        }
    }
    if let Some(stop) = stop {
        if incoming_dt == DatumType::I8 || incoming_dt == DatumType::U8 {
            let mut adhoc_model = TypedModel::default();
            let mut incoming_shape = tvec!(1; fact.rank());
            incoming_shape[0] = 256;
            let source = adhoc_model
                .add_source("ad-hoc", TypedFact::dt_shape(incoming_dt, &*incoming_shape)?)?;
            let mut wire = tvec!(source);
            let mut next = node;
            let mut name = None;
            loop {
                name.get_or_insert_with(|| &*next.name);
                wire = adhoc_model.wire_node(&*next.name, next.op.clone(), &wire)?;
                if next.id == stop.id {
                    break;
                }
                next = model.single_succ(next.id)?.unwrap();
            }
            adhoc_model.set_output_outlets(&wire)?;
            let input = (0u8..=255).collect::<Vec<u8>>();
            let mut input = match incoming_dt {
                DatumType::I8 => unsafe { tensor1(std::mem::transmute::<&[u8], &[i8]>(&*input)) },
                DatumType::U8 => tensor1(&input),
                _ => unreachable!(),
            };
            input.set_shape(&incoming_shape)?;
            let output = SimplePlan::new(adhoc_model)?.run(tvec!(input))?.remove(0);
            let table: &[u8] = match stop.outputs[0].fact.datum_type {
                DatumType::I8 => unsafe { std::mem::transmute(output.as_slice::<i8>()?) },
                DatumType::U8 => output.as_slice::<u8>()?,
                _ => unreachable!(),
            };
            let op = lookup_table((tract_linalg::ops().lut_u8)(table));
            let mut patch = TypedModelPatch::default();
            let wire: OutletId = patch.tap_model(model, node.inputs[0])?.into();
            let wire = patch.wire_node(name.unwrap_or(&*node.name), op, &[wire])?[0];
            patch.shunt_outside(model, stop.id.into(), wire)?;
            return Ok(Some(patch));
        }
    }
    Ok(None)
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
