use crate::internal::*;
use num_traits::Zero;
use num_traits::AsPrimitive;

#[derive(Clone, Debug)]
pub struct QParams {
    pub c_datum_type: DatumType,
    pub zero_point_a: Option<Arc<Tensor>>,
    pub zero_point_b: Option<Arc<Tensor>>,
    pub zero_point_c: Option<Arc<Tensor>>,
    pub scale_factor: Option<f32>,
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

element_wise_oop!(quantize_linear_u8, QuantizeLinearU8 {scale: f32, zero_point: u8},
    [f32,i32] => u8 |op, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)|
            *y = (((*x as f32 * op.scale).round() as i32) + op.zero_point as i32) as u8
        );
        Ok(())
    }
);

element_wise_oop!(quantize_linear_i8, QuantizeLinearI8 {scale: f32, zero_point: i8},
    [f32,i32] => i8 |op, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)|
            *y = (((*x as f32 * op.scale).round() as i32) + op.zero_point as i32) as i8
        );
        Ok(())
    }
);

#[derive(Clone, Debug, new)]
pub struct DequantizeLinearF32 {
    scale: f32,
    zero_point: i32,
}

impl DequantizeLinearF32 {
    fn eval_t<T: Datum + AsPrimitive<i32>>(&self, input: &Tensor) -> TractResult<Tensor> {
        let mut output = unsafe { Tensor::uninitialized::<f32>(input.shape())? };
        input
            .as_slice::<T>()?
            .iter()
            .zip(output.as_slice_mut::<f32>()?.iter_mut())
            .for_each(|(x, y)| *y = (x.as_() - self.zero_point) as f32 * self.scale);
        Ok(output)
    }
}

impl Op for DequantizeLinearF32 {
    fn name(&self) -> Cow<str> {
        "DequantizeLinear".into()
    }

    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for DequantizeLinearF32 {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let output = match inputs[0].datum_type() {
            DatumType::I8 => self.eval_t::<i8>(&inputs[0])?,
            DatumType::I32 => self.eval_t::<i32>(&inputs[0])?,
            DatumType::U8 => self.eval_t::<u8>(&inputs[0])?,
            dt => bail!("Unsupported type {:?}", dt),
        };
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl TypedOp for DequantizeLinearF32 {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = f32::datum_type();
        Ok(tvec!(fact))
    }

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        let a = model.outlet_fact(node.inputs[0])?;
        Ok((0..a.shape.rank()).into_iter().map(|axis| AxisInfo::simple(axis)).collect())
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        target.wire_node(&*node.name, self.clone(), &[input])
    }

    typed_op_as_op!();
}

impl PulsedOp for DequantizeLinearF32 {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = f32::datum_type();
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
