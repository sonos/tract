use crate::internal::*;

pub use tract_core::ops::quant::QParams;

#[derive(Clone, Debug, new, Educe)]
#[educe(Hash)]
pub struct QuantizeLinear {
    #[educe(Hash(method = "hash_f32"))]
    scale: f32,
    zero_point: i32,
    dt: DatumType,
}

tract_linalg::impl_dyn_hash!(QuantizeLinear);

impl Expansion for QuantizeLinear {
    fn name(&self) -> Cow<str> {
        "QuantizeLinear".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        tract_core::ops::quant::wire_quant_pipeline(prefix, model, self.scale, self.zero_point, self.dt, inputs)
    }
}

#[derive(Clone, Debug, new, Educe)]
#[educe(Hash)]
pub struct DequantizeLinear {
    #[educe(Hash(method = "hash_f32"))]
    scale: f32,
    zero_point: i32,
}

tract_linalg::impl_dyn_hash!(DequantizeLinear);

impl Expansion for DequantizeLinear {
    fn name(&self) -> Cow<str> {
        "DequantizeLinearF32".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale: {} zero_point: {}", self.scale, self.zero_point)])
    }

    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(inputs[0])?.clone();
        let rank = fact.rank();
        let mut wire: TVec<OutletId> = inputs.into();
        if self.zero_point != 0 {
            if fact.datum_type != i32::datum_type() {
                wire = model.wire_node(
                    format!("{}.cast-to-i32", prefix),
                    tract_core::ops::cast::cast(i32::datum_type()),
                    &wire,
                )?;
            }
            let zero_point = tensor0(-self.zero_point).broadcast_into_rank(rank)?;
            wire = model.wire_node(
                format!("{}.zero_point", prefix),
                tract_core::ops::math::add::unary(zero_point.into_arc_tensor()),
                &wire,
            )?;
        }
        wire = model.wire_node(
            format!("{}.cast-to-f32", prefix),
            tract_core::ops::cast::cast(f32::datum_type()),
            &wire,
        )?;
        if self.scale != 1.0 {
            let scale = tensor0(self.scale).broadcast_into_rank(rank)?;
            wire = model.wire_node(
                format!("{}.scale", prefix),
                tract_core::ops::math::mul::unary(scale.into_arc_tensor()),
                &wire,
            )?;
        }
        Ok(wire)
    }
}
