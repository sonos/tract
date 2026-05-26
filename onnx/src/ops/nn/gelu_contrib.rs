use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::math::{add, erf, mul, tanh};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

// com.microsoft fused Gelu activations. All lower onto existing element-wise primitives.
//   BiasGelu(x, bias)  = erf_gelu(x + bias)            (exact, erf-based)
//   FastGelu(x, bias?) = tanh_gelu(x + bias)           (tanh approximation)
//   QuickGelu(x)       = x * sigmoid(alpha * x)         (alpha attr, default 1.702)

fn scalar(model: &mut TypedModel, name: String, v: f32, dt: DatumType) -> TractResult<OutletId> {
    model.add_const(name, tensor0(v).cast_to_dt(dt)?.into_owned())
}

// 0.5 * x * (1 + erf(x / sqrt(2)))
fn erf_gelu(
    model: &mut TypedModel,
    prefix: &str,
    x: OutletId,
    dt: DatumType,
) -> TractResult<OutletId> {
    let inv_sqrt2 = scalar(model, format!("{prefix}.inv_sqrt2"), (2.0f32).sqrt().recip(), dt)?;
    let scaled =
        wire_with_rank_broadcast(format!("{prefix}.scale"), model, mul(), &[x, inv_sqrt2])?[0];
    let e = model.wire_node(format!("{prefix}.erf"), erf(), &[scaled])?[0];
    let one = scalar(model, format!("{prefix}.one"), 1.0, dt)?;
    let one_plus =
        wire_with_rank_broadcast(format!("{prefix}.one_plus"), model, add(), &[e, one])?[0];
    let half = scalar(model, format!("{prefix}.half"), 0.5, dt)?;
    let half_x = wire_with_rank_broadcast(format!("{prefix}.half_x"), model, mul(), &[x, half])?[0];
    Ok(wire_with_rank_broadcast(format!("{prefix}.out"), model, mul(), &[half_x, one_plus])?[0])
}

// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn tanh_gelu(
    model: &mut TypedModel,
    prefix: &str,
    x: OutletId,
    dt: DatumType,
) -> TractResult<OutletId> {
    let x2 = model.wire_node(format!("{prefix}.x2"), mul(), &[x, x])?[0];
    let x3 = model.wire_node(format!("{prefix}.x3"), mul(), &[x2, x])?[0];
    let c1 = scalar(model, format!("{prefix}.c1"), 0.044715, dt)?;
    let c1x3 = wire_with_rank_broadcast(format!("{prefix}.c1x3"), model, mul(), &[x3, c1])?[0];
    let inner = wire_with_rank_broadcast(format!("{prefix}.inner"), model, add(), &[x, c1x3])?[0];
    let c0 = scalar(model, format!("{prefix}.c0"), (2.0f32 / std::f32::consts::PI).sqrt(), dt)?;
    let scaled =
        wire_with_rank_broadcast(format!("{prefix}.c0inner"), model, mul(), &[inner, c0])?[0];
    let th = model.wire_node(format!("{prefix}.tanh"), tanh(), &[scaled])?[0];
    let one = scalar(model, format!("{prefix}.one"), 1.0, dt)?;
    let one_plus =
        wire_with_rank_broadcast(format!("{prefix}.one_plus"), model, add(), &[th, one])?[0];
    let half = scalar(model, format!("{prefix}.half"), 0.5, dt)?;
    let half_x = wire_with_rank_broadcast(format!("{prefix}.half_x"), model, mul(), &[x, half])?[0];
    Ok(wire_with_rank_broadcast(format!("{prefix}.out"), model, mul(), &[half_x, one_plus])?[0])
}

macro_rules! simple_rules {
    () => {
        fn rules<'r, 'p: 'r, 's: 'r>(
            &'s self,
            s: &mut Solver<'r>,
            inputs: &'p [TensorProxy],
            outputs: &'p [TensorProxy],
        ) -> InferenceResult {
            check_output_arity(outputs, 1)?;
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
            s.equals(&inputs[0].shape, &outputs[0].shape)?;
            Ok(())
        }
    };
}

// ---- BiasGelu ----
pub fn bias_gelu(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(BiasGelu), vec![]))
}
#[derive(Debug, Clone)]
struct BiasGelu;
impl Expansion for BiasGelu {
    fn name(&self) -> StaticName {
        "BiasGelu".into()
    }
    simple_rules!();
    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let biased = wire_with_rank_broadcast(
            format!("{prefix}.bias"),
            model,
            add(),
            &[inputs[0], inputs[1]],
        )?[0];
        Ok(tvec!(erf_gelu(model, prefix, biased, dt)?))
    }
}

// ---- FastGelu ----
pub fn fast_gelu(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(FastGelu), vec![]))
}
#[derive(Debug, Clone)]
struct FastGelu;
impl Expansion for FastGelu {
    fn name(&self) -> StaticName {
        "FastGelu".into()
    }
    simple_rules!();
    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let x = if inputs.len() > 1 {
            wire_with_rank_broadcast(
                format!("{prefix}.bias"),
                model,
                add(),
                &[inputs[0], inputs[1]],
            )?[0]
        } else {
            inputs[0]
        };
        Ok(tvec!(tanh_gelu(model, prefix, x, dt)?))
    }
}

// ---- QuickGelu ----
pub fn quick_gelu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt::<f32>("alpha")?.unwrap_or(1.702);
    Ok((expand(QuickGelu { alpha }), vec![]))
}
#[derive(Debug, Clone, new)]
struct QuickGelu {
    alpha: f32,
}
impl Expansion for QuickGelu {
    fn name(&self) -> StaticName {
        "QuickGelu".into()
    }
    simple_rules!();
    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let alpha = scalar(model, format!("{prefix}.alpha"), self.alpha, dt)?;
        let ax =
            wire_with_rank_broadcast(format!("{prefix}.ax"), model, mul(), &[inputs[0], alpha])?[0];
        let s =
            model.wire_node(format!("{prefix}.sigmoid"), tract_core::ops::nn::sigmoid(), &[ax])?[0];
        Ok(tvec!(wire_with_rank_broadcast(prefix, model, mul(), &[inputs[0], s])?[0]))
    }
}
