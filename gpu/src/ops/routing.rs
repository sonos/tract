use crate::tensor::DeviceTensor;
use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;
use tract_transformers::ops::moe_ffn::GateMode;
use tract_transformers::ops::moe_ffn::RouteSelect;
use tract_transformers::ops::moe_ffn::RouteTopK;
use tract_transformers::ops::moe_ffn::RoutedCombine;

/// Most experts a [`GpuRouteSelect`] scores: one warp of 32 lanes, eight scores each.
pub const ROUTE_SELECT_MAX_EXPERTS: usize = 256;
/// Most routes per token a [`GpuRouteSelect`] picks.
pub const ROUTE_SELECT_MAX_K: usize = 16;

/// The gate a select kernel applies: 0 softmax over the top k, 1 softmax over all experts,
/// 2 sigmoid, 3 raw scores.
pub fn gate_code(gate: &GateMode) -> u32 {
    match gate {
        GateMode::SoftmaxTopk => 0,
        GateMode::SoftmaxAll => 1,
        GateMode::Sigmoid => 2,
        GateMode::Raw => 3,
    }
}

/// One backend's routing kernels.
#[derive(Clone, Copy)]
pub struct RoutingKernels {
    /// `scores [T, E]` and the optional `bias [E]`, both f32, `k` and [`gate_code`] to the
    /// token-major `token_ids`, `expert_ids` (i64) and `weights` (f32), `[T * k]` each.
    #[allow(clippy::type_complexity)]
    pub select: fn(
        &DeviceTensor,
        Option<&DeviceTensor>,
        usize,
        u32,
        &DeviceTensor,
        &DeviceTensor,
        &DeviceTensor,
    ) -> TractResult<()>,
    /// Token-major `values [T * k, D]` and `weights [T * k]`, f32, to `out [T, D]` in f32 or
    /// f16: `out[t] = sum_j weights[t * k + j] * values[t * k + j]`.
    pub combine: fn(&DeviceTensor, &DeviceTensor, usize, &DeviceTensor) -> TractResult<()>,
}

/// Backend-agnostic [`RouteSelect`] over f32 scores for up to [`ROUTE_SELECT_MAX_EXPERTS`]
/// experts and [`ROUTE_SELECT_MAX_K`] routes per token, with the same order: ties go to the
/// lower expert index.
#[derive(Clone)]
pub struct GpuRouteSelect {
    pub k: usize,
    pub gate: GateMode,
    pub backend_name: &'static str,
    pub kernels: RoutingKernels,
}

impl GpuRouteSelect {
    pub fn from_tract_core(
        source: &TypedModel,
        node: &TypedNode,
        op: &RouteSelect,
        backend_name: &'static str,
        kernels: RoutingKernels,
    ) -> TractResult<Option<GpuRouteSelect>> {
        let facts = source.node_input_facts(node.id)?;
        rule_if!(facts.iter().all(|f| f.datum_type == f32::datum_type()));
        rule_if_some!(experts = facts[0].shape[1].as_i64());
        rule_if!((1..=ROUTE_SELECT_MAX_EXPERTS as i64).contains(&experts));
        rule_if!((1..=ROUTE_SELECT_MAX_K.min(experts as usize)).contains(&op.k));
        Ok(Some(GpuRouteSelect { k: op.k, gate: op.gate.clone(), backend_name, kernels }))
    }
}

impl std::fmt::Debug for GpuRouteSelect {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}RouteSelect(k={}, gate={:?})", self.backend_name, self.k, self.gate)
    }
}

impl PartialEq for GpuRouteSelect {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.k == other.k && self.gate == other.gate
    }
}

impl Eq for GpuRouteSelect {}

impl std::hash::Hash for GpuRouteSelect {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.k.hash(state);
        self.gate.hash(state);
    }
}

impl Op for GpuRouteSelect {
    fn name(&self) -> StaticName {
        format!("{}RouteSelect", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("k: {}, gate: {:?}", self.k, self.gate)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuRouteSelect {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let scores = inputs[0].to_device_tensor()?;
        let bias = inputs.get(1).map(|b| b.to_device_tensor()).transpose()?;
        ensure!(scores.strides() == &*Tensor::natural_strides(scores.shape()));
        let routes = scores.shape()[0] * self.k;
        let output = |slot: usize, dt: DatumType| {
            crate::turn_handler::make_tensor_for_node_output(ctx, slot, dt, &[routes])
        };
        let token_ids = output(0, i64::datum_type())?;
        let expert_ids = output(1, i64::datum_type())?;
        let weights = output(2, f32::datum_type())?;
        if routes > 0 {
            (self.kernels.select)(
                scores,
                bias,
                self.k,
                gate_code(&self.gate),
                &token_ids,
                &expert_ids,
                &weights,
            )
            .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        }
        Ok(tvec!(
            token_ids.into_tensor().into_tvalue(),
            expert_ids.into_tensor().into_tvalue(),
            weights.into_tensor().into_tvalue()
        ))
    }
}

impl TypedOp for GpuRouteSelect {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            RouteSelect { k: self.k, gate: self.gate.clone() }.output_facts(facts)
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

/// The routes per token of `route_token_ids` when a [`RouteSelect`] or [`RouteTopK`] made
/// them, which lists them token-major.
fn token_major_routes(source: &TypedModel, route_token_ids: OutletId) -> Option<usize> {
    let node = source.node(route_token_ids.node);
    node.op_as::<RouteSelect>()
        .map(|op| op.k)
        .or_else(|| node.op_as::<RouteTopK>().map(|op| op.k))
        .filter(|_| route_token_ids.slot == 0)
}

/// Backend-agnostic [`RoutedCombine`] over token-major routes, `k` per token: f32 values
/// and weights summed into an f32 or f16 output shaped like its first input.
#[derive(Clone)]
pub struct GpuRoutedCombine {
    pub k: usize,
    pub backend_name: &'static str,
    pub kernels: RoutingKernels,
}

impl GpuRoutedCombine {
    pub fn from_tract_core(
        source: &TypedModel,
        node: &TypedNode,
        _op: &RoutedCombine,
        backend_name: &'static str,
        kernels: RoutingKernels,
    ) -> TractResult<Option<GpuRoutedCombine>> {
        let facts = source.node_input_facts(node.id)?;
        rule_if!(matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16));
        rule_if!(
            facts[1].datum_type == f32::datum_type() && facts[3].datum_type == f32::datum_type()
        );
        rule_if_some!(k = token_major_routes(source, node.inputs[2]));
        Ok(Some(GpuRoutedCombine { k, backend_name, kernels }))
    }
}

impl std::fmt::Debug for GpuRoutedCombine {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}RoutedCombine(k={})", self.backend_name, self.k)
    }
}

impl PartialEq for GpuRoutedCombine {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.k == other.k
    }
}

impl Eq for GpuRoutedCombine {}

impl std::hash::Hash for GpuRoutedCombine {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.k.hash(state);
    }
}

impl Op for GpuRoutedCombine {
    fn name(&self) -> StaticName {
        format!("{}RoutedCombine", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("k: {}", self.k)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuRoutedCombine {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let shape_like = inputs[0].to_device_tensor()?;
        let values = inputs[1].to_device_tensor()?;
        let weights = inputs[3].to_device_tensor()?;
        let tokens = shape_like.shape()[..shape_like.rank() - 1].iter().product::<usize>();
        ensure!(
            values.shape()[0] == tokens * self.k,
            "{} expects {} routes",
            self.name(),
            tokens * self.k
        );
        ensure!(values.strides() == &*Tensor::natural_strides(values.shape()));
        let out = crate::turn_handler::make_tensor_for_node(
            ctx,
            shape_like.datum_type(),
            shape_like.shape(),
        )?;
        if out.len() > 0 {
            (self.kernels.combine)(values, weights, self.k, &out)
                .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        }
        Ok(tvec!(out.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuRoutedCombine {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| RoutedCombine.output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
