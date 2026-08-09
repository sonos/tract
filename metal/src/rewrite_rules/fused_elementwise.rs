use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::element_wise::ElementWiseMiniOp;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::cast::GpuCast;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::rule_ensure;

use crate::kernels::element_wise::{
    FUSED_EW_MAX_INPUTS, FUSED_EW_MAX_RANK, FUSED_EW_MAX_STACK, FUSED_EW_MAX_STEPS,
    fused_ew_codes,
};
use crate::ops::fused_elementwise::{FusedEwStep, MetalFusedElementwise};

/// Role a device node can play inside a fused elementwise chain.
enum ChainOp {
    /// Float cast: a pure f16 rounding when narrowing (widening is free, the
    /// interpreter accumulates in f32).
    Cast { to: DatumType },
    Unary { code: u32 },
    Binary { code: u32 },
}

fn unary_code(op: &dyn ElementWiseMiniOp) -> Option<u32> {
    use fused_ew_codes::*;
    match op.name().to_lowercase().as_str() {
        "neg" => Some(NEG),
        "exp" => Some(EXP),
        "ln" => Some(LN),
        "sigmoid" => Some(SIGMOID),
        "silu" => Some(SILU),
        "tanh" => Some(TANH),
        "sqrt" => Some(SQRT),
        "rsqrt" => Some(RSQRT),
        "recip" => Some(RECIP),
        "abs" => Some(ABS),
        "square" => Some(SQUARE),
        _ => None,
    }
}

fn binary_code(op: &dyn BinMiniOp) -> Option<u32> {
    use fused_ew_codes::*;
    match op.name().to_lowercase().as_str() {
        "add" => Some(ADD),
        "sub" => Some(SUB),
        "mul" => Some(MUL),
        "div" => Some(DIV),
        "min" => Some(MIN),
        "max" => Some(MAX),
        "pow" => Some(POW),
        _ => None,
    }
}

fn fact_dt_shape(fact: &TypedFact) -> (DatumType, TVec<TDim>) {
    fact.as_device_fact()
        .map(|df| (df.fact.datum_type, df.fact.shape.to_tvec()))
        .unwrap_or_else(|| (fact.datum_type, fact.shape.to_tvec()))
}

fn is_fusable_float(dt: DatumType) -> bool {
    matches!(dt, DatumType::F16 | DatumType::F32)
}

/// Identifies a node as a fusable elementwise chain member. All inputs and
/// the output must be f16/f32 device tensors of the same rank.
fn chain_op(model: &TypedModel, node: &TypedNode) -> Option<ChainOp> {
    if node.outputs.len() != 1 {
        return None;
    }
    let in_facts = model.node_input_facts(node.id).ok()?;
    let (out_dt, _) = fact_dt_shape(&node.outputs[0].fact);
    if !is_fusable_float(out_dt) || in_facts.iter().any(|f| !is_fusable_float(fact_dt_shape(f).0))
    {
        return None;
    }
    if let Some(op) = node.op_as::<GpuCast>() {
        return Some(ChainOp::Cast { to: op.to });
    }
    if let Some(op) = node.op_as::<GpuElementWise>() {
        return unary_code(&*op.mini_op).map(|code| ChainOp::Unary { code });
    }
    if let Some(op) = node.op_as::<GpuBinOp>() {
        return binary_code(&*op.mini_op).map(|code| ChainOp::Binary { code });
    }
    None
}

/// True when `node` will be folded into its (unique) fusable consumer, so it
/// must not seed a chain of its own.
fn absorbed_by_consumer(model: &TypedModel, node: &TypedNode) -> bool {
    let outlet = OutletId::new(node.id, 0);
    if model.outputs.contains(&outlet) {
        return false;
    }
    let succs = model.outlet_successors(outlet);
    if succs.len() != 1 {
        return false;
    }
    chain_op(model, &model.nodes()[succs[0].node]).is_some()
}

struct ChainBuilder<'m> {
    model: &'m TypedModel,
    out_shape: TVec<TDim>,
    steps: TVec<FusedEwStep>,
    leaves: TVec<OutletId>,
    node_count: usize,
    depth: isize,
    max_depth: isize,
}

impl<'m> ChainBuilder<'m> {
    fn push_step(&mut self, step: FusedEwStep) -> bool {
        if self.steps.len() >= FUSED_EW_MAX_STEPS {
            return false;
        }
        self.depth += step.stack_effect();
        self.max_depth = self.max_depth.max(self.depth);
        if self.max_depth as usize > FUSED_EW_MAX_STACK {
            return false;
        }
        self.steps.push(step);
        true
    }

    /// Adds `outlet` as a leaf input (deduplicated). Fails when the leaf
    /// cannot broadcast to the chain output or the input budget is exhausted.
    fn push_leaf(&mut self, outlet: OutletId) -> TractResult<bool> {
        let fact = self.model.outlet_fact(outlet)?;
        let (dt, shape) = fact_dt_shape(fact);
        if !is_fusable_float(dt) || fact.as_device_fact().is_none() {
            return Ok(false);
        }
        if shape.len() != self.out_shape.len()
            || shape.iter().zip(&self.out_shape).any(|(d, o)| !d.is_one() && d != o)
        {
            return Ok(false);
        }
        let ix = if let Some(ix) = self.leaves.iter().position(|l| *l == outlet) {
            ix
        } else {
            if self.leaves.len() >= FUSED_EW_MAX_INPUTS {
                return Ok(false);
            }
            self.leaves.push(outlet);
            self.leaves.len() - 1
        };
        Ok(self.push_step(FusedEwStep::PushInput(ix)))
    }

    /// Emits the program for `outlet`: recurses into the producer when it is
    /// a single-consumer fusable node, otherwise records a leaf input. Any
    /// failure downgrades to a leaf; a leaf failure aborts the fusion.
    fn add_input(&mut self, outlet: OutletId) -> TractResult<bool> {
        let pnode = &self.model.nodes()[outlet.node];
        let absorbable = outlet.slot == 0
            && chain_op(self.model, pnode).is_some()
            && !self.model.outputs.contains(&outlet)
            && self.model.outlet_successors(outlet).len() == 1;
        if absorbable {
            let checkpoint =
                (self.steps.len(), self.leaves.len(), self.depth, self.max_depth, self.node_count);
            if self.add_node(pnode)? {
                return Ok(true);
            }
            // Rolled back: the subtree exceeded a budget; keep it unfused.
            self.steps.truncate(checkpoint.0);
            self.leaves.truncate(checkpoint.1);
            self.depth = checkpoint.2;
            self.max_depth = checkpoint.3;
            self.node_count = checkpoint.4;
        }
        self.push_leaf(outlet)
    }

    fn add_node(&mut self, node: &TypedNode) -> TractResult<bool> {
        let Some(op) = chain_op(self.model, node) else { return Ok(false) };
        let (out_dt, _) = fact_dt_shape(&node.outputs[0].fact);
        let round_f16 = out_dt == DatumType::F16;
        self.node_count += 1;
        match op {
            ChainOp::Cast { to } => {
                if !self.add_input(node.inputs[0])? {
                    return Ok(false);
                }
                if to == DatumType::F16 {
                    // Pure rounding step: identity in f32, round through f16.
                    if !self.push_step(FusedEwStep::Unary {
                        code: fused_ew_codes::ID,
                        round_f16: true,
                    }) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            ChainOp::Unary { code } => {
                if !self.add_input(node.inputs[0])? {
                    return Ok(false);
                }
                Ok(self.push_step(FusedEwStep::Unary { code, round_f16 }))
            }
            ChainOp::Binary { code } => {
                if !self.add_input(node.inputs[0])? || !self.add_input(node.inputs[1])? {
                    return Ok(false);
                }
                Ok(self.push_step(FusedEwStep::Binary { code, round_f16 }))
            }
        }
    }
}

/// Collapses a maximal tree of fusable elementwise device ops rooted at
/// `node` into one `MetalFusedElementwise` dispatch. Fires only on roots
/// (nodes not absorbed by their own consumer) and only when it fuses at
/// least two ops.
pub fn fuse_elementwise_chain_at(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(std::env::var_os("TRACT_METAL_DISABLE_FUSED_ELEMENTWISE").is_none());
    rule_ensure!(chain_op(model, node).is_some());
    rule_ensure!(!absorbed_by_consumer(model, node));

    let (out_dt, out_shape) = fact_dt_shape(&node.outputs[0].fact);
    rule_ensure!(out_shape.len() <= FUSED_EW_MAX_RANK);

    let mut builder = ChainBuilder {
        model,
        out_shape: out_shape.clone(),
        steps: tvec![],
        leaves: tvec![],
        node_count: 0,
        depth: 0,
        max_depth: 0,
    };
    if !builder.add_node(node)? {
        return Ok(None);
    }

    // Only patch when at least two graph nodes collapse into the dispatch;
    // rewriting a lone op would churn without cutting anything.
    rule_ensure!(builder.node_count >= 2);
    rule_ensure!(!builder.steps.is_empty());

    let mut patch = TypedModelPatch::default();
    let taps: TVec<OutletId> = builder
        .leaves
        .iter()
        .map(|l| patch.tap_model(model, *l))
        .collect::<TractResult<_>>()?;
    let op = MetalFusedElementwise {
        steps: builder.steps,
        n_inputs: taps.len(),
        out_dt,
    };
    let out = patch.wire_node(format!("{}.fused_ew", node.name), op, &taps)?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

pub fn fuse_elementwise_chain_ew(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    _op: &GpuElementWise,
) -> TractResult<Option<TypedModelPatch>> {
    fuse_elementwise_chain_at(model, node)
}

pub fn fuse_elementwise_chain_bin(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    _op: &GpuBinOp,
) -> TractResult<Option<TypedModelPatch>> {
    fuse_elementwise_chain_at(model, node)
}

pub fn fuse_elementwise_chain_cast(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    _op: &GpuCast,
) -> TractResult<Option<TypedModelPatch>> {
    fuse_elementwise_chain_at(model, node)
}
