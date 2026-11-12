use tract_nnef::internal::*;

use super::cast_f32::{from_f32, to_f32_vec};

pub fn register(registry: &mut Registry) {
    fn deserialize(
        builder: &mut ModelBuilder,
        invocation: &ResolvedInvocation,
    ) -> TractResult<Value> {
        let inputs = ["x", "b", "c", "log_decay", "initial_state"]
            .map(|name| invocation.named_arg_as(builder, name))
            .into_iter()
            .collect::<TractResult<TVec<_>>>()?;
        builder.wire(Mamba2Recurrent, &inputs)
    }
    fn serialize(
        ast: &mut IntoAst,
        node: &TypedNode,
        _op: &Mamba2Recurrent,
    ) -> TractResult<Option<Arc<RValue>>> {
        let inputs: Vec<Arc<RValue>> = node.inputs.iter().map(|i| ast.mapping[i].clone()).collect();
        Ok(Some(invocation("tract_transformers_mamba2_recurrent", &inputs, &[])))
    }
    registry.register_dumper(serialize);
    registry.register_primitive(
        "tract_transformers_mamba2_recurrent",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Scalar.tensor().named("b"),
            TypeName::Scalar.tensor().named("c"),
            TypeName::Scalar.tensor().named("log_decay"),
            TypeName::Scalar.tensor().named("initial_state"),
        ],
        &[("output", TypeName::Scalar.tensor()), ("final_state", TypeName::Scalar.tensor())],
        deserialize,
    );
}

/// Mamba-2 selective state-space recurrence (SSD core), applied sequentially
/// over the S axis.
///
/// Per step and head h: `state = exp(log_decay) * state + x (x) b`, then
/// `output = state . c` (contracting the state axis). This is the core of
/// mamba_ssm `selective_state_update` with a per-head scalar A: the caller
/// supplies `x` already scaled by `dt` (post softplus and dt_bias) and
/// `log_decay = dt * A`. The `D * x` skip connection and the `silu(z)` gate
/// stay in the graph.
///
/// Layout: x `[b, S, H, P]`, b/c `[b, S, G, N]`, log_decay `[b, S, H]`,
/// initial_state `[b, H, P, N]`, with `H = R * G` for an integer R resolved
/// from the shapes: head h reads b/c group `h / R`, matching
/// `repeat_interleave` on the group axis. Outputs: `[b, S, H, P]` in the x
/// datum type and the final state in the initial_state datum type. All
/// compute is f32. Mamba-1 (decay varying along N) is out of scope.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mamba2Recurrent;

impl Op for Mamba2Recurrent {
    fn name(&self) -> StaticName {
        "Mamba2Recurrent".into()
    }
    op_as_typed_op!();
}

impl EvalOp for Mamba2Recurrent {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 5, "Mamba2 expects x, b, c, log_decay, state");
        let x_shape: TVec<usize> = inputs[0].shape().into();
        let bc_shape: TVec<usize> = inputs[1].shape().into();
        let state_shape: TVec<usize> = inputs[4].shape().into();
        ensure!(x_shape.len() == 4, "Mamba2 x must be [b, S, H, P], got {x_shape:?}");
        let (batch, s_len, heads, head_dim) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        ensure!(
            bc_shape.len() == 4
                && bc_shape[0] == batch
                && bc_shape[1] == s_len
                && heads.is_multiple_of(bc_shape[2]),
            "Mamba2 b must be [b, S, G, N] with H a multiple of G, got {bc_shape:?} for x {x_shape:?}"
        );
        ensure!(inputs[2].shape() == &*bc_shape, "Mamba2 b and c must have the same shape");
        let (groups, d_state) = (bc_shape[2], bc_shape[3]);
        let ratio = heads / groups;
        ensure!(
            inputs[3].len() == batch * s_len * heads,
            "Mamba2 log_decay must have b*S*H elements"
        );
        ensure!(
            *state_shape == [batch, heads, head_dim, d_state],
            "Mamba2 state must be [b, H, P, N], got {state_shape:?}"
        );

        let x = to_f32_vec(&inputs[0])?;
        let b = to_f32_vec(&inputs[1])?;
        let c = to_f32_vec(&inputs[2])?;
        let g = to_f32_vec(&inputs[3])?;
        let mut state = to_f32_vec(&inputs[4])?;

        let mut output = vec![0f32; x.len()];
        for bi in 0..batch {
            for si in 0..s_len {
                for h in 0..heads {
                    let gi = (bi * s_len + si) * heads + h;
                    let xb = gi * head_dim;
                    let bcb = ((bi * s_len + si) * groups + h / ratio) * d_state;
                    let b_row = &b[bcb..bcb + d_state];
                    let c_row = &c[bcb..bcb + d_state];
                    let decay = g[gi].exp();
                    for p in 0..head_dim {
                        let xp = x[xb + p];
                        let sb = ((bi * heads + h) * head_dim + p) * d_state;
                        let row = &mut state[sb..sb + d_state];
                        let mut acc = 0f32;
                        for n in 0..d_state {
                            let next = row[n] * decay + xp * b_row[n];
                            row[n] = next;
                            acc += next * c_row[n];
                        }
                        output[xb + p] = acc;
                    }
                }
            }
        }
        Ok(tvec![
            from_f32(output, &x_shape, inputs[0].datum_type())?.into_tvalue(),
            from_f32(state, &state_shape, inputs[4].datum_type())?.into_tvalue(),
        ])
    }
}

impl TypedOp for Mamba2Recurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 5);
        ensure!(
            inputs.iter().all(|i| matches!(i.datum_type, DatumType::F16 | DatumType::F32)),
            "Mamba2 inputs must be f16 or f32"
        );
        ensure!(
            inputs[1].datum_type == inputs[0].datum_type
                && inputs[2].datum_type == inputs[0].datum_type,
            "Mamba2 x, b and c must share a datum type"
        );
        ensure!(inputs[0].rank() == 4, "Mamba2 x must be [b, S, H, P]");
        ensure!(inputs[1].rank() == 4, "Mamba2 b must be [b, S, G, N]");
        ensure!(inputs[1].shape == inputs[2].shape, "Mamba2 b and c must have the same shape");
        ensure!(inputs[3].rank() == 3, "Mamba2 log_decay must be [b, S, H]");
        ensure!(inputs[4].rank() == 4, "Mamba2 state must be [b, H, P, N]");
        Ok(tvec![inputs[0].without_value(), inputs[4].without_value()])
    }
    as_op!();
}
