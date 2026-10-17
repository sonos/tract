//! APEX balanced assignment: which block-quant format a matmul weight gets.
//!
//! Bands follow apex-quant `generate_config.sh` for a model of `layers` blocks
//! (default 40). Edge experts are layers `0..=4` and the last 5, near-edge is
//! `5..=9` and the 5 layers before the trailing edge, and the rest are middle.
//! Attention uses a tighter edge: the first 3 and last 3 layers.
//!
//! Balanced (and I-balanced, which differs only in the imatrix this pass does
//! not consume) assigns routed experts Q6_K on the edge and Q5_K elsewhere,
//! shared experts Q8_0, and attention and every other matmul weight Q6_K.
//!
//! `dense-hybrid-quality` is the dense/hybrid recipe from the same script.
//! Full-attention projections are Q8_0 and linear-attention / SSM projections
//! are Q4_K at every layer. The FFN is split: gate is Q6_K on the edge and
//! Q5_K elsewhere, up is Q6_K until the middle where it becomes Q5_K, and down
//! is Q8_0 on the edge and Q6_K elsewhere. `token_embd` is Q5_K. The output
//! head and any tensor the recipe does not name stay Q6_K.
//!
//! `dense-flat` is the dense control. It has no layer gradient. Gate is
//! IQ4_XS, and up and down are Q5_K. Full attention is Q5_K except `attn_v`,
//! which is Q6_K. `attn_qkv`, `attn_gate`, and `ssm_out` are Q5_K. `ssm_alpha`
//! and `ssm_beta` are Q4_K. `token_embd` is Q4_K. The output head and any
//! tensor the recipe does not name stay Q6_K.
//!
//! `dense-grad` keeps that attention and embedding map and adds the FFN layer
//! gradient. Gate is Q5_K on the edge, Q4_K near the edge, and IQ4_XS in the
//! middle. Up is Q6_K, Q5_K, then Q4_K. Down is Q6_K on the edge and Q5_K
//! elsewhere. A tensor with no layer index is treated as edge.
//!
//! `compact` (and `i-compact`, same types) is the smaller MoE tier. Routed
//! experts are Q4_K on the edge and Q3_K on the near and middle bands, shared
//! experts are Q6_K, and attention is Q4_K at every layer. Tensors the recipe
//! does not name stay Q6_K.
//!
//! `quality` (and `i-quality`, same types) keeps the balanced assignment except
//! for middle routed experts, which are IQ4_XS. Shared experts stay Q8_0 and
//! attention stays Q6_K. Tensors the recipe does not name stay Q6_K.
//!
//! `mini` is the next smaller MoE tier. Routed experts are Q3_K on the edge
//! and near bands and IQ2_S in the middle. Shared experts are Q5_K on the
//! edge and Q4_K on the near and middle bands. Attention is Q4_K on its
//! tighter edge and Q3_K elsewhere. There is no `i-mini` profile. Tensors the
//! recipe does not name stay Q6_K.
use tract_linalg::block_quant::{BlockQuant, IQ2_S, IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApexProfile {
    Balanced,
    DenseHybridQuality,
    DenseFlat,
    DenseGrad,
    Compact,
    Quality,
    Mini,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Routed,
    Shared,
    Attention,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Band {
    Edge,
    Near,
    Mid,
}

/// Format for `name` under the profile. `None` is not used: every matmul weight
/// gets a type, matching llama-quantize's base type for tensors the recipe does
/// not name. Callers still skip a weight whose K is not a multiple of the
/// format's block length.
pub fn assign(profile: ApexProfile, layers: usize, name: &str) -> Box<dyn BlockQuant> {
    let format: &dyn BlockQuant = match profile {
        ApexProfile::Balanced => balanced(layers, name),
        ApexProfile::DenseHybridQuality => dense_hybrid_quality(layers, name),
        ApexProfile::DenseFlat => dense_flat(name),
        ApexProfile::DenseGrad => dense_grad(layers, name),
        ApexProfile::Compact => compact(layers, name),
        ApexProfile::Quality => quality(layers, name),
        ApexProfile::Mini => mini(layers, name),
    };
    dyn_clone::clone_box(format)
}

fn balanced(layers: usize, name: &str) -> &'static dyn BlockQuant {
    let role = role(name);
    let band = layer_index(name).map(|layer| band(role, layer, layers));
    match (role, band) {
        (Role::Shared, _) => &Q8_0,
        (Role::Routed, Some(Band::Edge) | None) => &Q6_K,
        (Role::Routed, Some(Band::Near | Band::Mid)) => &Q5_K,
        (_, _) => &Q6_K,
    }
}

fn dense_hybrid_quality(layers: usize, name: &str) -> &'static dyn BlockQuant {
    let placed = layer_index(name).map(|layer| band(Role::Other, layer, layers));
    match dense_part(name) {
        Some(DensePart::Embed) => &Q5_K,
        Some(DensePart::FullAttn | DensePart::FullAttnV) => &Q8_0,
        Some(DensePart::LinearAttn | DensePart::LinearAttnSmall) => &Q4_K,
        Some(DensePart::Gate) => match placed {
            Some(Band::Near | Band::Mid) => &Q5_K,
            Some(Band::Edge) | None => &Q6_K,
        },
        Some(DensePart::Up) => match placed {
            Some(Band::Mid) => &Q5_K,
            Some(Band::Edge | Band::Near) | None => &Q6_K,
        },
        Some(DensePart::Down) => match placed {
            Some(Band::Edge) | None => &Q8_0,
            Some(Band::Near | Band::Mid) => &Q6_K,
        },
        Some(DensePart::Output) | None => &Q6_K,
    }
}

fn dense_flat(name: &str) -> &'static dyn BlockQuant {
    match dense_part(name) {
        Some(part) => dense_flat_rest(part),
        None => &Q6_K,
    }
}

fn dense_grad(layers: usize, name: &str) -> &'static dyn BlockQuant {
    let placed = layer_index(name).map(|layer| band(Role::Other, layer, layers));
    match dense_part(name) {
        Some(part) => dense_grad_ffn(part, placed).unwrap_or_else(|| dense_flat_rest(part)),
        None => &Q6_K,
    }
}

/// FFN gradient for `dense-grad`. `None` means the part is not an FFN projection.
fn dense_grad_ffn(part: DensePart, placed: Option<Band>) -> Option<&'static dyn BlockQuant> {
    match part {
        DensePart::Gate => Some(match placed {
            Some(Band::Edge) | None => &Q5_K,
            Some(Band::Near) => &Q4_K,
            Some(Band::Mid) => &IQ4_XS,
        }),
        DensePart::Up => Some(match placed {
            Some(Band::Edge) | None => &Q6_K,
            Some(Band::Near) => &Q5_K,
            Some(Band::Mid) => &Q4_K,
        }),
        DensePart::Down => Some(match placed {
            Some(Band::Edge) | None => &Q6_K,
            Some(Band::Near | Band::Mid) => &Q5_K,
        }),
        DensePart::Embed
        | DensePart::Output
        | DensePart::FullAttn
        | DensePart::FullAttnV
        | DensePart::LinearAttn
        | DensePart::LinearAttnSmall => None,
    }
}

/// `dense-flat` types. `dense-grad` uses this for every part that is not an FFN projection.
fn dense_flat_rest(part: DensePart) -> &'static dyn BlockQuant {
    match part {
        DensePart::Embed => &Q4_K,
        DensePart::FullAttn => &Q5_K,
        DensePart::FullAttnV => &Q6_K,
        DensePart::LinearAttn => &Q5_K,
        DensePart::LinearAttnSmall => &Q4_K,
        DensePart::Gate => &IQ4_XS,
        DensePart::Up | DensePart::Down => &Q5_K,
        DensePart::Output => &Q6_K,
    }
}

fn compact(layers: usize, name: &str) -> &'static dyn BlockQuant {
    let role = role(name);
    let placed = layer_index(name).map(|layer| band(role, layer, layers));
    match role {
        Role::Shared => &Q6_K,
        Role::Routed => match placed {
            Some(Band::Edge) | None => &Q4_K,
            Some(Band::Near | Band::Mid) => &Q3_K,
        },
        Role::Attention => &Q4_K,
        Role::Other => &Q6_K,
    }
}

fn quality(layers: usize, name: &str) -> &'static dyn BlockQuant {
    let role = role(name);
    let placed = layer_index(name).map(|layer| band(role, layer, layers));
    match role {
        Role::Shared => &Q8_0,
        Role::Routed => match placed {
            Some(Band::Edge) | None => &Q6_K,
            Some(Band::Near) => &Q5_K,
            Some(Band::Mid) => &IQ4_XS,
        },
        Role::Attention | Role::Other => &Q6_K,
    }
}

fn mini(layers: usize, name: &str) -> &'static dyn BlockQuant {
    let role = role(name);
    let placed = layer_index(name).map(|layer| band(role, layer, layers));
    match role {
        Role::Shared => match placed {
            Some(Band::Edge) | None => &Q5_K,
            Some(Band::Near) | Some(Band::Mid) => &Q4_K,
        },
        Role::Routed => match placed {
            Some(Band::Edge) | Some(Band::Near) | None => &Q3_K,
            Some(Band::Mid) => &IQ2_S,
        },
        Role::Attention => match placed {
            Some(Band::Edge) | None => &Q4_K,
            Some(Band::Near) | Some(Band::Mid) => &Q3_K,
        },
        Role::Other => &Q6_K,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DensePart {
    Embed,
    Output,
    FullAttn,
    FullAttnV,
    LinearAttn,
    LinearAttnSmall,
    Gate,
    Up,
    Down,
}

fn name_parts(name: &str) -> impl Iterator<Item = &str> {
    name.split(['.', '/']).filter(|part| !part.is_empty())
}

fn has_part(name: &str, key: &str) -> bool {
    name_parts(name).any(|part| part == key)
}

fn has_any_part(name: &str, keys: &[&str]) -> bool {
    keys.iter().any(|key| has_part(name, key))
}

/// GGUF component names from the dense profiles in `generate_config.sh`, plus
/// the HF names for the same projections. `ffn_gate` does not match
/// `ffn_gate_inp` or `ffn_gate_exps`. `attn_v` is not the other full-attention
/// projections, and `ssm_alpha` / `ssm_beta` are not the other linear ones.
fn dense_part(name: &str) -> Option<DensePart> {
    if has_any_part(name, &["experts", "shared_expert", "shared_experts"])
        || has_any_part(name, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"])
        || has_any_part(name, &["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"])
    {
        return None;
    }
    if has_any_part(name, &["token_embd", "embed_tokens"]) {
        return Some(DensePart::Embed);
    }
    if has_any_part(name, &["ssm_alpha", "ssm_beta"]) {
        return Some(DensePart::LinearAttnSmall);
    }
    if has_any_part(name, &["attn_qkv", "attn_gate", "ssm_out"]) {
        return Some(DensePart::LinearAttn);
    }
    if has_any_part(name, &["attn_v", "v_proj"]) {
        return Some(DensePart::FullAttnV);
    }
    if has_any_part(name, &["attn_q", "attn_k", "attn_output", "q_proj", "k_proj", "o_proj"]) {
        return Some(DensePart::FullAttn);
    }
    if has_any_part(name, &["ffn_gate", "gate_proj"]) {
        return Some(DensePart::Gate);
    }
    if has_any_part(name, &["ffn_up", "up_proj"]) {
        return Some(DensePart::Up);
    }
    if has_any_part(name, &["ffn_down", "down_proj"]) {
        return Some(DensePart::Down);
    }
    if has_any_part(name, &["output", "lm_head"]) {
        return Some(DensePart::Output);
    }
    None
}

fn band(role: Role, layer: usize, layers: usize) -> Band {
    let (edge_hi, edge_lo) = match role {
        Role::Attention => (2usize, layers.saturating_sub(3)),
        _ => (4, layers.saturating_sub(5)),
    };
    let near_hi = 9usize;
    let near_lo = layers.saturating_sub(10);
    if layer <= edge_hi || layer >= edge_lo {
        Band::Edge
    } else if layer <= near_hi || layer >= near_lo {
        Band::Near
    } else {
        Band::Mid
    }
}

fn role(name: &str) -> Role {
    const SHARED: &[&str] = &["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "shared_expert"];
    if SHARED.iter().any(|k| name.contains(k)) {
        return Role::Shared;
    }
    const ROUTED: &[&str] =
        &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", ".experts.", "/experts/"];
    if ROUTED.iter().any(|k| name.contains(k))
        || (name.contains("experts") && !name.contains("shared"))
    {
        return Role::Routed;
    }
    const ATTN: &[&str] =
        &["self_attn", "attn_", ".attention.", "shortconv", "ssm_alpha", "ssm_beta", "ssm_out"];
    if ATTN.iter().any(|k| name.contains(k)) {
        return Role::Attention;
    }
    Role::Other
}

fn layer_index(name: &str) -> Option<usize> {
    for key in ["layers.", "layer.", "blk.", "block."] {
        let Some(at) = name.find(key) else { continue };
        let rest = &name[at + key.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse() {
            return Some(n);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(name: &str) -> String {
        assign(ApexProfile::Balanced, 40, name).to_string()
    }

    #[test]
    fn balanced_bands() {
        assert_eq!(kind("blk.0.ffn_gate_exps.weight"), "Q6_K");
        assert_eq!(kind("blk.4.ffn_down_exps.weight"), "Q6_K");
        assert_eq!(kind("blk.5.ffn_up_exps.weight"), "Q5_K");
        assert_eq!(kind("blk.20.ffn_gate_exps.weight"), "Q5_K");
        assert_eq!(kind("blk.35.ffn_down_exps.weight"), "Q6_K");
        assert_eq!(kind("model.layers.0.mlp.experts.3.gate_proj.weight"), "Q6_K");
        assert_eq!(kind("model.layers.20.mlp.experts.1.down_proj.weight"), "Q5_K");
        assert_eq!(kind("model.layers.20.mlp.shared_expert.up_proj.weight"), "Q8_0");
        assert_eq!(kind("blk.20.ffn_gate_shexp.weight"), "Q8_0");
        assert_eq!(kind("blk.0.attn_q.weight"), "Q6_K");
        assert_eq!(kind("model.layers.10.self_attn.o_proj.weight"), "Q6_K");
        assert_eq!(kind("blk.12.ssm_out.weight"), "Q6_K");
        assert_eq!(kind("blk.20.ffn_gate_inp.weight"), "Q6_K");
        assert_eq!(kind("lm_head.weight"), "Q6_K");
    }

    fn dense(name: &str) -> String {
        assign(ApexProfile::DenseHybridQuality, 40, name).to_string()
    }

    #[test]
    fn dense_hybrid_quality_bands() {
        assert_eq!(dense("blk.0.ffn_gate.weight"), "Q6_K");
        assert_eq!(dense("blk.0.ffn_up.weight"), "Q6_K");
        assert_eq!(dense("blk.0.ffn_down.weight"), "Q8_0");
        assert_eq!(dense("blk.5.ffn_gate.weight"), "Q5_K");
        assert_eq!(dense("blk.5.ffn_up.weight"), "Q6_K");
        assert_eq!(dense("blk.5.ffn_down.weight"), "Q6_K");
        assert_eq!(dense("blk.20.ffn_gate.weight"), "Q5_K");
        assert_eq!(dense("blk.20.ffn_up.weight"), "Q5_K");
        assert_eq!(dense("blk.20.ffn_down.weight"), "Q6_K");
        assert_eq!(dense("blk.35.ffn_down.weight"), "Q8_0");
        assert_eq!(dense("blk.0.attn_q.weight"), "Q8_0");
        assert_eq!(dense("blk.20.attn_v.weight"), "Q8_0");
        assert_eq!(dense("blk.20.attn_output.weight"), "Q8_0");
        assert_eq!(dense("blk.20.attn_qkv.weight"), "Q4_K");
        assert_eq!(dense("blk.20.attn_gate.weight"), "Q4_K");
        assert_eq!(dense("blk.12.ssm_alpha.weight"), "Q4_K");
        assert_eq!(dense("blk.12.ssm_out.weight"), "Q4_K");
        assert_eq!(dense("token_embd.weight"), "Q5_K");
        assert_eq!(dense("output.weight"), "Q6_K");
        assert_eq!(dense("lm_head.weight"), "Q6_K");
        assert_eq!(dense("blk.20.ffn_gate_inp.weight"), "Q6_K");
        assert_eq!(dense("model.layers.0.mlp.gate_proj.weight"), "Q6_K");
        assert_eq!(dense("model.layers.20.mlp.up_proj.weight"), "Q5_K");
        assert_eq!(dense("model.layers.20.self_attn.q_proj.weight"), "Q8_0");
        assert_eq!(dense("model.layers.20.mlp.experts.1.gate_proj.weight"), "Q6_K");
    }

    fn flat_kind(name: &str) -> String {
        assign(ApexProfile::DenseFlat, 40, name).to_string()
    }

    #[test]
    fn dense_flat_bands() {
        for layer in [0, 5, 10, 20, 30, 35] {
            assert_eq!(flat_kind(&format!("blk.{layer}.ffn_gate.weight")), "IQ4_XS");
            assert_eq!(flat_kind(&format!("blk.{layer}.ffn_up.weight")), "Q5_K");
            assert_eq!(flat_kind(&format!("blk.{layer}.ffn_down.weight")), "Q5_K");
            assert_eq!(flat_kind(&format!("blk.{layer}.attn_q.weight")), "Q5_K");
            assert_eq!(flat_kind(&format!("blk.{layer}.attn_v.weight")), "Q6_K");
            assert_eq!(flat_kind(&format!("blk.{layer}.attn_qkv.weight")), "Q5_K");
            assert_eq!(flat_kind(&format!("blk.{layer}.ssm_alpha.weight")), "Q4_K");
        }
        assert_eq!(flat_kind("ffn_gate.weight"), "IQ4_XS");
        assert_eq!(flat_kind("blk.20.attn_k.weight"), "Q5_K");
        assert_eq!(flat_kind("blk.20.attn_output.weight"), "Q5_K");
        assert_eq!(flat_kind("blk.20.attn_gate.weight"), "Q5_K");
        assert_eq!(flat_kind("blk.12.ssm_out.weight"), "Q5_K");
        assert_eq!(flat_kind("blk.12.ssm_beta.weight"), "Q4_K");
        assert_eq!(flat_kind("token_embd.weight"), "Q4_K");
        assert_eq!(flat_kind("embed_tokens.weight"), "Q4_K");
        assert_eq!(flat_kind("output.weight"), "Q6_K");
        assert_eq!(flat_kind("lm_head.weight"), "Q6_K");
        assert_eq!(flat_kind("blk.20.ffn_gate_inp.weight"), "Q6_K");
        assert_eq!(flat_kind("blk.2.shortconv.in_proj.weight"), "Q6_K");
        assert_eq!(flat_kind("model.layers.0.mlp.gate_proj.weight"), "IQ4_XS");
        assert_eq!(flat_kind("model.layers.20.mlp.up_proj.weight"), "Q5_K");
        assert_eq!(flat_kind("model.layers.20.mlp.down_proj.weight"), "Q5_K");
        assert_eq!(flat_kind("model.layers.20.self_attn.q_proj.weight"), "Q5_K");
        assert_eq!(flat_kind("model.layers.20.self_attn.v_proj.weight"), "Q6_K");
        assert_eq!(flat_kind("model.layers.20.self_attn.o_proj.weight"), "Q5_K");
        assert_eq!(flat_kind("model.layers.20.mlp.experts.1.gate_proj.weight"), "Q6_K");
    }

    fn grad_kind(name: &str) -> String {
        assign(ApexProfile::DenseGrad, 40, name).to_string()
    }

    #[test]
    fn dense_grad_bands() {
        assert_eq!(grad_kind("blk.0.ffn_gate.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.4.ffn_gate.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.5.ffn_gate.weight"), "Q4_K");
        assert_eq!(grad_kind("blk.9.ffn_gate.weight"), "Q4_K");
        assert_eq!(grad_kind("blk.10.ffn_gate.weight"), "IQ4_XS");
        assert_eq!(grad_kind("blk.20.ffn_gate.weight"), "IQ4_XS");
        assert_eq!(grad_kind("blk.29.ffn_gate.weight"), "IQ4_XS");
        assert_eq!(grad_kind("blk.30.ffn_gate.weight"), "Q4_K");
        assert_eq!(grad_kind("blk.34.ffn_gate.weight"), "Q4_K");
        assert_eq!(grad_kind("blk.35.ffn_gate.weight"), "Q5_K");
        assert_eq!(grad_kind("ffn_gate.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.0.ffn_up.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.5.ffn_up.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.20.ffn_up.weight"), "Q4_K");
        assert_eq!(grad_kind("blk.30.ffn_up.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.35.ffn_up.weight"), "Q6_K");
        assert_eq!(grad_kind("ffn_up.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.0.ffn_down.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.5.ffn_down.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.20.ffn_down.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.35.ffn_down.weight"), "Q6_K");
        assert_eq!(grad_kind("ffn_down.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.0.attn_q.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.0.attn_v.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.20.attn_q.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.20.attn_v.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.20.attn_qkv.weight"), "Q5_K");
        assert_eq!(grad_kind("blk.12.ssm_alpha.weight"), "Q4_K");
        assert_eq!(grad_kind("token_embd.weight"), "Q4_K");
        assert_eq!(grad_kind("output.weight"), "Q6_K");
        assert_eq!(grad_kind("lm_head.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.20.ffn_gate_exps.weight"), "Q6_K");
        assert_eq!(grad_kind("blk.2.shortconv.in_proj.weight"), "Q6_K");
        assert_eq!(grad_kind("model.layers.0.mlp.gate_proj.weight"), "Q5_K");
        assert_eq!(grad_kind("model.layers.5.mlp.gate_proj.weight"), "Q4_K");
        assert_eq!(grad_kind("model.layers.20.mlp.gate_proj.weight"), "IQ4_XS");
        assert_eq!(grad_kind("model.layers.20.mlp.up_proj.weight"), "Q4_K");
        assert_eq!(grad_kind("model.layers.20.mlp.down_proj.weight"), "Q5_K");
        assert_eq!(grad_kind("model.layers.0.mlp.down_proj.weight"), "Q6_K");
        assert_eq!(grad_kind("model.layers.20.mlp.experts.1.gate_proj.weight"), "Q6_K");
    }

    fn compact_kind(name: &str) -> String {
        assign(ApexProfile::Compact, 40, name).to_string()
    }

    #[test]
    fn compact_bands() {
        assert_eq!(compact_kind("blk.0.ffn_gate_exps.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.4.ffn_down_exps.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.5.ffn_up_exps.weight"), "Q3_K");
        assert_eq!(compact_kind("blk.9.ffn_gate_exps.weight"), "Q3_K");
        assert_eq!(compact_kind("blk.10.ffn_down_exps.weight"), "Q3_K");
        assert_eq!(compact_kind("blk.20.ffn_gate_exps.weight"), "Q3_K");
        assert_eq!(compact_kind("blk.30.ffn_up_exps.weight"), "Q3_K");
        assert_eq!(compact_kind("blk.35.ffn_down_exps.weight"), "Q4_K");
        assert_eq!(compact_kind("ffn_gate_exps.weight"), "Q4_K");
        assert_eq!(compact_kind("model.layers.0.mlp.experts.3.gate_proj.weight"), "Q4_K");
        assert_eq!(compact_kind("model.layers.20.mlp.experts.1.down_proj.weight"), "Q3_K");
        assert_eq!(compact_kind("model.layers.20.mlp.shared_expert.up_proj.weight"), "Q6_K");
        assert_eq!(compact_kind("blk.0.ffn_gate_shexp.weight"), "Q6_K");
        assert_eq!(compact_kind("blk.20.ffn_down_shexp.weight"), "Q6_K");
        assert_eq!(compact_kind("blk.0.attn_q.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.3.attn_k.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.20.attn_output.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.20.attn_qkv.weight"), "Q4_K");
        assert_eq!(compact_kind("model.layers.10.self_attn.o_proj.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.12.ssm_out.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.2.shortconv.in_proj.weight"), "Q4_K");
        assert_eq!(compact_kind("blk.20.ffn_gate_inp.weight"), "Q6_K");
        assert_eq!(compact_kind("token_embd.weight"), "Q6_K");
        assert_eq!(compact_kind("output.weight"), "Q6_K");
        assert_eq!(compact_kind("lm_head.weight"), "Q6_K");
    }

    fn quality_kind(name: &str) -> String {
        assign(ApexProfile::Quality, 40, name).to_string()
    }

    #[test]
    fn quality_bands() {
        assert_eq!(quality_kind("blk.0.ffn_gate_exps.weight"), "Q6_K");
        assert_eq!(quality_kind("blk.4.ffn_down_exps.weight"), "Q6_K");
        assert_eq!(quality_kind("blk.5.ffn_up_exps.weight"), "Q5_K");
        assert_eq!(quality_kind("blk.9.ffn_gate_exps.weight"), "Q5_K");
        assert_eq!(quality_kind("blk.10.ffn_down_exps.weight"), "IQ4_XS");
        assert_eq!(quality_kind("blk.20.ffn_gate_exps.weight"), "IQ4_XS");
        assert_eq!(quality_kind("blk.29.ffn_up_exps.weight"), "IQ4_XS");
        assert_eq!(quality_kind("blk.30.ffn_down_exps.weight"), "Q5_K");
        assert_eq!(quality_kind("blk.35.ffn_gate_exps.weight"), "Q6_K");
        assert_eq!(quality_kind("ffn_gate_exps.weight"), "Q6_K");
        assert_eq!(quality_kind("model.layers.0.mlp.experts.3.gate_proj.weight"), "Q6_K");
        assert_eq!(quality_kind("model.layers.20.mlp.experts.1.down_proj.weight"), "IQ4_XS");
        assert_eq!(quality_kind("model.layers.20.mlp.shared_expert.up_proj.weight"), "Q8_0");
        assert_eq!(quality_kind("blk.0.ffn_gate_shexp.weight"), "Q8_0");
        assert_eq!(quality_kind("blk.20.ffn_down_shexp.weight"), "Q8_0");
        assert_eq!(quality_kind("blk.0.attn_q.weight"), "Q6_K");
        assert_eq!(quality_kind("blk.20.attn_output.weight"), "Q6_K");
        assert_eq!(quality_kind("blk.20.attn_qkv.weight"), "Q6_K");
        assert_eq!(quality_kind("model.layers.10.self_attn.o_proj.weight"), "Q6_K");
        assert_eq!(quality_kind("blk.12.ssm_out.weight"), "Q6_K");
        assert_eq!(quality_kind("blk.2.shortconv.in_proj.weight"), "Q6_K");
        assert_eq!(quality_kind("blk.20.ffn_gate_inp.weight"), "Q6_K");
        assert_eq!(quality_kind("token_embd.weight"), "Q6_K");
        assert_eq!(quality_kind("output.weight"), "Q6_K");
        assert_eq!(quality_kind("lm_head.weight"), "Q6_K");
    }

    fn mini_kind(name: &str) -> String {
        assign(ApexProfile::Mini, 40, name).to_string()
    }

    #[test]
    fn mini_bands() {
        assert_eq!(mini_kind("blk.0.ffn_gate_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.4.ffn_down_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.5.ffn_up_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.9.ffn_gate_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.10.ffn_down_exps.weight"), "IQ2_S");
        assert_eq!(mini_kind("blk.20.ffn_gate_exps.weight"), "IQ2_S");
        assert_eq!(mini_kind("blk.29.ffn_up_exps.weight"), "IQ2_S");
        assert_eq!(mini_kind("blk.30.ffn_down_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.34.ffn_gate_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.35.ffn_gate_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("ffn_gate_exps.weight"), "Q3_K");
        assert_eq!(mini_kind("model.layers.0.mlp.experts.3.gate_proj.weight"), "Q3_K");
        assert_eq!(mini_kind("model.layers.20.mlp.experts.1.down_proj.weight"), "IQ2_S");
        assert_eq!(mini_kind("blk.0.ffn_gate_shexp.weight"), "Q5_K");
        assert_eq!(mini_kind("blk.4.ffn_down_shexp.weight"), "Q5_K");
        assert_eq!(mini_kind("blk.5.ffn_up_shexp.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.20.ffn_down_shexp.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.34.ffn_gate_shexp.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.35.ffn_gate_shexp.weight"), "Q5_K");
        assert_eq!(mini_kind("ffn_gate_shexp.weight"), "Q5_K");
        assert_eq!(mini_kind("model.layers.20.mlp.shared_expert.up_proj.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.0.attn_q.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.2.attn_output.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.3.attn_q.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.20.attn_output.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.20.attn_qkv.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.36.attn_q.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.37.attn_q.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.39.attn_output.weight"), "Q4_K");
        assert_eq!(mini_kind("model.layers.10.self_attn.o_proj.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.12.ssm_out.weight"), "Q3_K");
        assert_eq!(mini_kind("blk.2.shortconv.in_proj.weight"), "Q4_K");
        assert_eq!(mini_kind("attn_q.weight"), "Q4_K");
        assert_eq!(mini_kind("blk.20.ffn_gate_inp.weight"), "Q6_K");
        assert_eq!(mini_kind("token_embd.weight"), "Q6_K");
        assert_eq!(mini_kind("output.weight"), "Q6_K");
        assert_eq!(mini_kind("lm_head.weight"), "Q6_K");
    }
}
