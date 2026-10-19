//! APEX balanced assignment: which block-quant format a matmul weight gets.
//!
//! Bands follow apex-quant `generate_config.sh` for a model of `layers` blocks
//! (default 40). Edge experts are layers `0..=4` and the last 5, near-edge is
//! `5..=9` and the 5 layers before the trailing edge, and the rest are middle.
//! Attention uses a tighter edge: the first 3 and last 3 layers.
//!
//! Balanced (and I-balanced, which differs only in the imatrix this pass does
//! not consume) is the tier whose types tract can run: routed experts are
//! Q6_K on the edge and Q5_K elsewhere, shared experts are Q8_0, attention and
//! every other matmul weight are Q6_K.
use tract_linalg::block_quant::{BlockQuant, Q5_K, Q6_K, Q8_0};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApexProfile {
    Balanced,
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
    let role = role(name);
    let band = layer_index(name).map(|layer| band(profile, role, layer, layers));
    let format = match (profile, role, band) {
        (ApexProfile::Balanced, Role::Shared, _) => &Q8_0 as &dyn BlockQuant,
        (ApexProfile::Balanced, Role::Routed, Some(Band::Edge) | None) => &Q6_K,
        (ApexProfile::Balanced, Role::Routed, Some(Band::Near | Band::Mid)) => &Q5_K,
        (ApexProfile::Balanced, _, _) => &Q6_K,
    };
    dyn_clone::clone_box(format)
}

fn band(profile: ApexProfile, role: Role, layer: usize, layers: usize) -> Band {
    let _ = profile;
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
}
