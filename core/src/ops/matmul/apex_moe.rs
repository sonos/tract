//! `apex_moe` model transform. Quantizes constant matmul weights with an APEX
//! recipe. `balanced` (and `i-balanced`, same types) assigns routed experts
//! Q6_K on edge layers and Q5_K elsewhere, shared experts Q8_0, and attention
//! plus every other matmul weight Q6_K. `dense-hybrid-quality` assigns a dense
//! model's gate, up, and down separately, pins full attention at Q8_0, and
//! puts linear attention and SSM projections at Q4_K. `compact` (and
//! `i-compact`, same types) assigns routed experts Q4_K on the edge and Q3_K
//! elsewhere, shared experts Q6_K, and attention Q4_K. `quality` (and
//! `i-quality`, same types) matches balanced except middle routed experts,
//! which are IQ4_XS.
//!
//! A weight is left in its original dtype when its contraction axis is not a
//! multiple of the chosen format's block (32 for Q8_0, 256 for the K-quants and IQ4_XS).
//! `layers` is the block count the edge bands are computed from; the default
//! is 40, which is the depth the published APEX configs were written for.
use crate::internal::*;
use crate::ops::matmul::apex_policy::ApexProfile;
use crate::ops::matmul::de_block_quant::{self, QuantMode};
use crate::transform::ModelTransform;

fn default_profile() -> String {
    "balanced".into()
}

fn default_layers() -> usize {
    40
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ApexMoeConfig {
    #[serde(default = "default_profile")]
    pub profile: String,
    #[serde(default = "default_layers")]
    pub layers: usize,
}

impl Default for ApexMoeConfig {
    fn default() -> Self {
        ApexMoeConfig { profile: default_profile(), layers: default_layers() }
    }
}

#[derive(Debug)]
pub struct ApexMoeTransform {
    profile: ApexProfile,
    layers: usize,
}

impl ApexMoeTransform {
    pub fn from_config(config: ApexMoeConfig) -> TractResult<Box<dyn ModelTransform>> {
        let profile = match config.profile.as_str() {
            "balanced" | "i-balanced" => ApexProfile::Balanced,
            "dense-hybrid-quality" => ApexProfile::DenseHybridQuality,
            "compact" | "i-compact" => ApexProfile::Compact,
            "quality" | "i-quality" => ApexProfile::Quality,
            other => {
                bail!(
                    "apex_moe profile '{other}' is not implemented (available: balanced, i-balanced, dense-hybrid-quality, compact, i-compact, quality, i-quality)"
                )
            }
        };
        Ok(Box::new(ApexMoeTransform { profile, layers: config.layers }))
    }
}

impl ModelTransform for ApexMoeTransform {
    fn name(&self) -> StaticName {
        "apex_moe".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        de_block_quant::apply_weight_quant(
            model,
            &QuantMode::Apex { profile: self.profile, layers: self.layers },
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::einsum::EinSum;
    use crate::ops::konst::Const;
    use tract_linalg::block_quant::BlockQuantFact;

    fn fill(shape: &[usize], seed: usize) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> =
            (0..n).map(|i| (((i * 13 + seed * 7) % 29) as f32 - 14.0) / 14.0).collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    fn quant_name(
        profile: ApexProfile,
        weight_node: &str,
        w_shape: &[usize],
    ) -> TractResult<Option<String>> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([4, w_shape[0]]))?;
        let w =
            model.wire_node(weight_node, Const::new(fill(w_shape, 3).into_arc_tensor())?, &[])?[0];
        let out = model.wire_node(
            "mm",
            EinSum { axes: "mk,kn->mn".parse()?, operating_dt: f32::datum_type(), q_params: None },
            &[x, w],
        )?;
        model.select_output_outlets(&out)?;
        let mut model = model.into_decluttered()?;
        ApexMoeTransform { profile, layers: 40 }.transform(&mut model)?;
        let Some(node) = model.nodes().iter().find(|n| n.name.ends_with(".bq")) else {
            return Ok(None);
        };
        let fact = model.outlet_fact(node.id.into())?;
        let bqf = fact
            .exotic_fact()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .context("bq node has no BlockQuantFact")?;
        Ok(Some(bqf.format.to_string()))
    }

    #[test]
    fn assigns_by_role_and_skips_short_k() -> TractResult<()> {
        assert_eq!(
            quant_name(ApexProfile::Balanced, "blk.0.ffn_gate_exps.weight", &[256, 32])?.as_deref(),
            Some("Q6_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Balanced, "blk.20.ffn_up_exps.weight", &[256, 32])?.as_deref(),
            Some("Q5_K")
        );
        assert_eq!(
            quant_name(
                ApexProfile::Balanced,
                "model.layers.12.mlp.shared_expert.down_proj.weight",
                &[256, 32]
            )?
            .as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            quant_name(ApexProfile::Balanced, "lm_head.weight", &[256, 32])?.as_deref(),
            Some("Q6_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Balanced, "blk.20.ffn_gate_exps.weight", &[64, 32])?,
            None
        );
        assert_eq!(
            quant_name(ApexProfile::DenseHybridQuality, "blk.0.ffn_down.weight", &[256, 32])?
                .as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            quant_name(ApexProfile::DenseHybridQuality, "blk.20.ffn_up.weight", &[256, 32])?
                .as_deref(),
            Some("Q5_K")
        );
        assert_eq!(
            quant_name(ApexProfile::DenseHybridQuality, "blk.20.attn_qkv.weight", &[256, 32])?
                .as_deref(),
            Some("Q4_K")
        );
        assert_eq!(
            quant_name(ApexProfile::DenseHybridQuality, "blk.20.attn_output.weight", &[256, 32])?
                .as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            quant_name(ApexProfile::DenseHybridQuality, "token_embd.weight", &[256, 32])?
                .as_deref(),
            Some("Q5_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Compact, "blk.0.ffn_gate_exps.weight", &[256, 32])?.as_deref(),
            Some("Q4_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Compact, "blk.20.ffn_up_exps.weight", &[256, 32])?.as_deref(),
            Some("Q3_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Compact, "blk.20.ffn_gate_shexp.weight", &[256, 32])?
                .as_deref(),
            Some("Q6_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Compact, "blk.20.attn_q.weight", &[256, 32])?.as_deref(),
            Some("Q4_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Quality, "blk.0.ffn_gate_exps.weight", &[256, 32])?.as_deref(),
            Some("Q6_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Quality, "blk.5.ffn_up_exps.weight", &[256, 32])?.as_deref(),
            Some("Q5_K")
        );
        assert_eq!(
            quant_name(ApexProfile::Quality, "blk.20.ffn_down_exps.weight", &[256, 32])?.as_deref(),
            Some("IQ4_XS")
        );
        assert_eq!(
            quant_name(ApexProfile::Quality, "blk.20.ffn_gate_shexp.weight", &[256, 32])?
                .as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            quant_name(ApexProfile::Quality, "blk.20.attn_q.weight", &[256, 32])?.as_deref(),
            Some("Q6_K")
        );
        Ok(())
    }

    #[test]
    fn q6k_matmul_matches_dequant() -> TractResult<()> {
        use tract_linalg::block_quant::{BlockQuant, Q6_K};
        let x_shape = &[4usize, 256];
        let w_shape = &[256usize, 32];
        let x = fill(x_shape, 1);
        let w = fill(w_shape, 2);
        let w_deq = Q6_K.simulate_precision_loss(w.clone().move_axis(0, 1)?, 1)?.move_axis(1, 0)?;

        let eval = |weight: &Tensor| -> TractResult<Tensor> {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in = model.wire_node(
                "blk.0.ffn_gate_exps.weight",
                Const::new(weight.clone().into_arc_tensor())?,
                &[],
            )?[0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            let model = model.into_decluttered()?;
            let out = model.into_runnable()?.run(tvec!(x.clone().into_tvalue()))?;
            Ok(out[0].clone().into_tensor())
        };

        let reference = eval(&w_deq)?;
        let mut quant = {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in = model.wire_node(
                "blk.0.ffn_gate_exps.weight",
                Const::new(w.into_arc_tensor())?,
                &[],
            )?[0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            model.into_decluttered()?
        };
        ApexMoeTransform { profile: ApexProfile::Balanced, layers: 40 }.transform(&mut quant)?;
        let got = quant.into_runnable()?.run(tvec!(x.into_tvalue()))?[0].clone().into_tensor();
        got.close_enough(&reference, Approximation::Approximate)
    }

    #[test]
    fn q4k_linear_attn_matches_dequant() -> TractResult<()> {
        use tract_linalg::block_quant::{BlockQuant, Q4_K};
        let x_shape = &[4usize, 256];
        let w_shape = &[256usize, 32];
        let x = fill(x_shape, 1);
        let w = fill(w_shape, 2);
        let w_deq = Q4_K.simulate_precision_loss(w.clone().move_axis(0, 1)?, 1)?.move_axis(1, 0)?;

        let eval = |weight: &Tensor| -> TractResult<Tensor> {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in = model.wire_node(
                "blk.20.attn_qkv.weight",
                Const::new(weight.clone().into_arc_tensor())?,
                &[],
            )?[0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            let model = model.into_decluttered()?;
            let out = model.into_runnable()?.run(tvec!(x.clone().into_tvalue()))?;
            Ok(out[0].clone().into_tensor())
        };

        let reference = eval(&w_deq)?;
        let mut quant = {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in =
                model.wire_node("blk.20.attn_qkv.weight", Const::new(w.into_arc_tensor())?, &[])?
                    [0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            model.into_decluttered()?
        };
        ApexMoeTransform { profile: ApexProfile::DenseHybridQuality, layers: 40 }
            .transform(&mut quant)?;
        let got = quant.into_runnable()?.run(tvec!(x.into_tvalue()))?[0].clone().into_tensor();
        got.close_enough(&reference, Approximation::Approximate)
    }

    #[test]
    fn q3k_matmul_matches_dequant() -> TractResult<()> {
        use tract_linalg::block_quant::{BlockQuant, Q3_K};
        let x_shape = &[4usize, 256];
        let w_shape = &[256usize, 32];
        let x = fill(x_shape, 1);
        let w = fill(w_shape, 2);
        let w_deq = Q3_K.simulate_precision_loss(w.clone().move_axis(0, 1)?, 1)?.move_axis(1, 0)?;

        let eval = |weight: &Tensor| -> TractResult<Tensor> {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in = model.wire_node(
                "blk.20.ffn_gate_exps.weight",
                Const::new(weight.clone().into_arc_tensor())?,
                &[],
            )?[0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            let model = model.into_decluttered()?;
            let out = model.into_runnable()?.run(tvec!(x.clone().into_tvalue()))?;
            Ok(out[0].clone().into_tensor())
        };

        let reference = eval(&w_deq)?;
        let mut quant = {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in = model.wire_node(
                "blk.20.ffn_gate_exps.weight",
                Const::new(w.into_arc_tensor())?,
                &[],
            )?[0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            model.into_decluttered()?
        };
        ApexMoeTransform { profile: ApexProfile::Compact, layers: 40 }.transform(&mut quant)?;
        let got = quant.into_runnable()?.run(tvec!(x.into_tvalue()))?[0].clone().into_tensor();
        got.close_enough(&reference, Approximation::Approximate)
    }

    #[test]
    fn iq4xs_matmul_matches_dequant() -> TractResult<()> {
        use tract_linalg::block_quant::{BlockQuant, IQ4_XS};
        let x_shape = &[4usize, 256];
        let w_shape = &[256usize, 32];
        let x = fill(x_shape, 1);
        let w = fill(w_shape, 2);
        let w_deq =
            IQ4_XS.simulate_precision_loss(w.clone().move_axis(0, 1)?, 1)?.move_axis(1, 0)?;

        let eval = |weight: &Tensor| -> TractResult<Tensor> {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in = model.wire_node(
                "blk.20.ffn_down_exps.weight",
                Const::new(weight.clone().into_arc_tensor())?,
                &[],
            )?[0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            let model = model.into_decluttered()?;
            let out = model.into_runnable()?.run(tvec!(x.clone().into_tvalue()))?;
            Ok(out[0].clone().into_tensor())
        };

        let reference = eval(&w_deq)?;
        let mut quant = {
            let mut model = TypedModel::default();
            let x_in = model.add_source("x", f32::fact(x_shape))?;
            let w_in = model.wire_node(
                "blk.20.ffn_down_exps.weight",
                Const::new(w.into_arc_tensor())?,
                &[],
            )?[0];
            let out = model.wire_node(
                "mm",
                EinSum {
                    axes: "mk,kn->mn".parse()?,
                    operating_dt: f32::datum_type(),
                    q_params: None,
                },
                &[x_in, w_in],
            )?;
            model.select_output_outlets(&out)?;
            model.into_decluttered()?
        };
        ApexMoeTransform { profile: ApexProfile::Quality, layers: 40 }.transform(&mut quant)?;
        let got = quant.into_runnable()?.run(tvec!(x.into_tvalue()))?[0].clone().into_tensor();
        got.close_enough(&reference, Approximation::Approximate)
    }
}
