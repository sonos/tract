//! `apex_moe` model transform. Quantizes constant matmul weights with the APEX
//! balanced recipe: routed experts Q6_K on edge layers and Q5_K elsewhere,
//! shared experts Q8_0, attention and every other matmul weight Q6_K.
//!
//! A weight is left in its original dtype when its contraction axis is not a
//! multiple of the chosen format's block (32 for Q8_0, 256 for the K-quants).
//! `layers` is the block count the edge bands are computed from; the default
//! is 40, which is the depth the published APEX configs were written for.
use crate::internal::*;
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
    layers: usize,
}

impl ApexMoeTransform {
    pub fn from_config(config: ApexMoeConfig) -> TractResult<Box<dyn ModelTransform>> {
        match config.profile.as_str() {
            "balanced" | "i-balanced" => Ok(Box::new(ApexMoeTransform { layers: config.layers })),
            other => bail!(
                "apex_moe profile '{other}' is not implemented (available: balanced, i-balanced)"
            ),
        }
    }
}

impl ModelTransform for ApexMoeTransform {
    fn name(&self) -> StaticName {
        "apex_moe".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        de_block_quant::apply_weight_quant(model, &QuantMode::ApexBalanced { layers: self.layers })
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

    fn quant_name(weight_node: &str, w_shape: &[usize]) -> TractResult<Option<String>> {
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
        ApexMoeTransform { layers: 40 }.transform(&mut model)?;
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
        assert_eq!(quant_name("blk.0.ffn_gate_exps.weight", &[256, 32])?.as_deref(), Some("Q6_K"));
        assert_eq!(quant_name("blk.20.ffn_up_exps.weight", &[256, 32])?.as_deref(), Some("Q5_K"));
        assert_eq!(
            quant_name("model.layers.12.mlp.shared_expert.down_proj.weight", &[256, 32])?
                .as_deref(),
            Some("Q8_0")
        );
        assert_eq!(quant_name("lm_head.weight", &[256, 32])?.as_deref(), Some("Q6_K"));
        assert_eq!(quant_name("blk.20.ffn_gate_exps.weight", &[64, 32])?, None);
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
        ApexMoeTransform { layers: 40 }.transform(&mut quant)?;
        let got = quant.into_runnable()?.run(tvec!(x.into_tvalue()))?[0].clone().into_tensor();
        got.close_enough(&reference, Approximation::Approximate)
    }
}
