use infra::{Test, TestResult, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0};
use tract_transformers::ops::moe_ffn::{ExpertLayout, GateMode, MoeFfn};

#[derive(Clone, Copy, Debug, Default)]
enum Storage {
    #[default]
    Float,
    Q40,
    Q40WithF16Down,
}

#[derive(Clone, Copy, Debug, Default)]
enum WeightInputs {
    #[default]
    Constants,
    Router,
    All,
}

#[derive(Clone, Debug)]
struct MoeProblem {
    x: Tensor,
    wg: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Option<Tensor>,
    k: usize,
    gate: GateMode,
    linear: bool,
    storage: Storage,
    weight_inputs: WeightInputs,
}

fn tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
    let shape = shape.to_vec();
    vec(-8i8..=8, shape.iter().product::<usize>())
        .prop_map(move |values| {
            Tensor::from_shape(&shape, &values.iter().map(|&v| v as f32 / 32.0).collect::<Vec<_>>())
                .unwrap()
        })
        .boxed()
}

impl Arbitrary for MoeProblem {
    type Parameters = (bool, Storage, WeightInputs);
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((linear, storage, weight_inputs): Self::Parameters) -> Self::Strategy {
        let dimension = if matches!(storage, Storage::Float) { 1usize..33 } else { 1usize..3 };
        (1usize..17, dimension.clone(), dimension, 1usize..9, any::<bool>(), 0usize..4)
            .prop_flat_map(move |(tokens, d, h, experts, has_w3, gate)| {
                let (d, h) =
                    if matches!(storage, Storage::Float) { (d, h) } else { (d * 32, h * 32) };
                let up = if linear { [experts, h, d] } else { [experts, d, h] };
                let down = if linear { [experts, d, h] } else { [experts, h, d] };
                (
                    tensor(&[tokens, d]),
                    tensor(&[experts, d]),
                    tensor(&up),
                    tensor(&down),
                    if has_w3 { tensor(&up).prop_map(Some).boxed() } else { Just(None).boxed() },
                    1..=experts,
                    Just(gate),
                )
            })
            .prop_map(move |(x, wg, w1, w2, w3, k, gate)| Self {
                x,
                wg,
                w1,
                w2,
                w3,
                k,
                gate: match gate {
                    0 => GateMode::SoftmaxTopk,
                    1 => GateMode::SoftmaxAll,
                    2 => GateMode::Sigmoid,
                    _ => GateMode::Raw,
                },
                linear,
                storage,
                weight_inputs,
            })
            .boxed()
    }
}

fn weight(tensor: &Tensor, storage: Storage, down: bool) -> TractResult<(Tensor, Tensor)> {
    if matches!(storage, Storage::Float) {
        return Ok((tensor.clone(), tensor.clone()));
    }
    if down && matches!(storage, Storage::Q40WithF16Down) {
        let half = tensor.cast_to::<f16>()?.into_owned();
        let plain = half.cast_to::<f32>()?.into_owned();
        return Ok((half, plain));
    }
    let shape = tensor.shape();
    let k = shape[2];
    let quant = Q4_0.quant_f32(tensor.try_as_plain_ram()?.as_slice::<f32>()?)?;
    let plain = Q4_0.dequant_f32(&quant)?.into_shape(shape)?;
    let storage = BlockQuantStorage::new(Box::new(Q4_0), shape[0] * shape[1], k, Arc::new(quant))?;
    Ok((storage.into_tensor_with_shape(DatumType::F32, shape), plain))
}

fn add_weight(model: &mut TypedModel, name: &str, tensor: Tensor) -> TractResult<OutletId> {
    if tensor.storage_as::<BlockQuantStorage>().is_some() {
        let fact = BlockQuantFact::new(Box::new(Q4_0), tensor.shape().iter().copied().collect());
        Ok(model.wire_node(
            name,
            Const::new_with_exotic_fact(Arc::new(tensor), Box::new(fact))?,
            &[],
        )?[0])
    } else {
        model.add_const(name, tensor)
    }
}

impl MoeProblem {
    fn reference(&self, w1: &Tensor, w2: &Tensor, w3: Option<&Tensor>) -> TractResult<Tensor> {
        let x = self.x.to_plain_array_view::<f32>()?;
        let wg = self.wg.to_plain_array_view::<f32>()?;
        let w1 = w1.to_plain_array_view::<f32>()?;
        let w2 = w2.to_plain_array_view::<f32>()?;
        let w3 = w3.map(|w| w.to_plain_array_view::<f32>()).transpose()?;
        let (tokens, d, experts) = (x.shape()[0], x.shape()[1], wg.shape()[0]);
        let h = w1.shape()[if self.linear { 1 } else { 2 }];
        let mut output = vec![0f32; tokens * d];
        for t in 0..tokens {
            let logits: Vec<f64> = (0..experts)
                .map(|e| (0..d).map(|i| f64::from(x[[t, i]]) * f64::from(wg[[e, i]])).sum())
                .collect();
            let mut selected: Vec<usize> = (0..experts).collect();
            selected.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            selected.truncate(self.k);
            let denominator: f64 = match self.gate {
                GateMode::SoftmaxTopk => selected.iter().map(|&e| logits[e].exp()).sum(),
                GateMode::SoftmaxAll => logits.iter().map(|l| l.exp()).sum(),
                _ => 1.0,
            };
            for e in selected {
                let scale = match self.gate {
                    GateMode::SoftmaxTopk | GateMode::SoftmaxAll => logits[e].exp() / denominator,
                    GateMode::Sigmoid => 1.0 / (1.0 + (-logits[e]).exp()),
                    GateMode::Raw => logits[e],
                };
                let mut hidden = vec![0f64; h];
                for (j, value) in hidden.iter_mut().enumerate() {
                    let project = |weights: &tract_ndarray::ArrayViewD<'_, f32>| -> f64 {
                        (0..d)
                            .map(|i| {
                                let index = if self.linear { [e, j, i] } else { [e, i, j] };
                                f64::from(x[[t, i]]) * f64::from(weights[index])
                            })
                            .sum()
                    };
                    let gate = project(&w1);
                    *value = gate / (1.0 + (-gate).exp());
                    if let Some(up) = &w3 {
                        *value *= project(up);
                    }
                }
                for i in 0..d {
                    let value: f64 = hidden
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            let index = if self.linear { [e, i, j] } else { [e, j, i] };
                            v * f64::from(w2[index])
                        })
                        .sum();
                    output[t * d + i] += (scale * value) as f32;
                }
            }
        }
        Tensor::from_shape(self.x.shape(), &output)
    }
}

impl Test for MoeProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let (w1, ref_w1) = weight(&self.w1, self.storage, false)?;
        let (w2, ref_w2) = weight(&self.w2, self.storage, true)?;
        let w3 = self.w3.as_ref().map(|w| weight(w, self.storage, false)).transpose()?;
        let expected = self.reference(&ref_w1, &ref_w2, w3.as_ref().map(|(_, plain)| plain))?;
        let mut model = TypedModel::default();
        let tokens = model.symbols.sym("tokens");
        let x = model.add_source("x", f32::fact([tokens.to_dim(), self.x.shape()[1].to_dim()]))?;
        let mut extra_inputs = tvec![];
        let wg = if matches!(self.weight_inputs, WeightInputs::Constants) {
            model.add_const("wg", self.wg.clone())?
        } else {
            extra_inputs.push(self.wg.clone().into_tvalue());
            model.add_source("wg", TypedFact::shape_and_dt_of(&self.wg))?
        };
        let mut expert = |model: &mut TypedModel, name: &str, tensor: Tensor| {
            if matches!(self.weight_inputs, WeightInputs::All) {
                let outlet = model.add_source(name, TypedFact::shape_and_dt_of(&tensor))?;
                extra_inputs.push(tensor.into_tvalue());
                Ok(outlet)
            } else {
                add_weight(model, name, tensor)
            }
        };
        let w1 = expert(&mut model, "w1", w1)?;
        let w2 = expert(&mut model, "w2", w2)?;
        let mut inputs = tvec![x, wg, w1, w2];
        if let Some((w3, _)) = w3 {
            inputs.push(expert(&mut model, "w3", w3)?);
        }
        let op = MoeFfn {
            k: self.k,
            activation: "silu".into(),
            gate: self.gate.clone(),
            has_w3: self.w3.is_some(),
            has_wg_bias: false,
            has_w1_bias: false,
            has_w3_bias: false,
            has_w2_bias: false,
            act_alpha_bits: None,
            act_limit_bits: None,
            expert_layout: if self.linear { ExpertLayout::Linear } else { ExpertLayout::Canonical },
        };
        let output = model.wire_node("moe", op, &inputs)?;
        model.select_output_outlets(&output)?;
        model.properties.insert("tract-rt-test.id".into(), rctensor0(id.to_string()));
        let run_inputs = |x: Tensor| {
            let mut inputs = tvec![x.into_tvalue()];
            inputs.extend(extra_inputs.iter().cloned());
            inputs
        };
        let mut state = runtime.prepare(model)?.spawn()?;
        let actual = state.run(run_inputs(self.x.clone()))?;
        actual[0].close_enough(&expected, approx)?;
        let mut cloned = tract_core::dyn_clone::clone_box(&*state);
        let larger = Tensor::zero::<f32>(&[self.x.shape()[0] + 1, self.x.shape()[1]])?;
        for state in [&mut state, &mut cloned] {
            let actual = state.run(run_inputs(larger.clone()))?;
            actual[0].close_enough(&larger, approx)?;
            let actual = state.run(run_inputs(self.x.clone()))?;
            actual[0].close_enough(&expected, approx)?;
        }
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    for (name, storage) in
        [("float", Storage::Float), ("q40", Storage::Q40), ("mixed", Storage::Q40WithF16Down)]
    {
        let mut layouts = TestSuite::default();
        layouts.add_arbitrary::<MoeProblem>("canonical", (false, storage, WeightInputs::Constants));
        layouts.add_arbitrary::<MoeProblem>("linear", (true, storage, WeightInputs::Constants));
        suite.add(name, layouts);
    }
    suite.add_arbitrary::<MoeProblem>(
        "cached_routes",
        (false, Storage::Float, WeightInputs::Router),
    );
    suite.add_arbitrary::<MoeProblem>("dynamic_routes", (false, Storage::Float, WeightInputs::All));
    Ok(suite)
}
