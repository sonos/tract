use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::BlockQuant;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::BlockQuantStorage;
use tract_core::tract_linalg::block_quant::Q2_0_T;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_ndarray::Array2;
use tract_ndarray::Axis;
use tract_transformers::ops::moe_ffn::ExpertLayout;
use tract_transformers::ops::moe_ffn::GateMode;
use tract_transformers::ops::moe_ffn::MoeFfn;
use tract_transformers::ops::moe_ffn::RouteSelect;
use tract_transformers::ops::moe_ffn::RouteTopK;
use tract_transformers::ops::moe_ffn::RoutedActivation;
use tract_transformers::ops::moe_ffn::RoutedInputMode;
use tract_transformers::ops::moe_ffn::RoutedLinear;

#[derive(Clone, Copy, Debug)]
enum Format {
    F32,
    F16,
    Q40,
    Q20T,
}

fn fill(shape: &[usize], salt: usize) -> Tensor {
    let len = shape.iter().product::<usize>();
    let data = (0..len).map(|i| ((i * 7919 + salt * 104729) % 199) as f32 / 99.0 - 1.0);
    Tensor::from_shape(shape, &data.collect::<Vec<_>>()).unwrap()
}

/// The weight as the op sees it, and its values as f32.
fn weight(format: Format, shape: &[usize], salt: usize) -> TractResult<(Tensor, Tensor)> {
    let values = fill(shape, salt);
    let bq: &dyn BlockQuant = match format {
        Format::F32 => return Ok((values.clone(), values)),
        Format::F16 => {
            let half = values.cast_to::<f16>()?.into_owned();
            let plain = half.cast_to::<f32>()?.into_owned();
            return Ok((half, plain));
        }
        Format::Q40 => &Q4_0,
        Format::Q20T => &Q2_0_T,
    };
    let k = shape[shape.len() - 1];
    let quant = bq.quant_f32(values.try_as_plain_ram()?.as_slice::<f32>()?)?;
    let plain = bq.dequant_f32(&quant)?.into_shape(shape)?;
    let rows = shape.iter().product::<usize>() / k;
    let storage =
        BlockQuantStorage::new(tract_core::dyn_clone::clone_box(bq), rows, k, Arc::new(quant))?;
    Ok((storage.into_tensor_with_shape(DatumType::F32, shape), plain))
}

fn add_weight(model: &mut TypedModel, name: &str, tensor: Tensor) -> TractResult<OutletId> {
    if let Some(bqs) = tensor.storage_as::<BlockQuantStorage>() {
        let fact = BlockQuantFact::new(
            tract_core::dyn_clone::clone_box(bqs.format()),
            tensor.shape().into(),
        );
        Ok(model.wire_node(
            name,
            Const::new_with_exotic_fact(Arc::new(tensor), Box::new(fact))?,
            &[],
        )?[0])
    } else {
        model.add_const(name, tensor)
    }
}

/// A [`RoutedLinear`] against a plain f32 evaluation of its contract.
#[derive(Clone, Debug)]
struct RoutedLinearProblem {
    format: Format,
    mode: RoutedInputMode,
    pair: bool,
    bias: bool,
    activation: RoutedActivation,
    /// Enough routes per expert for a backend's prefill path, and a partial output tile.
    prefill: bool,
}

impl RoutedLinearProblem {
    const EXPERTS: usize = 4;

    fn n_k(&self) -> (usize, usize) {
        if self.prefill { (70, 96) } else { (24, 64) }
    }

    fn tokens(&self) -> usize {
        if self.prefill { 75 } else { 3 }
    }

    fn ids(&self) -> (Vec<i64>, Vec<i64>) {
        if !self.prefill {
            return (vec![0, 2, 1, 2, 0, 1, 2], vec![3, 0, 3, 1, 0, 3, 2]);
        }
        let routes = self.tokens() * 8;
        let tokens = (0..routes).map(|r| ((r * 37) % self.tokens()) as i64).collect();
        let experts = (0..routes).map(|r| ((r * 7 + r / 5) % Self::EXPERTS) as i64).collect();
        (tokens, experts)
    }

    fn rows(&self) -> usize {
        if self.mode == RoutedInputMode::TokenRows { self.tokens() } else { self.ids().0.len() }
    }
}

fn activate(activation: &RoutedActivation, v: f32) -> f32 {
    match activation {
        RoutedActivation::Silu => v / (1.0 + (-v).exp()),
        RoutedActivation::Gelu => {
            0.5 * v
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
        }
        RoutedActivation::Relu => v.max(0.0),
        _ => v,
    }
}

impl Test for RoutedLinearProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let (e, (n, k)) = (Self::EXPERTS, self.n_k());
        let (token_ids, expert_ids) = self.ids();
        let x = fill(&[self.rows(), k], 1);
        let mut model = TypedModel::default();
        let source = model.add_source("x", f32::fact([self.rows(), k]))?;
        let tokens = model.add_const("token_ids", tensor1(&token_ids))?;
        let experts = model.add_const("expert_ids", tensor1(&expert_ids))?;
        let mut inputs = tvec!(source, tokens, experts);
        let mut plain_weights = vec![];
        for p in 0..1 + self.pair as usize {
            let (w, plain) = weight(self.format, &[e, n, k], 2 + p)?;
            inputs.push(add_weight(&mut model, &format!("w{p}"), w)?);
            plain_weights.push(plain);
        }
        let mut biases = vec![];
        if self.bias {
            for p in 0..1 + self.pair as usize {
                let b = fill(&[e, n], 5 + p);
                inputs.push(model.add_const(format!("b{p}"), b.clone())?);
                biases.push(b);
            }
        }
        let op = RoutedLinear {
            input_mode: self.mode,
            pair: self.pair,
            bias: self.bias,
            activation: self.activation,
        };
        let output = model.wire_node("routed", op, &inputs)?;
        model.select_output_outlets(&output)?;
        model.properties.insert("tract-rt-test.id".into(), rctensor0(id.to_string()));

        let xa = x.to_plain_array_view::<f32>()?.into_dimensionality::<tract_ndarray::Ix2>()?;
        let mut expected = Array2::<f32>::zeros((token_ids.len(), n));
        for (r, mut out) in expected.axis_iter_mut(Axis(0)).enumerate() {
            let row =
                if self.mode == RoutedInputMode::TokenRows { token_ids[r] as usize } else { r };
            let expert = expert_ids[r] as usize;
            let project = |p: usize| -> TractResult<Vec<f32>> {
                let w = plain_weights[p].to_plain_array_view::<f32>()?;
                Ok((0..n)
                    .map(|j| {
                        let dot = (0..k).map(|i| xa[(row, i)] * w[[expert, j, i]]).sum::<f32>();
                        dot + biases
                            .get(p)
                            .map(|b| b.to_plain_array_view::<f32>().unwrap()[[expert, j]])
                            .unwrap_or(0.0)
                    })
                    .collect())
            };
            let gate = project(0)?;
            let values: Vec<f32> = if self.pair {
                let up = project(1)?;
                gate.iter()
                    .zip(&up)
                    .map(|(&g, &u)| match self.activation {
                        RoutedActivation::ClampedSwiGlu { alpha_bits, limit_bits } => {
                            let (alpha, limit) =
                                (f32::from_bits(alpha_bits), f32::from_bits(limit_bits));
                            let g = g.min(limit);
                            (u.clamp(-limit, limit) + 1.0) * g / (1.0 + (-alpha * g).exp())
                        }
                        act => activate(&act, g) * u,
                    })
                    .collect()
            } else {
                gate.iter().map(|&g| activate(&self.activation, g)).collect()
            };
            out.assign(&tract_ndarray::ArrayView1::from(&values));
        }
        let actual = runtime.prepare(model)?.run(tvec!(x.into_tvalue()))?;
        actual[0].close_enough(&expected.into_tensor(), approx)
    }
}

/// A [`RoutedLinear`] reading an out-of-range route must fail rather than read past its input.
#[derive(Clone, Debug)]
struct BoundsProblem {
    token: i64,
    expert: i64,
    mode: RoutedInputMode,
}

impl Test for BoundsProblem {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TestResult {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([2, 32]))?;
        let tokens = model.add_const("token", tensor1(&[self.token]))?;
        let experts = model.add_const("expert", tensor1(&[self.expert]))?;
        let w = model.add_const("w", Tensor::zero::<f32>(&[2, 3, 32])?)?;
        let op = RoutedLinear {
            input_mode: self.mode,
            pair: false,
            bias: false,
            activation: RoutedActivation::None,
        };
        let output = model.wire_node("routed", op, &[x, tokens, experts, w])?;
        model.select_output_outlets(&output)?;
        let result = runtime
            .prepare(model)
            .and_then(|plan| plan.run(tvec!(Tensor::zero::<f32>(&[2, 32])?.into_tvalue())));
        let token_ok = self.mode == RoutedInputMode::RouteRows || (0..2).contains(&self.token);
        if token_ok && (0..2).contains(&self.expert) {
            result?;
        } else {
            ensure!(result.is_err(), "out-of-range route was accepted");
        }
        Ok(())
    }
}

/// [`RouteSelect`] on router scores against [`RouteTopK`] on the same router. Every other
/// token's scores are quantized coarsely so that experts tie.
#[derive(Clone, Debug)]
struct RouteSelectProblem {
    experts: usize,
    k: usize,
    bias: bool,
    gate: GateMode,
}

impl Test for RouteSelectProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let (tokens, d) = (6, 8);
        let mut x = fill(&[tokens, d], 1);
        let wg = fill(&[self.experts, d], 2);
        for (t, mut row) in x.to_plain_array_view_mut::<f32>()?.outer_iter_mut().enumerate() {
            if t % 2 == 1 {
                row.mapv_inplace(|v| (v * 2.0).round() / 2.0);
            }
        }
        let bias = self.bias.then(|| fill(&[self.experts], 3));

        let mut reference = TypedModel::default();
        let rx = reference.add_source("x", f32::fact([tokens, d]))?;
        let mut inputs = tvec!(rx, reference.add_const("wg", wg.clone())?);
        if let Some(b) = &bias {
            inputs.push(reference.add_const("bias", b.clone())?);
        }
        let outputs = reference.wire_node(
            "topk",
            RouteTopK { k: self.k, gate: self.gate.clone() },
            &inputs,
        )?;
        reference.select_output_outlets(&outputs)?;
        let expected = SimplePlan::new(reference)?.run(tvec!(x.clone().into_tvalue()))?;

        let xa =
            x.to_plain_array_view::<f32>()?.into_dimensionality::<tract_ndarray::Ix2>()?.to_owned();
        let wga = wg
            .to_plain_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix2>()?
            .to_owned();
        let scores = xa.dot(&wga.t()).into_tensor();
        let mut model = TypedModel::default();
        let source = model.add_source("scores", f32::fact([tokens, self.experts]))?;
        let mut inputs = tvec!(source);
        if let Some(b) = bias {
            inputs.push(model.add_const("bias", b)?);
        }
        let outputs = model.wire_node(
            "select",
            RouteSelect { k: self.k, gate: self.gate.clone() },
            &inputs,
        )?;
        model.select_output_outlets(&outputs)?;
        model.properties.insert("tract-rt-test.id".into(), rctensor0(id.to_string()));
        let actual = runtime.prepare(model)?.run(tvec!(scores.into_tvalue()))?;
        actual[0].close_enough(&expected[0], Approximation::Exact)?;
        actual[1].close_enough(&expected[1], Approximation::Exact)?;
        actual[2].close_enough(&expected[2], approx)
    }
}

/// A [`MoeFfn`] rewritten by `routed_linear_patch` against the op itself, unoptimized.
#[derive(Clone, Debug)]
struct DecompositionProblem {
    format: Format,
    layout: ExpertLayout,
    activation: &'static str,
    has_w3: bool,
    clamped: bool,
    biases: bool,
    gate: GateMode,
}

impl Test for DecompositionProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let (e, d, h, tokens) = (4, 64, 32, 5);
        let linear = self.layout == ExpertLayout::Linear;
        let (w13_shape, w2_shape) =
            if linear { ([e, h, d], [e, d, h]) } else { ([e, d, h], [e, h, d]) };
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, tokens, d]))?;
        let wg = model.add_const("wg", fill(&[e, d], 9))?;
        let w1 = add_weight(&mut model, "w1", weight(self.format, &w13_shape, 2)?.0)?;
        let w2 = add_weight(&mut model, "w2", weight(self.format, &w2_shape, 3)?.0)?;
        let mut inputs = tvec!(x, wg, w1, w2);
        if self.has_w3 {
            inputs.push(add_weight(&mut model, "w3", weight(self.format, &w13_shape, 4)?.0)?);
        }
        if self.biases {
            inputs.push(model.add_const("wg_bias", fill(&[e], 10))?);
            inputs.push(model.add_const("w1_bias", fill(&[e, h], 11))?);
            if self.has_w3 {
                inputs.push(model.add_const("w3_bias", fill(&[e, h], 12))?);
            }
            inputs.push(model.add_const("w2_bias", fill(&[e, d], 13))?);
        }
        let op = MoeFfn {
            k: 2,
            activation: self.activation.into(),
            gate: self.gate.clone(),
            has_w3: self.has_w3,
            has_wg_bias: self.biases,
            has_w1_bias: self.biases,
            has_w3_bias: self.biases && self.has_w3,
            has_w2_bias: self.biases,
            act_alpha_bits: self.clamped.then_some(1.702f32.to_bits()),
            act_limit_bits: self.clamped.then_some(1.5f32.to_bits()),
            expert_layout: self.layout,
        };
        let output = model.wire_node("moe", op.clone(), &inputs)?;
        model.select_output_outlets(&output)?;

        let x_value = fill(&[1, tokens, d], 1);
        let expected = SimplePlan::new(model.clone())?.run(tvec!(x_value.clone().into_tvalue()))?;
        let node = model.node(output[0].node);
        let patch = op.routed_linear_patch(&model, node)?.context("MoeFfn was not decomposed")?;
        patch.apply(&mut model)?;
        ensure!(!model.nodes().iter().any(|n| n.op_is::<MoeFfn>()), "MoeFfn left in the graph");
        model.properties.insert("tract-rt-test.id".into(), rctensor0(id.to_string()));
        let actual = runtime.prepare(model)?.run(tvec!(x_value.into_tvalue()))?;
        actual[0].close_enough(&expected[0], approx)
    }
}

/// The full grid of activations, input modes, pairs and biases runs on f32 weights. Other
/// formats only change how weights are read, so a few cases between them cover both input
/// modes, single and paired projections, biases, a plain activation and the clamped SwiGLU.
pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    let clamped = RoutedActivation::ClampedSwiGlu {
        alpha_bits: 1.702f32.to_bits(),
        limit_bits: 1.5f32.to_bits(),
    };
    let activations = [
        RoutedActivation::None,
        RoutedActivation::Silu,
        RoutedActivation::Gelu,
        RoutedActivation::Relu,
        clamped,
    ];
    let mut add_sized =
        |format: Format, mode, pair, bias, activation: RoutedActivation, prefill| {
            let act = format!("{activation:?}").split(' ').next().unwrap().to_lowercase();
            let size = if prefill { "_prefill" } else { "" };
            suite.add(
                format!("{format:?}_{mode:?}_pair{pair}_bias{bias}_{act}{size}").to_lowercase(),
                RoutedLinearProblem { format, mode, pair, bias, activation, prefill },
            );
        };
    for format in [Format::F32, Format::F16, Format::Q40, Format::Q20T] {
        add_sized(format, RoutedInputMode::TokenRows, true, true, RoutedActivation::Silu, true);
        add_sized(format, RoutedInputMode::RouteRows, false, false, RoutedActivation::None, true);
    }
    let mut add = |format, mode, pair, bias, activation| {
        add_sized(format, mode, pair, bias, activation, false)
    };
    for mode in [RoutedInputMode::TokenRows, RoutedInputMode::RouteRows] {
        for pair in [false, true] {
            for bias in [false, true] {
                for activation in activations {
                    if pair || activation != clamped {
                        add(Format::F32, mode, pair, bias, activation);
                    }
                }
            }
        }
    }
    for format in [Format::F16, Format::Q40, Format::Q20T] {
        add(format, RoutedInputMode::TokenRows, true, true, RoutedActivation::Silu);
        add(format, RoutedInputMode::TokenRows, true, false, clamped);
        add(format, RoutedInputMode::RouteRows, false, false, RoutedActivation::None);
        add(format, RoutedInputMode::RouteRows, false, true, RoutedActivation::Gelu);
    }
    for mode in [RoutedInputMode::TokenRows, RoutedInputMode::RouteRows] {
        for (token, expert) in [(0, 0), (-1, 0), (2, 0), (0, -1), (0, 2)] {
            suite.add(
                format!("bounds_{mode:?}_token{token}_expert{expert}")
                    .replace('-', "neg")
                    .to_lowercase(),
                BoundsProblem { token, expert, mode },
            );
        }
    }
    let gates = [GateMode::SoftmaxTopk, GateMode::SoftmaxAll, GateMode::Sigmoid, GateMode::Raw];
    let mut selects: Vec<(usize, usize, bool, GateMode)> = vec![];
    for (ix, gate) in gates.iter().enumerate() {
        selects.push((4, 1, ix % 2 == 0, gate.clone()));
        selects.push((40, 4, ix % 2 == 1, gate.clone()));
    }
    selects.push((300, 2, true, GateMode::SoftmaxAll));
    for (experts, k, bias, gate) in selects {
        suite.add(
            format!("select_e{experts}_k{k}_bias{bias}_{gate:?}").to_lowercase(),
            RouteSelectProblem { experts, k, bias, gate },
        );
    }
    let shapes = [
        ("silu", false, false, false),
        ("swiglu", true, false, false),
        ("gelu", true, false, true),
        ("relu", false, false, true),
        ("swiglu", true, true, true),
    ];
    let mut moes = vec![(Format::F32, ExpertLayout::Linear, &shapes[..])];
    moes.push((Format::F32, ExpertLayout::Canonical, &shapes[1..3]));
    for format in [Format::F16, Format::Q40, Format::Q20T] {
        moes.push((format, ExpertLayout::Linear, &shapes[2..]));
    }
    for (format, layout, shapes) in moes {
        for (ix, &(activation, has_w3, clamped, biases)) in shapes.iter().enumerate() {
            let gate = if ix % 2 == 0 { GateMode::SoftmaxTopk } else { GateMode::Sigmoid };
            suite.add(
                format!(
                    "moe_{format:?}_{layout:?}_{activation}_w3{has_w3}_clamped{clamped}_bias{biases}_{gate:?}"
                )
                .to_lowercase(),
                DecompositionProblem { format, layout, activation, has_w3, clamped, biases, gate },
            );
        }
    }
    Ok(suite)
}
