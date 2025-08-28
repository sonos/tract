use infra::{Test, TestSuite};
use proptest::{
    prelude::{any, Arbitrary, BoxedStrategy, Just, Strategy},
    prop_oneof,
};
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_ndarray::{s, Array2, Array3, Array4, ArrayView2, Ix3, Ix4, IxDyn};
use tract_transformers::ops::sdpa::Sdpa;

use crate::tensor;

#[derive(Clone, Debug)]
pub struct SdpaProblemParams {
    q_emb: usize,
}

impl Default for SdpaProblemParams {
    fn default() -> Self {
        Self { q_emb: 32 }
    }
}

#[derive(Debug, Clone)]
pub struct SdpaProblem<F>
where
    F: Datum + Float,
{
    q: ArrayD<F>,
    k: ArrayD<F>,
    v: ArrayD<F>,
    mask: Option<ArrayD<F>>,
    scale: Option<F>,
    is_causal: bool,
}

impl Arbitrary for SdpaProblem<f32> {
    type Parameters = SdpaProblemParams;
    type Strategy = BoxedStrategy<SdpaProblem<f32>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        prop_oneof![generate_3d_problem(args.clone()), generate_4d_problem(args.clone()),].boxed()
    }
}

fn generate_3d_problem(args: SdpaProblemParams) -> BoxedStrategy<SdpaProblem<f32>> {
    (1..3usize, 1..16usize)
        .prop_flat_map(move |(b, seq_len)| {
            let q = tensor(&[b, seq_len, args.q_emb]);
            let k = tensor(&[b, seq_len, args.q_emb]);
            let v = tensor(&[b, seq_len, args.q_emb]);

            let scale_strategy = prop_oneof![Just(None), (0.1f32..2.0).prop_map(Some)];
            let causal_strategy = any::<bool>();
            (Just((b, seq_len)), q, k, v, scale_strategy, causal_strategy)
        })
        .prop_flat_map(|((b, seq_len), q, k, v, scale, is_causal)| {
            let mask_strategy = if is_causal {
                Just(None).boxed()
            } else {
                prop_oneof![Just(None), tensor(&[b, seq_len, seq_len]).prop_map(Some)].boxed()
            };

            (Just(q), Just(k), Just(v), Just(scale), Just(is_causal), mask_strategy)
        })
        .prop_map(|(q, k, v, scale, is_causal, mask)| SdpaProblem {
            q,
            k,
            v,
            mask,
            scale,
            is_causal,
        })
        .boxed()
}

fn generate_4d_problem(args: SdpaProblemParams) -> BoxedStrategy<SdpaProblem<f32>> {
    let head_strategy = (1..=4usize).prop_map(|i| i * 2).prop_flat_map(|q_heads| {
        let kv_heads_strategy = (1..=q_heads)
            .prop_filter("q_heads should be a multiple of q_emb", move |kh| q_heads % *kh == 0);
        (Just(q_heads), kv_heads_strategy)
    });

    head_strategy
        .prop_flat_map(move |(q_heads, kv_heads)| {
            let head_dim = args.q_emb / q_heads;
            (1..3usize, 1..16usize).prop_flat_map(move |(b, seq_len)| {
                let q = tensor(&[b, q_heads, seq_len, head_dim]);
                let k = tensor(&[b, kv_heads, seq_len, head_dim]);
                let v = tensor(&[b, kv_heads, seq_len, head_dim]);

                let scale_strategy = prop_oneof![Just(None), (0.1f32..2.0).prop_map(Some)];
                let causal_strategy = any::<bool>();
                (Just((b, q_heads, seq_len)), q, k, v, scale_strategy, causal_strategy)
            })
        })
        .prop_flat_map(|((b, q_heads, seq_len), q, k, v, scale, is_causal)| {
            let mask_strategy = if is_causal {
                Just(None).boxed()
            } else {
                prop_oneof![Just(None), tensor(&[b, q_heads, seq_len, seq_len]).prop_map(Some)]
                    .boxed()
            };

            (Just(q), Just(k), Just(v), Just(scale), Just(is_causal), mask_strategy)
        })
        .prop_map(|(q, k, v, scale, is_causal, mask)| SdpaProblem {
            q,
            k,
            v,
            mask,
            scale,
            is_causal,
        })
        .boxed()
}

impl SdpaProblem<f32> {
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let q = self.q.clone().into_tensor();
        let k = self.k.clone().into_tensor();
        let v = self.v.clone().into_tensor();
        let scale = self.scale.map(tensor0);
        let q_in = model.add_source("Q", TypedFact::shape_and_dt_of(&q))?;
        let k_in = model.add_source("K", TypedFact::shape_and_dt_of(&k))?;
        let v_in = model.add_source("V", TypedFact::shape_and_dt_of(&v))?;
        let mut inputs = vec![q_in, k_in, v_in];
        if let Some(mask) = &self.mask {
            let mask_in = model
                .add_source("mask", TypedFact::shape_and_dt_of(&mask.clone().into_tensor()))?;
            inputs.push(mask_in);
        }
        let output = model.wire_node(
            "SDPA",
            Sdpa {
                scale,
                datum_type: DatumType::F32,
                acc_datum_type: DatumType::F32,
                is_causal: self.is_causal,
                subgraph: None,
            },
            &inputs,
        )?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn softmax(input: &Array2<f32>, axis: usize) -> Array2<f32> {
        let axis = tract_ndarray::Axis(axis);

        let max_per_axis =
            input.map_axis(axis, |lane| lane.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));

        let shifted = input - &max_per_axis.insert_axis(axis);
        let exp = shifted.mapv(f32::exp);
        let sum_exp = exp.sum_axis(axis);

        let norm = sum_exp.insert_axis(axis);

        &exp / &norm
    }

    fn scaled_dot_product_attention_2d(
        queries: &ArrayView2<f32>,
        keys: &ArrayView2<f32>,
        values: &ArrayView2<f32>,
        mask: Option<ArrayView2<f32>>,
        scale: Option<f32>,
        is_causal: bool,
    ) -> Array2<f32> {
        let d_k = keys.shape()[1] as f32;
        let scale_factor = scale.unwrap_or(1.0 / d_k.sqrt());
        let q_dot_kt = queries.dot(&keys.t());
        let mut scaled_input = q_dot_kt * scale_factor;

        if is_causal {
            scaled_input.indexed_iter_mut().for_each(|((r, c), value)| {
                if c > r {
                    *value = f32::NEG_INFINITY;
                }
            });
        }

        if let Some(m) = mask {
            scaled_input += &m;
        }

        let att_weights = Self::softmax(&scaled_input, 1);
        att_weights.dot(values)
    }

    fn reference_3d(&self) -> Array3<f32> {
        let q = self.q.view().into_dimensionality::<Ix3>().unwrap();
        let k = self.k.view().into_dimensionality::<Ix3>().unwrap();
        let v = self.v.view().into_dimensionality::<Ix3>().unwrap();
        let mask = self.mask.as_ref().map(|m| m.view().into_dimensionality::<Ix3>().unwrap());

        let [b, seq_len, _] = q.shape() else { unreachable!() };
        let [_, _, v_emb] = v.shape() else { unreachable!() };
        let mut output = Array3::<f32>::zeros((*b, *seq_len, *v_emb));

        for i in 0..*b {
            let q_slice = &q.slice(s![i, .., ..]);
            let k_slice = &k.slice(s![i, .., ..]);
            let v_slice = &v.slice(s![i, .., ..]);
            let mask_slice = mask.as_ref().map(|m| m.slice(s![i, .., ..]));
            let output_2d = Self::scaled_dot_product_attention_2d(
                q_slice,
                k_slice,
                v_slice,
                mask_slice,
                self.scale,
                self.is_causal,
            );
            output.slice_mut(s![i, .., ..]).assign(&output_2d);
        }
        output
    }

    fn reference_4d(&self) -> Array4<f32> {
        let q = self.q.view().into_dimensionality::<Ix4>().unwrap();
        let k = self.k.view().into_dimensionality::<Ix4>().unwrap();
        let v = self.v.view().into_dimensionality::<Ix4>().unwrap();
        let mask = self.mask.as_ref().map(|m| m.view().into_dimensionality::<Ix4>().unwrap());

        let [b, q_heads, seq_len, _] = q.shape() else { unreachable!() };
        let [_, kv_heads, _, v_emb] = v.shape() else { unreachable!() };
        let mut output = Array4::<f32>::zeros((*b, *q_heads, *seq_len, *v_emb));
        let repeat_factor = q_heads / kv_heads;

        for batch_idx in 0..*b {
            for q_head_idx in 0..*q_heads {
                let kv_head_idx = q_head_idx / repeat_factor;

                let q_slice = q.slice(s![batch_idx, q_head_idx, .., ..]);
                let k_slice = k.slice(s![batch_idx, kv_head_idx, .., ..]);
                let v_slice = v.slice(s![batch_idx, kv_head_idx, .., ..]);
                let mask_slice = mask.as_ref().map(|m| m.slice(s![batch_idx, q_head_idx, .., ..]));

                let output_2d = Self::scaled_dot_product_attention_2d(
                    &q_slice,
                    &k_slice,
                    &v_slice,
                    mask_slice,
                    self.scale,
                    self.is_causal,
                );
                output.slice_mut(s![batch_idx, q_head_idx, .., ..]).assign(&output_2d);
            }
        }
        output
    }

    fn reference(&self) -> ArrayD<f32> {
        match self.q.ndim() {
            3 => self.reference_3d().into_dyn(),
            4 => self.reference_4d().into_dyn(),
            _ => unreachable!(),
        }
    }
}

impl Test for SdpaProblem<f32> {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn tract_core::runtime::Runtime,
        approx: tract_core::internal::Approximation,
    ) -> infra::TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut inputs = tvec![
            self.q.clone().into_tvalue(),
            self.k.clone().into_tvalue(),
            self.v.clone().into_tvalue()
        ];
        if let Some(mask) = &self.mask {
            inputs.push(mask.clone().into_tvalue());
        }
        let mut output = runtime.prepare(model)?.run(inputs)?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    let params = SdpaProblemParams::default();
    suite.add_arbitrary::<SdpaProblem<f32>>("proptest_f32", params.clone());
    suite.add(
        "trivial_f32_0",
        SdpaProblem {
            q: tensor3(&[[[0f32]]]).into_array()?,
            k: tensor3(&[[[0f32]]]).into_array()?,
            v: tensor3(&[[[0f32]]]).into_array()?,
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_0",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 8, 5, 16])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 4, 5, 16])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 4, 5, 16])),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_1",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 8, 5, 16])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 5, 16])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 5, 16])),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_2",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 2, 3, 16])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 16])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 16])),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_mask",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 2, 3, 16])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 16])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 16])),
            mask: Some(ArrayD::<f32>::zeros(IxDyn(&[2, 2, 3, 3]))),
            scale: None,
            is_causal: false,
        },
    );
    Ok(suite)
}
