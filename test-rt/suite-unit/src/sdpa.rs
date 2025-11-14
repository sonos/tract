use infra::{Test, TestSuite};
use proptest::{
    prelude::{any, Arbitrary, BoxedStrategy, Just, Strategy},
    prop_oneof,
};
use tract_core::internal::*;
use tract_core::ndarray::{arr2, arr3, ArrayD, ArrayView4};
use tract_core::num_traits::Float;
use tract_ndarray::{s, Array2, Array4, ArrayView2, Axis, Ix3, Ix4, IxDyn};
use tract_transformers::ops::sdpa::Sdpa;

use crate::tensor;

#[derive(Debug, Clone)]
pub struct SdpaProblemParams {
    pub embed_dims: Vec<usize>,
}

impl Default for SdpaProblemParams {
    fn default() -> SdpaProblemParams {
        SdpaProblemParams { embed_dims: vec![1, 2, 3] }
    }
}

#[derive(Debug, Clone)]
pub struct SdpaProblem<F>
where
    F: Datum + Float,
{
    q: ArrayD<F>,
    pub k: ArrayD<F>,
    v: ArrayD<F>,
    mask: Option<Array2<F>>,
    scale: Option<f32>,
    is_causal: bool,
}

impl<F> Arbitrary for SdpaProblem<F>
where
    F: Datum + Float,
{
    type Parameters = SdpaProblemParams;
    type Strategy = BoxedStrategy<SdpaProblem<F>>;

    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        prop_oneof![
            generate_3d_single_head::<F>(params.clone()),
            generate_4d_group_query_att::<F>(params, 4, 4)
        ]
        .boxed()
    }
}

fn generate_3d_single_head<F: Datum + Float>(
    params: SdpaProblemParams,
) -> BoxedStrategy<SdpaProblem<F>> {
    use tract_ndarray::Axis;
    generate_4d_group_query_att::<F>(params, 1, 1)
        .prop_map(|mut gqa| {
            gqa.q.index_axis_inplace(Axis(1), 0);
            gqa.k.index_axis_inplace(Axis(1), 0);
            gqa.v.index_axis_inplace(Axis(1), 0);
            gqa
        })
        .boxed()
}

fn sdpa_tensor<F: Datum + Float>(shape: &[usize]) -> BoxedStrategy<ArrayD<F>> {
    let len = shape.iter().product::<usize>();
    let shape: Vec<usize> = shape.into();
    proptest::collection::vec(
        (-80i8..=80i8).prop_map(|i| F::from(i as f32 / 100f32).unwrap()),
        len..=len,
    )
    .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
    .boxed()
}

fn generate_4d_group_query_att<F: Datum + Float>(
    params: SdpaProblemParams,
    max_heads_repeat_factor: usize,
    max_kv_heads: usize,
) -> BoxedStrategy<SdpaProblem<F>> {
    (
        1..3usize,
        1..max_heads_repeat_factor + 1,
        1..max_kv_heads + 1,
        0..5usize,
        2..5usize,
        0..params.embed_dims.len(),
    )
        .prop_flat_map(move |(b, repeat_factor, n_kv_heads, past_seq_len, seq_len, embed_idx)| {
            let embed = params.embed_dims[embed_idx];
            let n_q_heads = repeat_factor * n_kv_heads;
            let q = sdpa_tensor::<F>(&[b, n_q_heads, seq_len, embed]);
            let k = sdpa_tensor::<F>(&[b, n_kv_heads, past_seq_len + seq_len, embed]);
            let v = sdpa_tensor::<F>(&[b, n_kv_heads, past_seq_len + seq_len, embed]);

            let scale_strategy = prop_oneof![Just(None), (0.1f32..1.0).prop_map(Some)];
            let causal_strategy = any::<bool>();
            (Just(past_seq_len), Just(seq_len), q, k, v, scale_strategy, causal_strategy)
        })
        .prop_flat_map(|(past_seq_len, seq_len, q, k, v, scale, is_causal)| {
            let mask_strategy = if is_causal {
                Just(None).boxed()
            } else {
                prop_oneof![Just(None), tensor(&[seq_len, past_seq_len + seq_len]).prop_map(Some)]
                    .boxed()
            };

            (Just(q), Just(k), Just(v), Just(scale), Just(is_causal), mask_strategy)
        })
        .prop_map(|(q, k, v, scale, is_causal, mask)| SdpaProblem {
            q,
            k,
            v,
            mask: mask.map(|m| m.into_dimensionality().unwrap()),
            scale,
            is_causal,
        })
        .boxed()
}

impl<F> SdpaProblem<F>
where
    F: Datum + Float + Copy + 'static,
{
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

        let dt = q.datum_type();
        let output = model.wire_node(
            "SDPA",
            Sdpa {
                scale,
                datum_type: dt,
                acc_datum_type: DatumType::F32,
                is_causal: self.is_causal,
            },
            &inputs,
        )?;
        model.set_output_outlets(&output)?;
        model.into_decluttered()
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
        queries: &ArrayView2<F>,
        keys: &ArrayView2<F>,
        values: &ArrayView2<F>,
        mask: Option<&Array2<f32>>,
        scale: Option<f32>,
        is_causal: bool,
    ) -> Array2<F> {
        let d_k = keys.shape()[1] as f32;
        let scale_factor = scale.unwrap_or(1.0 / d_k.sqrt());
        let queries = queries.mapv(|q| q * F::from(scale_factor).unwrap());

        let queries_f32 = queries.mapv(|q| q.to_f32().unwrap());
        let keys_f32 = keys.mapv(|k| k.to_f32().unwrap());
        let values_f32 = values.mapv(|v| v.to_f32().unwrap());

        let mut scaled_input = queries_f32.dot(&keys_f32.t());

        if is_causal {
            let (q_len, k_len) = (queries.nrows(), keys.nrows());
            let p = k_len.saturating_sub(q_len);
            scaled_input.indexed_iter_mut().for_each(|((r, c), z)| {
                if c > p + r {
                    *z = f32::NEG_INFINITY;
                }
            });
        }

        if let Some(m) = mask {
            scaled_input += m;
        }

        let att_weights = Self::softmax(&scaled_input, 1);
        att_weights.dot(&values_f32).mapv(|r| F::from(r).unwrap())
    }

    fn reference_4d(&self, q: ArrayView4<F>, k: ArrayView4<F>, v: ArrayView4<F>) -> Array4<F> {
        let [b, q_heads, seq_len, _] = q.shape() else { unreachable!() };
        let [_, kv_heads, _, v_emb] = v.shape() else { unreachable!() };
        let mut output = Array4::<F>::zeros((*b, *q_heads, *seq_len, *v_emb));
        let repeat_factor = q_heads / kv_heads;

        let mask_f32: Option<Array2<f32>> =
            self.mask.as_ref().map(|m| m.mapv(|x| x.to_f32().unwrap()));

        for batch_idx in 0..*b {
            for kv_head_idx in 0..*kv_heads {
                for q_head_idx_in_group in 0..repeat_factor {
                    let q_head_idx = q_head_idx_in_group + repeat_factor * kv_head_idx;

                    let q_slice = q.slice(s![batch_idx, q_head_idx, .., ..]);
                    let k_slice = k.slice(s![batch_idx, kv_head_idx, .., ..]);
                    let v_slice = v.slice(s![batch_idx, kv_head_idx, .., ..]);

                    let out2 = Self::scaled_dot_product_attention_2d(
                        &q_slice,
                        &k_slice,
                        &v_slice,
                        mask_f32.as_ref(),
                        self.scale, // still f32
                        self.is_causal,
                    );
                    output.slice_mut(s![batch_idx, q_head_idx, .., ..]).assign(&out2);
                }
            }
        }
        output
    }

    fn reference(&self) -> TractResult<ArrayD<F>> {
        match self.q.ndim() {
            3 => {
                let q = self.q.view().into_dimensionality::<Ix3>()?.insert_axis(Axis(1));
                let k = self.k.view().into_dimensionality::<Ix3>()?.insert_axis(Axis(1));
                let v = self.v.view().into_dimensionality::<Ix3>()?.insert_axis(Axis(1));
                let out_4d = self.reference_4d(q, k, v);
                Ok(out_4d.remove_axis(Axis(1)).into_dyn())
            }
            4 => {
                let q = self.q.view().into_dimensionality::<Ix4>().unwrap();
                let k = self.k.view().into_dimensionality::<Ix4>().unwrap();
                let v = self.v.view().into_dimensionality::<Ix4>().unwrap();
                Ok(self.reference_4d(q, k, v).into_dyn())
            }
            _ => unreachable!(),
        }
    }
}

impl<F> Test for SdpaProblem<F>
where
    F: Datum + Float + Copy + 'static,
{
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn tract_core::runtime::Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let reference = self.reference()?.into_tensor();
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

    suite.add_arbitrary::<SdpaProblem<f32>>("proptest_f32", SdpaProblemParams::default());
    suite.add_arbitrary::<SdpaProblem<f16>>("proptest_f16", SdpaProblemParams::default());
    suite.add(
        "trivial_f32_0",
        SdpaProblem {
            q: tensor3(&[[[0f32]]]).into_array::<f32>()?,
            k: tensor3(&[[[0f32]]]).into_array::<f32>()?,
            v: tensor3(&[[[0f32]]]).into_array::<f32>()?,
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "causal_f32_0",
        SdpaProblem {
            q: arr3(&[[[0f32]]]).into_dyn(),
            k: arr3(&[[[0f32]]]).into_dyn(),
            v: arr3(&[[[0f32]]]).into_dyn(),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "causal_f32_1",
        SdpaProblem {
            q: arr3(&[[[0f32], [0.]]]).into_dyn(),
            k: arr3(&[[[0f32], [0.]]]).into_dyn(),
            v: arr3(&[[[0f32], [0.]]]).into_dyn(),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "causal_f32_2",
        SdpaProblem {
            q: arr3(&[[[0f32], [0f32]]]).into_dyn(),
            k: arr3(&[[[0f32], [0f32]]]).into_dyn(),
            v: arr3(&[[[0f32], [0f32]]]).into_dyn(),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "causal_with_s_and_p_0",
        SdpaProblem {
            q: arr3(&[[[0f32], [0.0]]]).into_dyn(),
            k: arr3(&[[[0f32], [0f32], [0f32]]]).into_dyn(),
            v: arr3(&[[[0f32], [0f32], [1f32]]]).into_dyn(),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_0",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[1, 2, 1, 1])),
            k: ArrayD::<f32>::zeros(IxDyn(&[1, 2, 1, 1])),
            v: arr4(&[[[[0f32]], [[1f32]]]]).into_dyn(),
            mask: None,
            scale: None,
            is_causal: false,
        },
    );
    suite.add(
        "gqa_f32_1",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[1, 2, 1, 1])),
            k: ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1, 1])),
            v: ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1, 1])),
            mask: None,
            scale: None,
            is_causal: false,
        },
    );
    suite.add(
        "gqa_f32_big_0",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 8, 5, 64])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 4, 5, 64])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 4, 5, 64])),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_big_1",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 8, 5, 64])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 5, 64])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 5, 64])),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_big_2",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 2, 3, 64])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 64])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 64])),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "mask_0",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[1, 2, 1])),
            k: ArrayD::<f32>::zeros(IxDyn(&[1, 2, 1])),
            v: arr3(&[[[2f32], [0.]]]).into_dyn(),
            mask: Some(arr2(&[[0.0f32, 0.0], [0.0, -1.0]])),
            scale: None,
            is_causal: false,
        },
    );
    suite.add(
        "gqa_f32_mask_simple",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1])),
            k: ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1])),
            v: ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1])),
            mask: Some(arr2(&[[0f32]])),
            scale: None,
            is_causal: false,
        },
    );
    suite.add(
        "gqa_f32_mask_0",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[2, 2, 3, 64])),
            k: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 64])),
            v: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 3, 64])),
            mask: Some(Array2::<f32>::zeros([3, 3])),
            scale: None,
            is_causal: false,
        },
    );
    suite.add(
        "gqa_f16_0",
        SdpaProblem {
            q: ArrayD::<f16>::zeros(IxDyn(&[1, 2, 64])),
            k: ArrayD::<f16>::zeros(IxDyn(&[1, 3, 64])),
            v: ArrayD::<f16>::zeros(IxDyn(&[1, 3, 64])),
            mask: None,
            scale: None,
            is_causal: true,
        },
    );
    suite.add(
        "gqa_f32_nocausal_nomask",
        SdpaProblem {
            q: ArrayD::<f32>::zeros(IxDyn(&[1, 1, 2, 1])),
            k: ArrayD::<f32>::zeros(IxDyn(&[1, 1, 2, 1])),
            v: arr4(&[[[[0f32], [1f32]]]]).into_dyn(),
            mask: None,
            scale: None,
            is_causal: false,
        },
    );
    Ok(suite)
}
