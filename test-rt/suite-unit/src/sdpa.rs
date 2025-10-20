use infra::{Test, TestSuite};
use proptest::{
    prelude::{any, Arbitrary, BoxedStrategy, Just, Strategy},
    prop_oneof,
};
use tract_core::internal::*;
use tract_core::ndarray::{ArrayD, ArrayView4};
use tract_core::num_traits::Float;
use tract_ndarray::{s, Array2, Array4, ArrayView2, Axis, Ix2, Ix3, Ix4, IxDyn};
use tract_transformers::ops::sdpa::Sdpa;

use crate::tensor;

pub trait SdpaAlgo: Datum + Copy + 'static {
    fn sdp_2d(
        q: &ArrayView2<Self>,          
        k: &ArrayView2<Self>,          
        v: &ArrayView2<Self>,          
        mask: Option<ArrayView2<Self>>,
        scale: Option<f32>,            
        is_causal: bool,
    ) -> Array2<Self>;
}

impl SdpaAlgo for f32 {
    fn sdp_2d(
        q: &ArrayView2<f32>, k: &ArrayView2<f32>, v: &ArrayView2<f32>,
        mask: Option<ArrayView2<f32>>, scale: Option<f32>, is_causal: bool,
    ) -> Array2<f32> {
        let d = q.ncols() as f32;
        let s = scale.unwrap_or(1.0 / d.sqrt());

        let mut logits = q.dot(&k.t());   // [Q,K]
        logits *= s;                      // f32 supports scalar mul

        if is_causal {
            let (q_len, k_len) = (q.nrows(), k.nrows());
            let p = k_len.saturating_sub(q_len);
            logits.indexed_iter_mut().for_each(|((r,c), z)| {
                if c > p + r { *z = f32::NEG_INFINITY; }
            });
        }
        if let Some(m) = mask { logits += &m; }

        for mut row in logits.axis_iter_mut(Axis(0)) {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !m.is_finite() { row.fill(0.0); continue; }
            for x in row.iter_mut() { *x = (*x - m).exp(); }
            let s = row.iter().sum::<f32>().max(1e-20);
            for x in row.iter_mut() { *x /= s; }
        }
        logits.dot(v)
    }
}

impl SdpaAlgo for f16 {
    fn sdp_2d(
        q: &ArrayView2<f16>, k: &ArrayView2<f16>, v: &ArrayView2<f16>,
        mask: Option<ArrayView2<f16>>, scale: Option<f32>, is_causal: bool,
    ) -> Array2<f16> {
        let (q_len, d)   = (q.nrows(), q.ncols());
        let (k_len, d_k) = (k.nrows(), k.ncols());
        assert_eq!(d, d_k);
        let v_dim = v.ncols();

        let s_h = f16::from_f32(scale.unwrap_or(1.0 / (d as f32).sqrt()));
        //println!("Scale: {}", s_h);
        // logits in f16 with f16 accumulation
        let mut logits = Array2::<f16>::from_elem((q_len, k_len), f16::from_f32(0.0));
        for i in 0..q_len {
            for j in 0..k_len {
                let mut acc = f16::from_f32(0.0);
                for t in 0..d {
                    acc = acc + q[(i,t)] * k[(j,t)];
                }
                logits[(i,j)] = acc * s_h;
            }
        }

        //println!("Scaled_scores: {:?}", &logits);
        if is_causal {
            let p = k_len.saturating_sub(q_len);
            for i in 0..q_len {
                for j in (p + i + 1)..k_len {
                    logits[(i,j)] = f16::NEG_INFINITY;
                }
            }
        }
        if let Some(m) = mask {
            assert_eq!(m.dim(), logits.dim());
            for i in 0..q_len { for j in 0..k_len { logits[(i,j)] = logits[(i,j)] + m[(i,j)]; } }
        }
        //println!("Masked_scores: {:?}", &logits);
        // softmax in f16
        let mut att = Array2::<f16>::from_elem((q_len, k_len), f16::from_f32(0.0));
        for i in 0..q_len {
            let mut m = f16::NEG_INFINITY;
            for j in 0..k_len { let v = logits[(i,j)]; if v > m { m = v; } }
            if !m.to_f32().is_finite() { continue; }

            let mut s = f16::from_f32(0.0);
            for j in 0..k_len {
                let e = (logits[(i,j)] - m).exp();
                att[(i,j)] = e;
                s = s + e;
            }
            if s.to_f32() == 0.0 { continue; }
            for j in 0..k_len { att[(i,j)] = att[(i,j)] / s; }
        }
        //println!("Post Softmax: {:?}", &att);
        // att @ V in f16
        let mut out = Array2::<f16>::from_elem((q_len, v_dim), f16::from_f32(0.0));
        for i in 0..q_len {
            for vv in 0..v_dim {
                let mut acc = f16::from_f32(0.0);
                for kk in 0..k_len {
                    acc = acc + att[(i,kk)] * v[(kk,vv)];
                }
                out[(i,vv)] = acc;
            }
        }
        //println!("Output: {:?}", &out);
        out
    }
}

#[derive(Debug, Clone)]
pub struct SdpaProblemParams {
    pub embed_dims: Vec<usize>
}

impl Default for SdpaProblemParams {
    fn default() -> SdpaProblemParams {
        SdpaProblemParams {
            embed_dims: vec![1, 2, 3]
        }
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
        prop_oneof![generate_3d_single_head::<F>(params.clone()), generate_4d_group_query_att::<F>(params, 1, 1)].boxed()
    }
}

fn generate_3d_single_head<F: Datum + Float>(params: SdpaProblemParams) -> BoxedStrategy<SdpaProblem<F>> {
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

fn generate_4d_group_query_att<F: Datum + Float>(
    params: SdpaProblemParams,
    max_heads_repeat_factor: usize,
    max_kv_heads: usize,
) -> BoxedStrategy<SdpaProblem<F>> {
    (1..2usize, 1..max_heads_repeat_factor + 1, 1..max_kv_heads + 1, 0..3usize, 2..5usize, 0..params.embed_dims.len())
        .prop_flat_map(move |(b, repeat_factor, n_kv_heads, past_seq_len, seq_len, embed_idx)| {
            let embed = params.embed_dims[embed_idx];
            let n_q_heads = repeat_factor * n_kv_heads;
            let q = tensor::<F>(&[b, n_q_heads, seq_len, embed]);
            let k = tensor::<F>(&[b, n_kv_heads, past_seq_len + seq_len, embed]);
            let v = tensor::<F>(&[b, n_kv_heads, past_seq_len + seq_len, embed]);

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
            mask,
            scale,
            is_causal,
        })
        .boxed()
}

impl<F> SdpaProblem<F>
    where
        F: Datum + Float + SdpaAlgo + Copy + 'static,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let q = self.q.clone().into_tensor();
        let k = self.k.clone().into_tensor();
        let v = self.v.clone().into_tensor();

        let scale = self.scale.map(|s| tensor0(s));

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
                acc_datum_type: dt,
                is_causal: self.is_causal,
            },
            &inputs,
        )?;
        model.set_output_outlets(&output)?;
        Ok(model.into_decluttered()?)
    }

    fn reference_4d(
        &self,
        q: ArrayView4<F>,
        k: ArrayView4<F>,
        v: ArrayView4<F>,
        mask: Option<ArrayView2<F>>,
    ) -> Array4<F> {
        let [b, q_heads, seq_len, _] = q.shape() else { unreachable!() };
        let [_, kv_heads, _, v_emb] = v.shape() else { unreachable!() };
        let mut output = Array4::<F>::zeros((*b, *q_heads, *seq_len, *v_emb));
        let repeat_factor = q_heads / kv_heads;

        for batch_idx in 0..*b {
            for kv_head_idx in 0..*kv_heads {
                for q_head_idx_in_group in 0..repeat_factor {
                    let q_head_idx = q_head_idx_in_group + repeat_factor * kv_head_idx;

                    let q_slice = q.slice(s![batch_idx, q_head_idx, .., ..]);
                    let k_slice = k.slice(s![batch_idx, kv_head_idx, .., ..]);
                    let v_slice = v.slice(s![batch_idx, kv_head_idx, .., ..]);

                    let out2 = F::sdp_2d(
                            &q_slice,
                            &k_slice,
                            &v_slice,
                            mask,
                            self.scale,       // still f32
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
                let mask = self
                    .mask
                    .as_ref()
                    .map(|m| m.view().into_dimensionality::<Ix2>().unwrap());
                let out_4d = self.reference_4d(q, k, v, mask);
                Ok(out_4d.remove_axis(Axis(1)).into_dyn())
            }
            4 => {
                let q = self.q.view().into_dimensionality::<Ix4>().unwrap();
                let k = self.k.view().into_dimensionality::<Ix4>().unwrap();
                let v = self.v.view().into_dimensionality::<Ix4>().unwrap();
                let mask =
                    self.mask.as_ref().map(|m| m.view().into_dimensionality::<Ix2>().unwrap());
                Ok(self.reference_4d(q, k, v, mask).into_dyn())
            }
            _ => unreachable!(),
        }
    }
}

impl<F> Test for SdpaProblem<F>
where
        F: Datum + Float + SdpaAlgo + Copy + 'static,
{
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn tract_core::runtime::Runtime,
        approx: tract_core::internal::Approximation,
    ) -> infra::TestResult {
        let reference = self.reference()?.into_tensor();
        let mut model = self.tract()?;

        model
            .properties
            .insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

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
            mask: Some(ArrayD::<f32>::zeros(IxDyn(&[3, 3]))),
            scale: None,
            is_causal: false,
        },
    );
    Ok(suite)
}
