use super::{
    GateMode, as_2d_tokens, block_quant_group_tensor, plain_f32_slice, plain_i64_slice,
    router_weights_as_2d, select_routes, token_count_dim,
};
use crate::ops::routed_matmul::{
    PreparedRoutedMatMulState, RoutedInputRows, RoutedRowsInput, build_block_quant_routed_matmul,
    run_prepared_routed_matmul,
};
use tract_ndarray::{Array2, ArrayView2, ArrayView3, s};
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::OpState;
use tract_nnef::tract_core::tract_linalg::MmmDispatch;
use tract_nnef::tract_core::tract_linalg::block_quant::BlockQuantStorage;
use tract_nnef::tract_core::tract_linalg::mmm::{
    AsInputValue, FusedSpec, MMMInputValue, MatMatMul, Query,
};
use tract_nnef::tract_core::tract_linalg::pack::PackedFormat;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RouteTopK {
    pub k: usize,
    pub gate: GateMode,
}

impl Op for RouteTopK {
    fn name(&self) -> StaticName {
        "RouteTopK".into()
    }
    op_as_typed_op!();
}

impl EvalOp for RouteTopK {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (x_t, wg_t) = args_2!(inputs);
        let x_t = x_t.cast_to::<f32>()?;
        let wg_t = wg_t.cast_to::<f32>()?;
        let x = as_2d_tokens(x_t.to_plain_array_view::<f32>()?)?;
        let wg = router_weights_as_2d(wg_t.to_plain_array_view::<f32>()?)?;

        ensure!(
            x.shape()[1] == wg.shape()[1],
            "router dimension mismatch: x {:?}, wg {:?}",
            x.shape(),
            wg.shape()
        );
        ensure!(self.k <= wg.shape()[0], "top-k {} exceeds expert count {}", self.k, wg.shape()[0]);

        let t_tokens = x.shape()[0];
        let route_count = t_tokens * self.k;
        let mut route_token_ids = Vec::with_capacity(route_count);
        let mut route_expert_ids = Vec::with_capacity(route_count);
        let mut route_weights = Vec::with_capacity(route_count);

        let router_logits: Array2<f32> = x.dot(&wg.t());
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            select_routes(&mut scores, self.k);

            let gate_weights = self.gate.weights(&full, &scores);
            for ((eid, _), gw) in scores.iter().zip(gate_weights) {
                route_token_ids.push(t as i64);
                route_expert_ids.push(*eid as i64);
                route_weights.push(gw);
            }
        }

        Ok(tvec![
            Tensor::from_shape(&[route_count], &route_token_ids)?.into_tvalue(),
            Tensor::from_shape(&[route_count], &route_expert_ids)?.into_tvalue(),
            Tensor::from_shape(&[route_count], &route_weights)?.into_tvalue(),
        ])
    }
}

impl TypedOp for RouteTopK {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2);
        ensure!(
            inputs[0].rank() == 2 || inputs[0].rank() == 3,
            "RouteTopK expects rank-2 or rank-3 x, got {}",
            inputs[0].rank()
        );
        ensure!(
            inputs[1].rank() == 2 || inputs[1].rank() == 3,
            "RouteTopK expects rank-2 or rank-3 router weights, got {}",
            inputs[1].rank()
        );
        let route_count = token_count_dim(&inputs[0].shape) * self.k;
        Ok(tvec![
            i64::datum_type().fact(std::slice::from_ref(&route_count)),
            i64::datum_type().fact(std::slice::from_ref(&route_count)),
            f32::datum_type().fact(&[route_count]),
        ])
    }

    as_op!();
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RoutedInputMode {
    TokenRows,
    RouteRows,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RoutedMatMul {
    pub input_mode: RoutedInputMode,
    pub cache_weights: bool,
}

#[derive(Clone, Debug)]
struct RoutedMatMulWeightsCache {
    k_dim: usize,
    n_dim: usize,
    num_experts: usize,
    packing: usize,
    a_format: PackedFormat,
    packed_by_expert: Vec<Box<dyn MMMInputValue>>,
}

#[derive(Clone, Debug)]
struct RoutedMatMulState {
    op: RoutedMatMul,
    cache: Option<RoutedMatMulWeightsCache>,
}

/// A plain f32 mmm kernel for this shape, packed on both sides -- no panel
/// extractor, since the caller (`RoutedMatMul`) packs its `a` rows and `b`
/// weights itself, directly in whichever format is chosen, rather than
/// extracting from data already packed some other way.
fn plain_f32_packed_mmm(
    k_dim: usize,
    n_dim: usize,
) -> Option<(Box<dyn MatMatMul>, usize, PackedFormat, PackedFormat)> {
    let mut query = Query::plain(f32::datum_type(), None, Some(k_dim), Some(n_dim));
    query.allow_extractor = false;
    let (mmm, packing, _extractor) = MmmDispatch::native().pick(&query)?;
    let (a, b) = &mmm.packings()[packing];
    let a_format = a.downcast_ref::<PackedFormat>()?.clone();
    let b_format = b.downcast_ref::<PackedFormat>()?.clone();
    Some((mmm, packing, a_format, b_format))
}

impl RoutedMatMul {
    fn input_view<'a>(&self, input_t: &'a Tensor) -> TractResult<ArrayView2<'a, f32>> {
        match self.input_mode {
            RoutedInputMode::TokenRows => as_2d_tokens(input_t.to_plain_array_view::<f32>()?),
            RoutedInputMode::RouteRows => {
                let input = input_t.to_plain_array_view::<f32>()?;
                ensure!(
                    input.ndim() == 2,
                    "route-row input must be rank 2, got {:?}",
                    input.shape()
                );
                Ok(input.into_dimensionality()?)
            }
        }
    }

    fn group_routes(
        &self,
        route_count: usize,
        num_experts: usize,
        route_expert_ids: &[i64],
    ) -> TractResult<Vec<Vec<usize>>> {
        let mut expert_routes = vec![Vec::new(); num_experts];
        for (r, &eid) in route_expert_ids.iter().enumerate().take(route_count) {
            ensure!(eid >= 0, "route {r} has negative expert id");
            let eid = eid as usize;
            ensure!(
                eid < num_experts,
                "route {r} references expert {eid}, but weights have {num_experts} experts"
            );
            expert_routes[eid].push(r);
        }
        Ok(expert_routes)
    }

    fn source_row(&self, r: usize, route_token_ids: &[i64]) -> TractResult<usize> {
        match self.input_mode {
            RoutedInputMode::TokenRows => {
                ensure!(route_token_ids[r] >= 0, "route {r} has negative token id");
                Ok(route_token_ids[r] as usize)
            }
            RoutedInputMode::RouteRows => Ok(r),
        }
    }

    fn eval_with_mmm(
        &self,
        input_t: &Tensor,
        weights_t: &Tensor,
        route_token_ids: &[i64],
        route_expert_ids: &[i64],
    ) -> TractResult<Option<Tensor>> {
        let input = self.input_view(input_t)?;
        let weights: ArrayView3<f32> =
            weights_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        let route_count = route_token_ids.len();
        let k_dim = weights.shape()[1];
        let n_dim = weights.shape()[2];
        let num_experts = weights.shape()[0];
        ensure!(route_count == route_expert_ids.len());
        ensure!(input.shape()[1] == k_dim);
        if self.input_mode == RoutedInputMode::RouteRows {
            ensure!(input.shape()[0] == route_count);
        }
        if route_count == 0 {
            return Ok(Some(Array2::<f32>::zeros((0, n_dim)).into_tensor()));
        }

        let Some((mmm, packing, a_format, b_format)) = plain_f32_packed_mmm(k_dim, n_dim) else {
            return Ok(None);
        };

        let expert_routes = self.group_routes(route_count, num_experts, route_expert_ids)?;
        let mut output = Array2::<f32>::zeros((route_count, n_dim));
        let item_size = f32::datum_type().size_of() as isize;
        let base = input.as_ptr();
        let row_stride_bytes = input.strides()[0] * item_size;
        let k_stride_bytes = input.strides()[1] * item_size;

        for (eid, routes) in expert_routes.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let mut row_byte_offsets = Vec::with_capacity(routes.len());
            for &r in routes {
                let src = self.source_row(r, route_token_ids)?;
                ensure!(
                    src < input.shape()[0],
                    "route {r} references input row {src}, but input has {} rows",
                    input.shape()[0]
                );
                row_byte_offsets.push(row_stride_bytes * src as isize);
            }

            let a = RoutedRowsInput::new(
                base,
                row_byte_offsets,
                k_dim,
                k_stride_bytes,
                a_format.clone(),
            );
            let w_e = weights_t.view_at_prefix(&[eid])?;
            let b = b_format.pack_tensor_view(&w_e, 0, 1)?;
            let mut expert_output = Tensor::zero::<f32>(&[routes.len(), n_dim])?;
            let store = unsafe { mmm.c_view(Some(0), Some(1)).wrap(&expert_output.view_mut()) };
            let uops = tvec![
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&a),
                    b: AsInputValue::Borrowed(&*b),
                    packing,
                },
                FusedSpec::Store(store),
            ];
            unsafe { mmm.run(routes.len(), n_dim, &uops)? };
            let expert_output: ArrayView2<f32> =
                expert_output.to_plain_array_view::<f32>()?.into_dimensionality()?;
            for (slot, &r) in routes.iter().enumerate() {
                output.row_mut(r).assign(&expert_output.row(slot));
            }
        }

        Ok(Some(output.into_tensor()))
    }

    fn build_weights_cache(
        &self,
        weights_input: &Tensor,
    ) -> TractResult<Option<RoutedMatMulWeightsCache>> {
        let weights_t = weights_input.cast_to::<f32>()?.into_owned();
        let weights: ArrayView3<f32> =
            weights_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        let num_experts = weights.shape()[0];
        let k_dim = weights.shape()[1];
        let n_dim = weights.shape()[2];
        let Some((_mmm, packing, a_format, b_format)) = plain_f32_packed_mmm(k_dim, n_dim) else {
            return Ok(None);
        };

        let mut packed_by_expert = Vec::with_capacity(num_experts);
        for eid in 0..num_experts {
            let w_e = weights_t.view_at_prefix(&[eid])?;
            packed_by_expert.push(b_format.pack_tensor_view(&w_e, 0, 1)?);
        }

        Ok(Some(RoutedMatMulWeightsCache {
            k_dim,
            n_dim,
            num_experts,
            packing,
            a_format,
            packed_by_expert,
        }))
    }

    fn eval_with_cached_mmm(
        &self,
        input_t: &Tensor,
        route_token_ids: &[i64],
        route_expert_ids: &[i64],
        cache: &RoutedMatMulWeightsCache,
    ) -> TractResult<Tensor> {
        let input = self.input_view(input_t)?;
        let route_count = route_token_ids.len();
        ensure!(route_count == route_expert_ids.len());
        ensure!(
            input.shape()[1] == cache.k_dim,
            "routed matmul input dim {} does not match weight dim {}",
            input.shape()[1],
            cache.k_dim
        );
        if self.input_mode == RoutedInputMode::RouteRows {
            ensure!(
                input.shape()[0] == route_count,
                "route-row input has {} rows but route metadata has {} rows",
                input.shape()[0],
                route_count
            );
        }
        if route_count == 0 {
            return Ok(Array2::<f32>::zeros((0, cache.n_dim)).into_tensor());
        }

        let Some((mmm, _packing, _a_format, _b_format)) =
            plain_f32_packed_mmm(cache.k_dim, cache.n_dim)
        else {
            bail!("cached routed matmul MMM is no longer available");
        };

        let expert_routes = self.group_routes(route_count, cache.num_experts, route_expert_ids)?;
        let mut output = Array2::<f32>::zeros((route_count, cache.n_dim));
        let item_size = f32::datum_type().size_of() as isize;
        let base = input.as_ptr();
        let row_stride_bytes = input.strides()[0] * item_size;
        let k_stride_bytes = input.strides()[1] * item_size;

        for (eid, routes) in expert_routes.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let mut row_byte_offsets = Vec::with_capacity(routes.len());
            for &r in routes {
                let src = self.source_row(r, route_token_ids)?;
                ensure!(
                    src < input.shape()[0],
                    "route {r} references input row {src}, but input has {} rows",
                    input.shape()[0]
                );
                row_byte_offsets.push(row_stride_bytes * src as isize);
            }

            let a = RoutedRowsInput::new(
                base,
                row_byte_offsets,
                cache.k_dim,
                k_stride_bytes,
                cache.a_format.clone(),
            );
            let mut expert_output = Tensor::zero::<f32>(&[routes.len(), cache.n_dim])?;
            let store = unsafe { mmm.c_view(Some(0), Some(1)).wrap(&expert_output.view_mut()) };
            let uops = tvec![
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&a),
                    b: AsInputValue::Borrowed(&*cache.packed_by_expert[eid]),
                    packing: cache.packing,
                },
                FusedSpec::Store(store),
            ];
            unsafe { mmm.run(routes.len(), cache.n_dim, &uops)? };
            let expert_output: ArrayView2<f32> =
                expert_output.to_plain_array_view::<f32>()?.into_dimensionality()?;
            for (slot, &r) in routes.iter().enumerate() {
                output.row_mut(r).assign(&expert_output.row(slot));
            }
        }

        Ok(output.into_tensor())
    }

    fn eval_grouped_ndarray(
        &self,
        input_t: &Tensor,
        weights_t: &Tensor,
        route_token_ids: &[i64],
        route_expert_ids: &[i64],
    ) -> TractResult<Tensor> {
        let input = self.input_view(input_t)?;
        let weights: ArrayView3<f32> =
            weights_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        ensure!(route_token_ids.len() == route_expert_ids.len());

        let route_count = route_token_ids.len();
        let k_dim = weights.shape()[1];
        let n_dim = weights.shape()[2];
        let num_experts = weights.shape()[0];
        ensure!(
            input.shape()[1] == k_dim,
            "routed matmul input dim {} does not match weight dim {}",
            input.shape()[1],
            k_dim
        );
        if self.input_mode == RoutedInputMode::RouteRows {
            ensure!(
                input.shape()[0] == route_count,
                "route-row input has {} rows but route metadata has {} rows",
                input.shape()[0],
                route_count
            );
        }

        let expert_routes = self.group_routes(route_count, num_experts, route_expert_ids)?;
        let mut output = Array2::<f32>::zeros((route_count, n_dim));
        for (eid, routes) in expert_routes.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let mut gathered = Array2::<f32>::zeros((routes.len(), k_dim));
            for (slot, &r) in routes.iter().enumerate() {
                let src = self.source_row(r, route_token_ids)?;
                ensure!(
                    src < input.shape()[0],
                    "route {r} references input row {src}, but input has {} rows",
                    input.shape()[0]
                );
                gathered.row_mut(slot).assign(&input.row(src));
            }

            let expert_output = gathered.dot(&weights.slice(s![eid, .., ..]));
            for (slot, &r) in routes.iter().enumerate() {
                output.row_mut(r).assign(&expert_output.row(slot));
            }
        }
        Ok(output.into_tensor())
    }
}

impl Op for RoutedMatMul {
    fn name(&self) -> StaticName {
        "RoutedMatMul".into()
    }
    op_as_typed_op!();
}

impl OpState for RoutedMatMulState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        if self.cache.is_none() {
            self.cache = self.op.build_weights_cache(&inputs[1])?;
        }
        let Some(cache) = self.cache.as_ref() else {
            return self.op.eval(ctx, inputs);
        };

        let input_t = inputs[0].cast_to::<f32>()?.into_owned();
        let route_token_ids_t = inputs[2].cast_to::<i64>()?.into_owned();
        let route_expert_ids_t = inputs[3].cast_to::<i64>()?.into_owned();
        let route_token_ids = plain_i64_slice(&route_token_ids_t, "route_token_ids")?;
        let route_expert_ids = plain_i64_slice(&route_expert_ids_t, "route_expert_ids")?;
        let output =
            self.op.eval_with_cached_mmm(&input_t, route_token_ids, route_expert_ids, cache)?;
        Ok(tvec![output.into_tvalue()])
    }

    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        // The weights cache is shared, session-wide scratch keyed by the op
        // itself, not per-lane state, so there is nothing to discard here.
        Ok(())
    }
}

impl EvalOp for RoutedMatMul {
    fn eval_out_of_plan(&self, inputs: TVec<TValue>) -> TractResult<Option<TVec<TValue>>> {
        if self.cache_weights {
            Ok(None)
        } else {
            Ok(Some(EvalOp::eval(self, &EvalContext::out_of_plan(), inputs)?))
        }
    }

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        if self.cache_weights {
            Ok(Some(Box::new(RoutedMatMulState { op: self.clone(), cache: None })))
        } else {
            Ok(None)
        }
    }

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // inputs: data, weights [E,K,N], route_token_ids [R], route_expert_ids [R].
        let input_t = inputs[0].cast_to::<f32>()?.into_owned();
        let weights_t = inputs[1].cast_to::<f32>()?.into_owned();
        let route_token_ids_t = inputs[2].cast_to::<i64>()?.into_owned();
        let route_expert_ids_t = inputs[3].cast_to::<i64>()?.into_owned();
        let route_token_ids = plain_i64_slice(&route_token_ids_t, "route_token_ids")?;
        let route_expert_ids = plain_i64_slice(&route_expert_ids_t, "route_expert_ids")?;

        let output = if let Some(output) =
            self.eval_with_mmm(&input_t, &weights_t, route_token_ids, route_expert_ids)?
        {
            output
        } else {
            self.eval_grouped_ndarray(&input_t, &weights_t, route_token_ids, route_expert_ids)?
        };
        Ok(tvec![output.into_tvalue()])
    }
}

impl TypedOp for RoutedMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 4);
        ensure!(inputs[1].rank() == 3, "RoutedMatMul weights must be rank 3");
        ensure!(inputs[2].rank() == 1, "RoutedMatMul route_token_ids must be rank 1");
        ensure!(inputs[3].rank() == 1, "RoutedMatMul route_expert_ids must be rank 1");
        let route_count = inputs[2].shape.to_tvec()[0].clone();
        let out_dim = inputs[1].shape.to_tvec()[2].clone();
        Ok(tvec![f32::datum_type().fact(&[route_count, out_dim])])
    }

    as_op!();
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RoutedQ40MatMul {
    pub input_mode: RoutedInputMode,
}

impl RoutedQ40MatMul {
    fn source_row(&self, route: usize, route_token_ids: &[i64]) -> TractResult<usize> {
        match self.input_mode {
            RoutedInputMode::TokenRows => {
                ensure!(route_token_ids[route] >= 0, "route {route} has negative token id");
                Ok(route_token_ids[route] as usize)
            }
            RoutedInputMode::RouteRows => Ok(route),
        }
    }

    fn group_routes(
        route_count: usize,
        num_experts: usize,
        route_expert_ids: &[i64],
    ) -> TractResult<Vec<Vec<usize>>> {
        let mut expert_routes = vec![Vec::new(); num_experts];
        for (route, &eid) in route_expert_ids.iter().enumerate().take(route_count) {
            ensure!(eid >= 0, "route {route} has negative expert id");
            let eid = eid as usize;
            ensure!(
                eid < num_experts,
                "route {route} references expert {eid}, but weights have {num_experts} experts"
            );
            expert_routes[eid].push(route);
        }
        Ok(expert_routes)
    }
}

impl Op for RoutedQ40MatMul {
    fn name(&self) -> StaticName {
        "RoutedQ40MatMul".into()
    }
    op_as_typed_op!();
}

impl EvalOp for RoutedQ40MatMul {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // inputs: data [rows,K], Q4_0 weights [E,N,K], route_token_ids [R],
        // route_expert_ids [R]. Weights are in the linear layout consumed by
        // the Q40 kernels, unlike RoutedMatMul's canonical [E,K,N] contract.
        ensure!(inputs.len() == 4);
        let input_t = inputs[0].cast_to::<f32>()?.into_owned();
        let weights_t = inputs[1].clone().into_tensor();
        let route_token_ids_t = inputs[2].cast_to::<i64>()?.into_owned();
        let route_expert_ids_t = inputs[3].cast_to::<i64>()?.into_owned();
        let route_token_ids = plain_i64_slice(&route_token_ids_t, "route_token_ids")?;
        let route_expert_ids = plain_i64_slice(&route_expert_ids_t, "route_expert_ids")?;
        ensure!(route_token_ids.len() == route_expert_ids.len());
        ensure!(input_t.rank() == 2, "RoutedQ40MatMul input must be [rows,K]");
        ensure!(weights_t.rank() == 3, "RoutedQ40MatMul weights must be [E,N,K]");
        ensure!(
            weights_t.storage_as::<BlockQuantStorage>().is_some(),
            "RoutedQ40MatMul weights must use block-quant storage"
        );

        let route_count = route_token_ids.len();
        let num_experts = weights_t.shape()[0];
        let n_dim = weights_t.shape()[1];
        let k_dim = weights_t.shape()[2];
        ensure!(input_t.shape()[1] == k_dim);
        if self.input_mode == RoutedInputMode::RouteRows {
            ensure!(
                input_t.shape()[0] == route_count,
                "route-row input has {} rows but route metadata has {} rows",
                input_t.shape()[0],
                route_count
            );
        }
        if route_count == 0 {
            return Ok(tvec![Tensor::zero_dt(f32::datum_type(), &[0, n_dim])?.into_tvalue()]);
        }

        let group_weights = (0..num_experts)
            .map(|eid| block_quant_group_tensor(&weights_t, eid))
            .collect::<TractResult<Vec<_>>>()?;
        let plan = build_block_quant_routed_matmul(group_weights)?;
        let expert_routes = Self::group_routes(route_count, num_experts, route_expert_ids)?;
        let mut state = PreparedRoutedMatMulState::default();
        let mut output = Tensor::zero_dt(f32::datum_type(), &[route_count, n_dim])?;

        let input_plain = input_t.try_as_plain_ram()?;
        let input = input_plain.as_slice::<f32>()?;
        let base = input.as_ptr();
        let item_size = f32::datum_type().size_of() as isize;
        let row_stride_bytes = input_t.strides()[0] * item_size;
        let k_stride_bytes = input_t.strides()[1] * item_size;

        for (eid, routes) in expert_routes.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let mut row_byte_offsets = Vec::with_capacity(routes.len());
            for &route in routes {
                let src = self.source_row(route, route_token_ids)?;
                ensure!(
                    src < input_t.shape()[0],
                    "route {route} references input row {src}, but input has {} rows",
                    input_t.shape()[0]
                );
                row_byte_offsets.push(row_stride_bytes * src as isize);
            }

            let mut expert_output = Tensor::zero::<f32>(&[routes.len(), n_dim])?;
            run_prepared_routed_matmul(
                &plan,
                eid,
                RoutedInputRows::explicit(base, row_byte_offsets, k_stride_bytes),
                &mut expert_output,
                &mut state,
            )?;

            let expert_plain = expert_output.try_as_plain_ram()?;
            let expert = expert_plain.as_slice::<f32>()?;
            let mut output_plain = output.try_as_plain_ram_mut()?;
            let output = output_plain.as_slice_mut::<f32>()?;
            for (slot, &route) in routes.iter().enumerate() {
                output[route * n_dim..(route + 1) * n_dim]
                    .copy_from_slice(&expert[slot * n_dim..(slot + 1) * n_dim]);
            }
        }

        Ok(tvec![output.into_tvalue()])
    }
}

impl TypedOp for RoutedQ40MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 4);
        ensure!(inputs[0].rank() == 2, "RoutedQ40MatMul input must be rank 2");
        ensure!(inputs[1].rank() == 3, "RoutedQ40MatMul weights must be rank 3 [E,N,K]");
        ensure!(inputs[2].rank() == 1, "RoutedQ40MatMul route_token_ids must be rank 1");
        ensure!(inputs[3].rank() == 1, "RoutedQ40MatMul route_expert_ids must be rank 1");
        let route_count = inputs[2].shape.to_tvec()[0].clone();
        let out_dim = inputs[1].shape.to_tvec()[1].clone();
        Ok(tvec![f32::datum_type().fact(&[route_count, out_dim])])
    }

    as_op!();
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RoutedCombine;

impl Op for RoutedCombine {
    fn name(&self) -> StaticName {
        "RoutedCombine".into()
    }
    op_as_typed_op!();
}

impl EvalOp for RoutedCombine {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // inputs: shape_like, route_values [R,D], route_token_ids [R], route_weights [R].
        let output_dt = inputs[0].datum_type();
        let route_values_t = inputs[1].cast_to::<f32>()?.into_owned();
        let route_token_ids_t = inputs[2].cast_to::<i64>()?.into_owned();
        let route_weights_t = inputs[3].cast_to::<f32>()?.into_owned();

        let shape = inputs[0].shape().to_vec();
        ensure!(
            shape.len() == 2 || shape.len() == 3,
            "RoutedCombine shape_like must be rank 2 or 3, got {:?}",
            shape
        );
        let t_tokens: usize = shape[..shape.len() - 1].iter().product();
        let d_model = *shape.last().context("RoutedCombine shape_like has no feature axis")?;
        let route_values: ArrayView2<f32> =
            route_values_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        let route_token_ids = plain_i64_slice(&route_token_ids_t, "route_token_ids")?;
        let route_weights = plain_f32_slice(&route_weights_t, "route_weights")?;
        ensure!(route_values.shape()[0] == route_token_ids.len());
        ensure!(route_token_ids.len() == route_weights.len());
        ensure!(
            route_values.shape()[1] == d_model,
            "route value dim {} does not match output dim {}",
            route_values.shape()[1],
            d_model
        );

        let mut output = Array2::<f32>::zeros((t_tokens, d_model));
        for r in 0..route_token_ids.len() {
            ensure!(route_token_ids[r] >= 0, "route {r} has negative token id");
            let token = route_token_ids[r] as usize;
            ensure!(
                token < t_tokens,
                "route {r} references token {token}, but output has {t_tokens} tokens"
            );
            let route = route_values.row(r);
            let mut out = output.row_mut(token);
            out.scaled_add(route_weights[r], &route);
        }

        let output_tensor = if shape.len() == 3 {
            output.into_shape_with_order((shape[0], shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(output_dt)?.into_owned();
        Ok(tvec![output_tensor.into_tvalue()])
    }
}

impl TypedOp for RoutedCombine {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 4);
        Ok(tvec![inputs[0].datum_type.fact(inputs[0].shape.clone())])
    }

    as_op!();
}
