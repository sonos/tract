use crate::kernels::matmul::{GemmDispatchParams, GemmKernel, GgmlGemm};
use crate::utils::get_metal_buffer;
use anyhow::ensure;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};
use tract_gpu::device::get_context;
use tract_gpu::tensor::{DeviceArenaView, DeviceTensor, DeviceTensorExt, OwnedDeviceTensor};
use tract_gpu::utils::{as_quant_fact, facts_to_device_facts, get_quant_fact};

/// Horizontal fusion of sibling matmul nodes that share one activation: the
/// sibling weights are concatenated along their output-row axis into a single
/// tensor, and at decode (a single activation row) the whole family runs as
/// ONE mat-vec dispatch writing a [1, n_total] buffer whose column ranges are
/// returned as zero-copy views. This removes per-sibling kernel-launch
/// latency, which dominates tiny projections (q/k/v, GDN in-projections,
/// shared-expert gates). At prefill each output keeps its own dispatch (the
/// tiled GEMMs are bandwidth/ALU-bound so fusion buys nothing there) writing
/// straight into its own arena slot, byte-identical to the unfused graph.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalMultiGemm {
    /// Output rows per sibling, in weight-concatenation order.
    pub splits: TVec<usize>,
    /// True when the original nodes were weight-first (W @ x, output
    /// [.., n_i, m]) rather than activation-first (x @ W^t, [.., m, n_i]).
    pub weight_first: bool,
}

impl MetalMultiGemm {
    fn output_shape_for(&self, x_shape: &[TDim], n_i: usize) -> TVec<TDim> {
        let rank = x_shape.len();
        let mut shape: TVec<TDim> = x_shape.into();
        if self.weight_first {
            let m = shape[rank - 2].clone();
            shape[rank - 2] = n_i.to_dim();
            shape[rank - 1] = m;
        } else {
            shape[rank - 1] = n_i.to_dim();
        }
        shape
    }

    fn concrete_output_shape_for(&self, x_shape: &[usize], n_i: usize) -> TVec<usize> {
        let rank = x_shape.len();
        let mut shape: TVec<usize> = x_shape.into();
        if self.weight_first {
            let m = shape[rank - 2];
            shape[rank - 2] = n_i;
            shape[rank - 1] = m;
        } else {
            shape[rank - 1] = n_i;
        }
        shape
    }

    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2, "MetalMultiGemm expects [activation, weights]");
        let (x, w) = (inputs[0], inputs[1]);
        let n_total: usize = self.splits.iter().sum();
        let w_shape: TVec<usize> = if let Some(bqf) = as_quant_fact(w, &Q4_0) {
            bqf.shape().into()
        } else {
            w.shape
                .as_concrete()
                .context("MetalMultiGemm weights must have a concrete shape")?
                .into()
        };
        ensure!(
            w_shape.len() == 2 && w_shape[0] == n_total,
            "MetalMultiGemm weights must be [n_total, k], got {w_shape:?} for splits {:?}",
            self.splits
        );
        let k = w_shape[1];
        ensure!(x.rank() >= 2);
        ensure!(
            x.shape[x.rank() - 1] == k.to_dim(),
            "activation k {} != weight k {k}",
            x.shape[x.rank() - 1]
        );
        Ok(self
            .splits
            .iter()
            .map(|&n_i| x.datum_type.fact(self.output_shape_for(&x.shape, n_i)))
            .collect())
    }
}

impl Op for MetalMultiGemm {
    fn name(&self) -> StaticName {
        "MetalMultiGemm".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("splits: {:?} weight_first: {}", self.splits, self.weight_first)])
    }

    op_as_typed_op!();
}

/// True when the given strides describe a packed row-major layout of shape
/// (dims of 1 may carry arbitrary strides).
fn is_dense(shape: &[usize], strides: &[isize]) -> bool {
    let mut expect = 1isize;
    for (d, s) in shape.iter().zip(strides.iter()).rev() {
        if *d != 1 && *s != expect {
            return false;
        }
        expect *= *d as isize;
    }
    true
}

impl EvalOp for MetalMultiGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (x_val, w_val) = args_2!(inputs);
        let x = x_val.to_device_tensor()?;
        let w = w_val.to_device_tensor()?;
        let rank = x.rank();
        ensure!(rank >= 2);
        let k = x.shape()[rank - 1];
        let rows = if k == 0 { 0 } else { x.len() / k };
        let dt = x.datum_type();
        let w_dt = w.datum_type();
        let q40 = get_quant_fact(w, &Q4_0).is_some();
        let n_total: usize = self.splits.iter().sum();
        ensure!(
            is_dense(x.shape(), x.strides()),
            "MetalMultiGemm activation must be dense, got shape {:?} strides {:?}",
            x.shape(),
            x.strides()
        );
        let row_bytes =
            if q40 { (k / Q4_0.block_len()) * Q4_0.block_bytes() } else { k * w_dt.size_of() };

        if rows == 1 {
            // Decode: one fused mat-vec dispatch into a contiguous
            // [1, n_total] buffer; outputs are zero-copy views of it. The
            // buffer is pool-recycled (same size every step).
            let backing: Arc<Box<dyn OwnedDeviceTensor>> =
                Arc::new(get_context()?.uninitialized_device_tensor(&[n_total], dt)?);
            let c = DeviceTensor::ArenaView(DeviceArenaView::from_owned(
                backing.clone(),
                dt,
                tvec![1, n_total],
                tvec![n_total as isize, 1],
                0,
            )?);
            crate::with_metal_stream(|stream| {
                stream.retain_tensor(x);
                stream.retain_tensor(w);
                stream.retain_tensor(&c);
                let params = GemmDispatchParams {
                    dts: [dt, w_dt, dt],
                    a_batch: 1,
                    b_batch: 1,
                    m: 1,
                    k,
                    n: n_total,
                    transpose_a: false,
                    a_offset: x.buffer_offset(),
                    transpose_b: true,
                    b_offset: w.buffer_offset(),
                    q40_b: q40,
                    c_offset: 0,
                    a_strides: natural_strides(&[1, 1, k]),
                    b_strides: natural_strides(&[1, n_total, k]),
                };
                GgmlGemm.dispatch_eval(
                    stream,
                    params,
                    get_metal_buffer(x),
                    get_metal_buffer(w),
                    get_metal_buffer(&c),
                )
            })?;
            let mut outs = tvec![];
            let mut col = 0usize;
            for &n_i in &self.splits {
                let shape = self.concrete_output_shape_for(x.shape(), n_i);
                let view = DeviceArenaView::from_owned(
                    backing.clone(),
                    dt,
                    shape.clone(),
                    Tensor::natural_strides(&shape),
                    col * dt.size_of(),
                )?;
                outs.push(DeviceTensor::ArenaView(view).into_tensor().into_tvalue());
                col += n_i;
            }
            return Ok(outs);
        }

        // Prefill: one dispatch per sibling, each writing its own output,
        // same dispatches as the unfused graph.
        if self.weight_first {
            // The [n_i, rows] output layout only matches the declared fact
            // when there is no batch dim folded into rows.
            ensure!(
                rows == x.shape()[rank - 2],
                "MetalMultiGemm weight-first needs leading dims of 1, got {:?}",
                x.shape()
            );
            ensure!(!q40, "MetalMultiGemm weight-first weights cannot be Q4_0");
            ensure!(w_dt == dt);
        }
        let mut outs = tvec![];
        crate::with_metal_stream(|stream| {
            stream.retain_tensor(x);
            stream.retain_tensor(w);
            let mut row0 = 0usize;
            for (slot, &n_i) in self.splits.iter().enumerate() {
                let shape = self.concrete_output_shape_for(x.shape(), n_i);
                let c = tract_gpu::session_handler::make_tensor_for_node_output(
                    session, node_id, slot, dt, &shape,
                )?;
                stream.retain_tensor(&c);
                let w_offset = w.buffer_offset::<usize>() + row0 * row_bytes;
                let (params, a_buf, b_buf) = if self.weight_first {
                    (
                        GemmDispatchParams {
                            dts: [w_dt, dt, dt],
                            a_batch: 1,
                            b_batch: 1,
                            m: n_i,
                            k,
                            n: rows,
                            transpose_a: false,
                            a_offset: w_offset,
                            transpose_b: true,
                            b_offset: x.buffer_offset(),
                            q40_b: false,
                            c_offset: c.buffer_offset(),
                            a_strides: natural_strides(&[1, n_i, k]),
                            b_strides: natural_strides(&[1, rows, k]),
                        },
                        w,
                        x,
                    )
                } else {
                    (
                        GemmDispatchParams {
                            dts: [dt, w_dt, dt],
                            a_batch: 1,
                            b_batch: 1,
                            m: rows,
                            k,
                            n: n_i,
                            transpose_a: false,
                            a_offset: x.buffer_offset(),
                            transpose_b: true,
                            b_offset: w_offset,
                            q40_b: q40,
                            c_offset: c.buffer_offset(),
                            a_strides: natural_strides(&[1, rows, k]),
                            b_strides: natural_strides(&[1, n_i, k]),
                        },
                        x,
                        w,
                    )
                };
                GgmlGemm.dispatch_eval(
                    stream,
                    params,
                    get_metal_buffer(a_buf),
                    get_metal_buffer(b_buf),
                    get_metal_buffer(&c),
                )?;
                outs.push(c.into_tensor().into_tvalue());
                row0 += n_i;
            }
            Ok(())
        })?;
        Ok(outs)
    }
}

impl TypedOp for MetalMultiGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |input_facts| self.output_facts_inner(input_facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |input_facts| {
            let x = input_facts[0];
            let rows =
                x.shape.iter().rev().skip(1).fold(TDim::Val(1), |acc, d| acc * d.clone());
            let k = x.shape[x.rank() - 1].clone();
            let n_total: usize = self.splits.iter().sum();
            let fma = rows * k * n_total.to_dim();
            if x.datum_type == f16::datum_type() {
                Ok(tvec!((Cost::FMA(f16::datum_type()), fma)))
            } else {
                Ok(tvec!((Cost::FMA(f32::datum_type()), fma)))
            }
        })
    }

    as_op!();
}
