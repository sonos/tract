use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::math::Sub;
use crate::ops::nn::{Reduce, Reducer, RmsNorm};
use tract_itertools::Itertools;

/// Layer normalization along `axis`: `y = (x - mean) / sqrt(var + eps)`, mean and variance
/// taken over `axis` independently for each index of the other axes, optionally followed by
/// the learned affine part `y * scale + bias`.
///
/// Inputs: `[input]`, or `[input, scale, bias]` with `scale` and `bias` rank-1 F32 tensors as
/// long as the input dimension along `axis`. Statistics and the affine part are computed in
/// F32; the output keeps the input datum type.
///
/// Detected (without the affine part) from the `Reduce<Mean> -> Sub -> RmsNorm` chain that
/// ONNX exporters emit for LayerNormalization (see `detect_layer_norm`), after GroupNorm
/// detection had its chance. GPU backends fold the per-axis gamma Mul and beta Add that follow
/// into the affine inputs.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LayerNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for LayerNorm {
    fn name(&self) -> StaticName {
        "LayerNorm".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

impl EvalOp for LayerNorm {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 1 || inputs.len() == 3);
        let in_dt = inputs[0].datum_type();
        let affine = if inputs.len() == 3 {
            Some((
                inputs[1].cast_to::<f32>()?.into_owned(),
                inputs[2].cast_to::<f32>()?.into_owned(),
            ))
        } else {
            None
        };
        let input = inputs.into_iter().next().unwrap();
        let mut x = if in_dt == DatumType::F32 {
            input.into_tensor()
        } else {
            input.cast_to::<f32>()?.into_owned()
        };
        let shape = x.shape().to_vec();
        let dim = shape[self.axis];
        let inner: usize = shape[self.axis + 1..].iter().product();
        let eps = self.eps.cast_to_scalar::<f32>()?;
        let affine = affine
            .as_ref()
            .map(|(s, b)| -> TractResult<(&[f32], &[f32])> {
                let s = s.try_as_plain_ram()?.as_slice::<f32>()?;
                let b = b.try_as_plain_ram()?.as_slice::<f32>()?;
                ensure!(s.len() == dim && b.len() == dim);
                Ok((s, b))
            })
            .transpose()?;
        let block = dim * inner;
        if block > 0 {
            let mut view = x.try_as_plain_ram_mut()?;
            let data = view.as_slice_mut::<f32>()?;
            if inner == 1 {
                normalize_rows(data, dim, eps, affine)?;
            } else {
                normalize_strided(data, dim, inner, eps, affine)?;
            }
        }
        if in_dt == DatumType::F32 {
            return Ok(tvec!(x.into_tvalue()));
        }
        Ok(tvec!(x.cast_to_dt(in_dt)?.into_owned().into_tvalue()))
    }
}

/// Rows contiguous along the normalized axis: the mean through the linalg sum kernel, then the
/// centered row through the fused RmsNorm kernel, which leaves `(x - mean) / sqrt(var + eps)`.
fn normalize_rows(
    data: &mut [f32],
    dim: usize,
    eps: f32,
    affine: Option<(&[f32], &[f32])>,
) -> TractResult<()> {
    let sum = tract_linalg::routines::Func::ReduceSum.reduce_f32()?;
    let rms_norm = tract_linalg::routines::rms_norm_f32()?;
    let total = data.len();
    tract_linalg::multithread::par_chunks_mut(data, dim, total, |_, chunk| {
        for row in chunk.chunks_mut(dim) {
            let mean = sum.run_with_params(row, ())? / dim as f32;
            row.iter_mut().for_each(|v| *v -= mean);
            rms_norm(row, eps);
            if let Some((s, b)) = affine {
                row.iter_mut().zip(s).zip(b).for_each(|((v, s), b)| *v = *v * s + b);
            }
        }
        Ok(())
    })
}

/// The normalized axis has `inner` contiguous elements behind each of its `dim` steps: the
/// statistics accumulate one contiguous `inner` row at a time, so every loop runs over contiguous
/// lanes.
fn normalize_strided(
    data: &mut [f32],
    dim: usize,
    inner: usize,
    eps: f32,
    affine: Option<(&[f32], &[f32])>,
) -> TractResult<()> {
    let block = dim * inner;
    let total = data.len();
    tract_linalg::multithread::par_chunks_mut(data, block, total, |_, chunk| {
        let mut mean = vec![0f32; inner];
        let mut inv = vec![0f32; inner];
        for block in chunk.chunks_mut(block) {
            mean.fill(0.0);
            inv.fill(0.0);
            for row in block.chunks(inner) {
                mean.iter_mut().zip(row).for_each(|(m, x)| *m += x);
            }
            mean.iter_mut().for_each(|m| *m /= dim as f32);
            for row in block.chunks(inner) {
                inv.iter_mut().zip(row).zip(&mean).for_each(|((v, x), m)| *v += (x - m) * (x - m));
            }
            inv.iter_mut().for_each(|v| *v = (*v / dim as f32 + eps).sqrt().recip());
            for (d, row) in block.chunks_mut(inner).enumerate() {
                row.iter_mut().zip(&mean).zip(&inv).for_each(|((x, m), i)| *x = (*x - m) * i);
                if let Some((s, b)) = affine {
                    row.iter_mut().for_each(|x| *x = *x * s[d] + b[d]);
                }
            }
        }
        Ok(())
    })
}

impl TypedOp for LayerNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.eps.rank() == 0, "LayerNorm: eps must be a rank-0 tensor");
        ensure!(inputs.len() == 1 || inputs.len() == 3, "LayerNorm: 1 or 3 inputs");
        ensure!(self.axis < inputs[0].rank());
        for affine in &inputs[1..] {
            ensure!(affine.rank() == 1, "LayerNorm: scale and bias must be rank 1");
            ensure!(affine.shape[0] == inputs[0].shape[self.axis]);
        }
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.clone())))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut letters = 'a'..;
        let axes = (0..inputs[0].rank())
            .map(|ix| {
                let mut axis =
                    Axis::new(letters.next().unwrap(), inputs.len(), 1).input(0, ix).output(0, ix);
                if ix == self.axis && inputs.len() == 3 {
                    axis = axis.input(1, 0).input(2, 0);
                }
                axis
            })
            .collect_vec();
        AxesMapping::new(inputs.len(), 1, axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        rule_if!(node.inputs.len() == 1);
        rule_if_some!(axis = change.transform_axis(self.axis));
        let op = LayerNorm { axis, eps: self.eps.clone() };
        Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        rule_if!(output_axis != self.axis);
        patch.wire_node(&node.name, self.clone(), inputs).map(Some)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let count: TDim = inputs[0].shape.iter().product();
        Ok(tvec!((Cost::FMA(inputs[0].datum_type), count * 5)))
    }

    as_op!();
}

/// Search pattern => LayerNorm(X), from `RmsNorm(X - Mean(X))` over the same axis.
pub fn detect_layer_norm(
    op: &RmsNorm,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if_some!(sub = model.single_prev_node_as::<TypedBinOp>(node).map(|(_, n)| n));
    rule_if!(sub.op_as::<TypedBinOp>().unwrap().0.is::<Sub>());
    let (x, mean) = (sub.inputs[0], sub.inputs[1]);
    let mean_node = model.node(mean.node);
    rule_if_some!(reduce = mean_node.op_as::<Reduce>());
    rule_if!(reduce.reducer == Reducer::Mean && *reduce.axes == [op.axis]);
    rule_if!(mean_node.inputs[0] == x);
    let mut patch = TypedModelPatch::default();
    let x = patch.tap_model(model, x)?;
    let y = patch.wire_node(
        format!("{}.layer_norm", node.name),
        LayerNorm { axis: op.axis, eps: op.eps.clone() },
        &[x],
    )?;
    patch.shunt_outside(model, node.id.into(), y[0])?;
    Ok(Some(patch))
}
