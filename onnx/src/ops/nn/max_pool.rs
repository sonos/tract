use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops::cnn::{HirMaxPool, PoolSpec};
use tract_hir::ops::nn::DataFormat;

pub fn max_pool(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let kernel_shape = node.get_attr_tvec("kernel_shape")?;
    let pad = super::pad(node, true)?;
    let strides = super::strides(node)?;
    let dilations = super::dilations(node)?;
    let pool_spec = PoolSpec::new(DataFormat::NCHW, kernel_shape, pad, dilations, strides, 0, 0);
    if node.output.len() == 2 {
        let column_major = node.get_attr_opt::<i64>("storage_order")?.unwrap_or(0) == 1;
        let pool = HirMaxPool::new(pool_spec, Some(i64::datum_type()));
        Ok((expand(MaxPoolWithIndices { pool, column_major }), vec![]))
    } else {
        Ok((expand(HirMaxPool::new(pool_spec, None)), vec![]))
    }
}

/// MaxPool with its ONNX `Indices` output.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct MaxPoolWithIndices {
    pool: HirMaxPool,
    column_major: bool,
}

impl Expansion for MaxPoolWithIndices {
    fn name(&self) -> StaticName {
        "MaxPool".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        self.pool.rules(s, inputs, outputs)
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let spatial_dims = model.outlet_fact(inputs[0])?.shape[2..].into();
        let pooled = self.pool.wire(prefix, model, inputs)?;
        let indices = model.wire_node(
            format!("{prefix}.indices"),
            OnnxMaxPoolIndices { spatial_dims, column_major: self.column_major },
            &[pooled[1]],
        )?[0];
        Ok(tvec!(pooled[0], indices))
    }
}

/// Rewrites MaxPool indices from tract's convention to ONNX's.
///
/// tract's MaxPool yields each index within its own (batch, channel) plane, in row-major
/// order. ONNX flattens over the whole input tensor, so the index also counts the planes
/// before it, and with `storage_order = 1` the spatial part is column-major.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct OnnxMaxPoolIndices {
    /// Spatial dims of the MaxPool input.
    spatial_dims: TVec<TDim>,
    column_major: bool,
}

impl Op for OnnxMaxPoolIndices {
    fn name(&self) -> StaticName {
        "OnnxMaxPoolIndices".into()
    }

    op_as_typed_op!();
}

impl EvalOp for OnnxMaxPoolIndices {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let indices = args_1!(inputs);
        let spatial = self
            .spatial_dims
            .iter()
            .map(|d| Ok(d.eval(ctx.symbols).to_usize()?))
            .collect::<TractResult<TVec<usize>>>()?;
        let plane_len: usize = spatial.iter().product();
        let per_plane: usize = indices.shape()[2..].iter().product();
        let mut indices = indices.into_tensor();
        for (plane, chunk) in indices
            .try_as_plain_mut()?
            .as_slice_mut::<i64>()?
            .chunks_mut(per_plane.max(1))
            .enumerate()
        {
            for index in chunk {
                let mut within = *index as usize;
                if self.column_major {
                    within = row_major_to_column_major(within, &spatial);
                }
                *index = (plane * plane_len + within) as i64;
            }
        }
        Ok(tvec!(indices.into_tvalue()))
    }
}

fn row_major_to_column_major(mut index: usize, dims: &[usize]) -> usize {
    // Peel coordinates off from the fastest-moving (last) dim, then weight each by the
    // product of the dims before it.
    let mut coords = tvec!(0; dims.len());
    for (coord, dim) in coords.iter_mut().zip(dims).rev() {
        *coord = index % dim;
        index /= dim;
    }
    let mut stride = 1;
    let mut result = 0;
    for (coord, dim) in coords.iter().zip(dims) {
        result += coord * stride;
        stride *= dim;
    }
    result
}

impl TypedOp for OnnxMaxPoolIndices {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_major_2d() {
        // 2x3 plane: row-major (h, w) = h * 3 + w, column-major = h + w * 2
        let dims = [2, 3];
        for h in 0..2 {
            for w in 0..3 {
                assert_eq!(row_major_to_column_major(h * 3 + w, &dims), h + w * 2);
            }
        }
    }

    #[test]
    fn column_major_3d() {
        let dims = [2, 3, 4];
        for d in 0..2 {
            for h in 0..3 {
                for w in 0..4 {
                    let row = (d * 3 + h) * 4 + w;
                    assert_eq!(row_major_to_column_major(row, &dims), d + h * 2 + w * 6);
                }
            }
        }
    }
}
