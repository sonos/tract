use tract_ndarray::Dimension;

use crate::transform::ModelTransform;
use crate::{broadcast, internal::*};
use std::borrow::Cow;
use std::fmt::Debug;

use super::prefix_matmul::{rewrite_einsum_to_prefix_matmul, PrefixMatMul};

#[derive(Debug, Default)]
pub struct AsBlas;

impl ModelTransform for AsBlas {
    fn name(&self) -> Cow<str> {
        "as-blas".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        rewrite_einsum_to_prefix_matmul(model)?;
        Rewriter::default()
            .with_rule_for("matmul-to-sgemm", matmul_to_sgemm)
            .rewrite(&(), model)?;
        Ok(())
    }
}

fn matmul_to_sgemm(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &PrefixMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    if !op.transpose_a
        && !op.transpose_b
        && !op.transpose_c
        && op.quantize_output.is_none()
        && model.node_input_facts(node.id)?.iter().all(|f| f.datum_type == f32::datum_type())
    {
        TypedModelPatch::replace_single_op(model, node, &node.inputs, SGemm::default()).map(Some)
    } else {
        Ok(None)
    }
}

#[derive(Debug, Default, Clone)]
pub struct SGemm {}

impl Op for SGemm {
    fn name(&self) -> Cow<str> {
        "SGemm".into()
    }

    op_as_typed_op!();
}

impl SGemm {
    fn output_shape<D: DimLike>(&self, a: &[D], b: &[D]) -> TractResult<TVec<D>> {
        ensure!(a.len() == b.len());
        let a_rank = a.len();
        let b_rank = b.len();
        let m = a[a_rank - 2].clone();
        let n = b[b_rank - 1].clone();
        let mut c_shape = broadcast::multi_broadcast(&[&a[..a_rank - 2], &b[..b_rank - 2]])
            .context("Unable to broadcast")?;
        c_shape.push(m);
        c_shape.push(n);
        Ok(c_shape)
    }
}

impl EvalOp for SGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        let a_ptr = a.as_ptr::<f32>()?;
        let b_ptr = b.as_ptr::<f32>()?;
        let c_shape = self.output_shape(a.shape(), b.shape())?;
        let rank = c_shape.len();
        let m = c_shape[rank - 2];
        let n = c_shape[rank - 1];
        let k = a.shape()[rank - 1];
        unsafe {
            let mut c = Tensor::uninitialized::<f32>(&c_shape)?;
            let c_ptr = c.as_ptr_mut::<f32>()?;
            let silent_a_axis = c.rank() - a.rank();
            let silent_b_axis = c.rank() - b.rank();
            for prefix in ndarray::indices(&c_shape[0..rank - 2]) {
                let mut a_ptr = a_ptr;
                let mut b_ptr = b_ptr;
                let mut c_ptr = c_ptr;
                for (axis, x) in prefix.as_array_view().iter().enumerate() {
                    if axis >= silent_a_axis && a.shape()[axis - silent_a_axis] != 1 {
                        a_ptr = a_ptr.offset(*x as isize * a.strides()[axis - silent_a_axis]);
                    }
                    if axis >= silent_b_axis && b.shape()[axis - silent_b_axis] != 1 {
                        b_ptr = b_ptr.offset(*x as isize * b.strides()[axis - silent_b_axis]);
                    }
                    c_ptr = c_ptr.offset(*x as isize * c.strides()[axis]);
                }
                if m == 1 {
                    cblas::sgemv(
                        cblas::Layout::RowMajor,
                        cblas::Transpose::Ordinary,
                        k as _,
                        n as _,
                        1.0,
                        std::slice::from_raw_parts(b_ptr, n * k),
                        n as _,
                        std::slice::from_raw_parts(a_ptr, k),
                        1,
                        0.0,
                        std::slice::from_raw_parts_mut(c_ptr, n),
                        1,
                    )
                } else if n == 1 {
                    cblas::sgemv(
                        cblas::Layout::RowMajor,
                        cblas::Transpose::None,
                        m as _,
                        k as _,
                        1.0,
                        std::slice::from_raw_parts(a_ptr, m * k),
                        k as _,
                        std::slice::from_raw_parts(b_ptr, k),
                        1,
                        0.0,
                        std::slice::from_raw_parts_mut(c_ptr, m),
                        1,
                    )
                } else {
                    cblas::sgemm(
                        cblas::Layout::RowMajor,
                        cblas::Transpose::None,
                        cblas::Transpose::None,
                        m as _,
                        n as _,
                        k as _,
                        1.0,
                        std::slice::from_raw_parts(a_ptr, m * k),
                        k as _,
                        std::slice::from_raw_parts(b_ptr, k * n),
                        n as _,
                        0.0,
                        std::slice::from_raw_parts_mut(c_ptr, m * n),
                        n as _,
                    )
                }
            }

            Ok(tvec!(c.into_tvalue()))
        }
    }
}

impl TypedOp for SGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs[0].datum_type == f32::datum_type());
        ensure!(inputs[1].datum_type == f32::datum_type());
        Ok(tvec!(f32::fact(&self.output_shape(&inputs[0].shape, &inputs[1].shape)?)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let fma = self.output_shape(&inputs[0].shape, &inputs[1].shape)?.iter().product::<TDim>()
            * inputs[0].shape.last().unwrap();
        Ok(tvec!((Cost::FMA(f32::datum_type()), fma)))
    }

    as_op!();
}
