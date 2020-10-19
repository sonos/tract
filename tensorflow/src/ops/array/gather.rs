use tract_hir::internal::*;
use tract_ndarray::prelude::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

#[derive(Debug, Clone, new, Hash)]
pub struct GatherNd {}

tract_data::impl_dyn_hash!(GatherNd);

pub fn gather_nd(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(Box::new(GatherNd::new()))
}

impl GatherNd {
    fn compute_shape<D: DimLike>(
        &self,
        data_shape: &[D],
        indices_shape: &[D],
    ) -> TractResult<TVec<D>> {
        let mut shape: TVec<D> = indices_shape.into();
        let n = shape.pop().unwrap().to_usize()?;
        shape.extend(data_shape[n..].iter().cloned());
        Ok(shape)
    }

    unsafe fn eval_t<T: Datum>(
        &self,
        output: &mut Tensor,
        data: &Tensor,
        indices: &ArrayViewD<i32>,
    ) {
        let data = data.to_array_view_unchecked::<T>();
        for prefix in tract_ndarray::indices(&indices.shape()[0..indices.ndim() - 1]) {
            let mut dst = output.to_array_view_mut_unchecked();
            let mut coords = indices.view();
            for &x in prefix.slice().iter() {
                dst.index_axis_inplace(Axis(0), x);
                coords.index_axis_inplace(Axis(0), x);
            }
            let mut src = data.view();
            for &x in coords.iter() {
                src.index_axis_inplace(Axis(0), x as _);
            }
            dst.assign(&src);
        }
    }
}

impl Op for GatherNd {
    fn name(&self) -> Cow<str> {
        "GatherNd".into()
    }

    op_tf!();
    op_as_typed_op!();
}

impl EvalOp for GatherNd {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, indices) = args_2!(inputs);
        let shape = self.compute_shape(&data.shape(), &indices.shape())?;
        let indices = indices.cast_to::<i32>()?;
        let indices = indices.to_array_view::<i32>()?;
        unsafe {
            let mut output = Tensor::uninitialized_dt(data.datum_type(), &*shape)?;
            dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self,
                &mut output,
                &data,
                &indices
            ));
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}

impl InferenceRulesOp for GatherNd {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[1].rank, move |s, indices_rank| {
            let indices_rank = indices_rank as usize;
            for i in 0..(indices_rank - 1) {
                s.equals(&outputs[0].shape[i], &inputs[1].shape[i])?;
            }
            s.given_2(
                &inputs[1].shape[indices_rank - 1],
                &inputs[1].rank,
                move |s, n, input_rank| {
                    if let Ok(n) = n.to_i64() {
                        for i in 0..(input_rank - n) as usize {
                            s.equals(&outputs[0].shape[indices_rank - 1 + i], &inputs[1].shape[i])?;
                        }
                    }
                    Ok(())
                },
            )
        })
    }

    as_op!();
    to_typed!();
}

impl TypedOp for GatherNd {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = self.compute_shape(&inputs[0].shape.to_tvec(), &inputs[1].shape.to_tvec())?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*shape)?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(indices) = &model.outlet_fact(node.inputs[1])?.konst {
            if indices.rank() == 2 && indices.shape()[0] == 1 {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, node.inputs[0])?;
                for (axis, &i) in indices.cast_to::<i32>()?.as_slice::<i32>()?.iter().enumerate() {
                    wire = patch.wire_node(
                        format!("{}-slice-axis-{}", node.name, axis),
                        tract_hir::ops::array::Slice::new(axis, i as usize, (i + 1) as usize),
                        &[wire],
                    )?[0];
                }
                for i in (0..indices.shape()[1]).rev() {
                    wire = patch.wire_node(
                        format!("{}-remove_axis_{}", node.name, i),
                        tract_hir::tract_core::ops::change_axes::AxisOp::Rm(i),
                        &[wire],
                    )?[0];
                }
                wire = patch.wire_node(
                    format!("{}-add_axis", node.name),
                    tract_hir::tract_core::ops::change_axes::AxisOp::Add(0),
                    &[wire],
                )?[0];
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // https://www.tensorflow.org/api_docs/python/tf/gather_nd
    #[test]
    fn simple_indexing() {
        let g = GatherNd::new();
        assert_eq!(
            g.eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[0, 0], [1, 1]]))).unwrap(),
            tvec!(rctensor1(&[1, 4]))
        );
    }

    #[test]
    fn slice_indexing() {
        let g = GatherNd::new();
        assert_eq!(
            g.eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[1], [0]]))).unwrap(),
            tvec!(rctensor2(&[[3, 4], [1, 2]]))
        );
    }

    #[test]
    fn tensor_3d_1() {
        let g = GatherNd::new();
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[1]]))).unwrap(),
            tvec!(rctensor3(&[[[11, 21], [31, 41]]]))
        );
    }

    #[test]
    fn tensor_3d_2() {
        let g = GatherNd::new();
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[0, 1], [1, 0]]))).unwrap(),
            tvec!(rctensor2(&[[30, 40], [11, 21]]))
        );
    }

    #[test]
    fn tensor_3d_3() {
        let g = GatherNd::new();
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[0, 0, 1], [1, 0, 1]]))).unwrap(),
            tvec!(rctensor1(&[20, 21]))
        );
    }
}
