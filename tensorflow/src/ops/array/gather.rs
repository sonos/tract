use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

#[derive(Debug, Clone, new, Hash)]
pub struct GatherNd {}

tract_linalg::impl_dyn_hash!(GatherNd);

pub fn gather_nd(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(expand(GatherNd::new()))
}

impl Expansion for GatherNd {
    fn name(&self) -> Cow<str> {
        "GatherNd".into()
    }

    op_tf!();

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
                    if let Ok(n) = n.to_integer() {
                        for i in 0..(input_rank - n) as usize {
                            s.equals(&outputs[0].shape[indices_rank - 1 + i], &inputs[1].shape[i])?;
                        }
                    }
                    Ok(())
                },
            )
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(indices) = &model.outlet_fact(inputs[1])?.konst.clone() {
            if indices.rank() == 2 && indices.shape()[0] == 1 {
                let mut wire = tvec!(inputs[0].clone());
                for (axis, &i) in indices.cast_to::<i32>()?.as_slice::<i32>()?.iter().enumerate() {
                    wire = model.wire_node(
                        format!("{}.slice-axis-{}", prefix, axis),
                        tract_hir::ops::array::Slice::new(axis, i as usize, (i + 1) as usize),
                        &wire,
                    )?;
                }
                for i in (0..indices.shape()[1]).rev() {
                    wire = model.wire_node(
                        format!("{}.remove_axis_{}", prefix, i),
                        tract_hir::tract_core::ops::change_axes::AxisOp::Rm(i),
                        &wire,
                    )?;
                }
                model.wire_node(
                    format!("{}.add_axis", prefix),
                    tract_hir::tract_core::ops::change_axes::AxisOp::Add(0),
                    &wire,
                )
            } else {
                bail!("indices must be of rank 2 and of shape 1xN")
            }
        } else {
            bail!("indices input is expected to be const")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // https://www.tensorflow.org/api_docs/python/tf/gather_nd
    #[test]
    fn simple_indexing() {
        let g = expand(GatherNd::new());
        assert_eq!(
            g.as_stateless()
                .unwrap()
                .eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[0, 0], [1, 1]])))
                .unwrap(),
            tvec!(rctensor1(&[1, 4]))
        );
    }

    #[test]
    fn slice_indexing() {
        let g = expand(GatherNd::new());
        assert_eq!(
            g.as_stateless()
                .unwrap()
                .eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[1], [0]])))
                .unwrap(),
            tvec!(rctensor2(&[[3, 4], [1, 2]]))
        );
    }

    #[test]
    fn tensor_3d_1() {
        let g = expand(GatherNd::new());
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.as_stateless().unwrap().eval(tvec!(t.clone(), rctensor2(&[[1]]))).unwrap(),
            tvec!(rctensor3(&[[[11, 21], [31, 41]]]))
        );
    }

    #[test]
    fn tensor_3d_2() {
        let g = expand(GatherNd::new());
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.as_stateless().unwrap().eval(tvec!(t.clone(), rctensor2(&[[0, 1], [1, 0]]))).unwrap(),
            tvec!(rctensor2(&[[30, 40], [11, 21]]))
        );
    }

    #[test]
    fn tensor_3d_3() {
        let g = expand(GatherNd::new());
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.as_stateless()
                .unwrap()
                .eval(tvec!(t.clone(), rctensor2(&[[0, 0, 1], [1, 0, 1]])))
                .unwrap(),
            tvec!(rctensor1(&[20, 21]))
        );
    }
}
