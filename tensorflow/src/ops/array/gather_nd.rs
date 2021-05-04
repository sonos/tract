use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

use tract_hir::ops::array::GatherNd;

pub fn gather_nd(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let batch_dims = pb.get_attr_opt_int("batch_dims")?.unwrap_or(0);
    Ok(Box::new(GatherNd::new(batch_dims)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // https://www.tensorflow.org/api_docs/python/tf/gather_nd
    #[test]
    fn simple_indexing() {
        let g = GatherNd::new(0);
        assert_eq!(
            g.eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[0, 0], [1, 1]]))).unwrap(),
            tvec!(rctensor1(&[1, 4]))
        );
    }

    #[test]
    fn slice_indexing() {
        let g = GatherNd::new(0);
        assert_eq!(
            g.eval(tvec!(rctensor2(&[[1, 2], [3, 4]]), rctensor2(&[[1], [0]]))).unwrap(),
            tvec!(rctensor2(&[[3, 4], [1, 2]]))
        );
    }

    #[test]
    fn tensor_3d_1() {
        let g = GatherNd::new(0);
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[1]]))).unwrap(),
            tvec!(rctensor3(&[[[11, 21], [31, 41]]]))
        );
    }

    #[test]
    fn tensor_3d_2() {
        let g = GatherNd::new(0);
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[0, 1], [1, 0]]))).unwrap(),
            tvec!(rctensor2(&[[30, 40], [11, 21]]))
        );
    }

    #[test]
    fn tensor_3d_3() {
        let g = GatherNd::new(0);
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[0, 0, 1], [1, 0, 1]]))).unwrap(),
            tvec!(rctensor1(&[20, 21]))
        );
    }

    #[test]
    fn tensor_bd1_1() {
        let g = GatherNd::new(1);
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor2(&[[1], [0]]))).unwrap(),
            tvec!(rctensor2(&[[30, 40], [11, 21]]))
        );
    }

    #[test]
    fn tensor_bd1_2() {
        let g = GatherNd::new(1);
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor3(&[[[1]], [[0]]]))).unwrap(),
            tvec!(rctensor3(&[[[30, 40]], [[11, 21]]]))
        );
    }

    #[test]
    fn tensor_bd1_3() {
        let g = GatherNd::new(1);
        let t = rctensor3(&[[[10, 20], [30, 40]], [[11, 21], [31, 41]]]);
        assert_eq!(
            g.eval(tvec!(t.clone(), rctensor3(&[[[1, 0]], [[0, 1]]]))).unwrap(),
            tvec!(rctensor2(&[[30], [21]]))
        );
    }
}
