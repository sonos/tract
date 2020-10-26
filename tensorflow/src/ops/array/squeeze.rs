use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;
use tract_hir::ops::array::Squeeze;

pub fn squeeze(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let squeeze_dims = pb.get_attr_opt_list_int("squeeze_dims")?;
    if let Some(mut squeeze_dims) = squeeze_dims {
        if squeeze_dims.len() > 0 {
            squeeze_dims.sort();
            return Ok(expand(Squeeze::new(Some(squeeze_dims))));
        }
    }
    Ok(expand(Squeeze::default()))
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use tract_ndarray::Array;

    fn run<I>(op: Squeeze, input: I) -> Tensor
    where
        I: Into<Tensor>,
    {
        expand(op).eval(tvec![input.into().into()]).unwrap().pop().unwrap().into_tensor()
    }

    #[test]
    fn squeeze_1() {
        assert_eq!(
            run(Squeeze::new(None), Array::from_elem([1, 2, 1, 3, 1, 1], 0)).shape(),
            &[2, 3]
        );
    }

    #[test]
    fn squeeze_2() {
        assert_eq!(
            run(Squeeze::new(Some(vec![2, 4])), Array::from_elem([1, 2, 1, 3, 1, 1], 0)).shape(),
            &[1, 2, 3, 1]
        );
    }
}
