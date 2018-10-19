use tfdeploy::ops::array::Squeeze;
use tfdeploy::ops::prelude::*;

pub fn squeeze(pb: &::tfpb::node_def::NodeDef) -> TfdResult<Box<Op>> {
    let squeeze_dims = pb.get_attr_opt_list_int("squeeze_dims")?;
    if let Some(mut squeeze_dims) = squeeze_dims {
        if squeeze_dims.len() > 0 {
            squeeze_dims.sort();
            return Ok(Box::new(Squeeze::new(Some(squeeze_dims))));
        }
    }
    Ok(Box::new(Squeeze::default()))
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::*;
    use tfdeploy::dim::TDim;
    use tfdeploy::ops::InferenceOp;
    use tfdeploy::Tensor;

    fn run<I>(op: Squeeze, input: I) -> Tensor
    where
        I: Into<Tensor>,
    {
        op.eval(tvec![input.into().into()])
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
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
            run(
                Squeeze::new(Some(vec![2, 4])),
                Array::from_elem([1, 2, 1, 3, 1, 1], 0)
            ).shape(),
            &[1, 2, 3, 1]
        );
    }

    #[test]
    fn squeeze_inference_1() {
        let input = TensorFact::default()
            .with_datum_type(DatumType::TDim)
            .with_shape(shapefact![1, 1, (TDim::stream() - 2), 16]);
        let any = TensorFact::default();

        let op = Squeeze::new(Some(vec![1]));
        let inferred = op.infer_facts(tvec!(&input), tvec!(&any)).unwrap();

        let expect: TVec<_> = tvec!(
            TensorFact::default()
                .with_datum_type(DatumType::TDim)
                .with_shape(shapefact![1, (TDim::stream() - 2), 16])
        );

        assert_eq!(inferred.1, expect);
    }
}
