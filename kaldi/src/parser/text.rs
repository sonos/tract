use tract_core::internal::*;

use nom::IResult;
use nom::{
    bytes::complete::*, character::complete::*, combinator::*, multi::separated_list,
    number::complete::float, sequence::*,
};

use super::{integer, multispaced, open_any, spaced};

pub fn attributes(i: &[u8]) -> IResult<&[u8], HashMap<String, Arc<Tensor>>> {
    let (i, attributes) = nom::multi::many0(map(pair(open_any, tensor), |(k, v)| {
        (k.to_string(), v.into_arc_tensor())
    }))(i)?;
    Ok((i, attributes.into_iter().collect()))
}

pub fn tensor(i: &[u8]) -> IResult<&[u8], Tensor> {
    nom::branch::alt((scalar, vector, matrix))(i)
}

pub fn scalar(i: &[u8]) -> IResult<&[u8], Tensor> {
    nom::branch::alt((
        map(float, Tensor::from),
        map(integer(false), Tensor::from),
        map(tag("F"), |_| Tensor::from(false)),
        map(tag("T"), |_| Tensor::from(true)),
    ))(i)
}

pub fn vector(i: &[u8]) -> IResult<&[u8], Tensor> {
    map(delimited(spaced(tag("[")), separated_list(space1, float), spaced(tag("]"))), |t| {
        tensor1(&*t)
    })(i)
}

pub fn matrix(i: &[u8]) -> IResult<&[u8], Tensor> {
    let (i, v) = delimited(
        multispaced(tag("[")),
        separated_list(spaced(tag("\n")), separated_list(space1, float)),
        multispaced(tag("]")),
    )(i)?;
    let lines = v.len();
    let data: Vec<_> = v.into_iter().flat_map(|v| v.into_iter()).collect();
    let cols = data.len() / lines;
    let t = tract_core::ndarray::Array1::from_vec(data);
    let t = t.into_shape((lines, cols)).unwrap();
    Ok((i, t.into_tensor()))
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::super::nnet3;
    use super::*;

    #[test]
    fn test_nnet3_1() {
        let slice = r#"<Nnet3>

input-node name=input dim=3
component-node name=fixed1 input=input component=fixed1
output-node name=output input=fixed1

<NumComponents> 1
<ComponentName> foo <FixedAffineComponent> <LinearParams> [
  1.0 2.0 3.0
  4.0 5.0 6.0 ]
<BiasParams> [ 7.0 8.0 ]
</FixedAffineComponent>
</Nnet3>"#;
        nnet3(slice.as_bytes()).unwrap();
    }

    #[test]
    fn test_vector() {
        let slice = r#"[ 7.0 8.0 ]"#;
        assert_eq!(tensor(slice.as_bytes()).unwrap().1, tensor1(&[7.0f32, 8.0]));
    }

    #[test]
    fn test_matrix() {
        let slice = r#"[
            1.0 2.0 3.0
            4.0 5.0 6.0 ]"#;
        assert_eq!(
            tensor(slice.as_bytes()).unwrap().1,
            tensor2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])
        );
    }

    #[test]
    fn fixed_affine_40x10_T40_S3() {
        let slice = std::fs::read("test_cases/fixed_affine_40x10_T40_S3/model.raw.txt").unwrap();
        nnet3(&slice).unwrap();
    }
}
