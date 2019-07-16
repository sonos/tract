use tract_core::internal::*;

use nom::IResult;
use nom::{
    bytes::complete::*, character::complete::*, combinator::*, multi::separated_list,
    number::complete::float, sequence::*,
};

use super::{integer, multispaced, spaced, open_any};

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
