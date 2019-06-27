use tract_core::internal::*;

use nom::{
    bytes::complete::*, character::complete::*, combinator::*, multi::separated_list,
    number::complete::float, sequence::*, IResult,
};
use std::collections::HashMap;

use error_chain::bail;

pub struct NNet;

pub fn nnet3(i: &[u8]) -> IResult<&[u8], NNet> {
    let (i, _) = open(i, "Nnet3")?;
    let (i, _config_lines) = config_lines(i)?;
    let (i, num_components) = num_components(i)?;
    let mut components: HashMap<String, Box<InferenceOp>> = HashMap::new();
    let mut i = i;
    for _ in 0..num_components {
        let (new_i, (name, op)) = pair(component_name, component)(i)?;
        i = new_i;
        components.insert(name.to_owned(), op);
    }
    let (i, _) = close(i, "Nnet3")?;
    Ok((i, NNet))
}

fn config_lines(i: &[u8]) -> IResult<&[u8], NNet> {
    let (i, lines) = take_until("\n\n")(i)?;
    println!("{:?}", lines);
    Ok((i, NNet))
}

fn num_components(i: &[u8]) -> IResult<&[u8], usize> {
    let (i, _) = open(i, "NumComponents")?;
    let (i, n) =
        map_res(map_res(multispaced(take_while(nom::character::is_digit)), std::str::from_utf8), |s| {
            s.parse::<usize>()
        })(i)?;
    Ok((i, n))
}

fn component(i: &[u8]) -> IResult<&[u8], Box<InferenceOp>> {
    let (i, klass) = open_any(i)?;
    let (i, op) = match klass {
        "FixedAffineComponent" => crate::ops::affine::fixed_affine_component(i)?,
        e => panic!(),
    };
    let (i, _) = close(i, klass)?;
    Ok((i, op))
}

fn component_name(i: &[u8]) -> IResult<&[u8], &str> {
    multispaced(delimited(|i| open(i, "ComponentName"), name, multispace0))(i)
}

pub fn open<'a>(i: &'a [u8], t: &str) -> IResult<&'a [u8], ()> {
    map(multispaced(tuple((tag("<"), tag(t.as_bytes()), tag(">")))), |_| ())(i)
}

pub fn close<'a>(i: &'a [u8], t: &str) -> IResult<&'a [u8], ()> {
    map(multispaced(tuple((tag("</"), tag(t.as_bytes()), tag(">")))), |_| ())(i)
}

pub fn open_any(i: &[u8]) -> IResult<&[u8], &str> {
    multispaced(delimited(tag("<"), name, tag(">")))(i)
}

pub fn name(i: &[u8]) -> IResult<&[u8], &str> {
    map_res(take_while(nom::character::is_alphanumeric), std::str::from_utf8)(i)
}

pub fn matrix(i: &[u8]) -> IResult<&[u8], Tensor> {
    let (i, v) = delimited(
        multispaced(tag("[")),
        separated_list(spaced(tag("\n")), separated_list(space1, float)),
        multispaced(tag("]")),
    )(i)?;
    let lines = v.len();
    let data: Vec<f32> = v.into_iter().flat_map(|v| v.into_iter()).collect();
    let cols = data.len() / lines;
    let t = tract_core::ndarray::Array1::from_vec(data);
    let t = t.into_shape((lines, cols)).unwrap();
    Ok((i, t.into_tensor()))
}

pub fn vector(i: &[u8]) -> IResult<&[u8], Tensor> {
    map(
        delimited(multispaced(tag("[")), separated_list(space1, float), multispaced(tag("]"))),
        |t| tensor1(&*t),
    )(i)
}

pub fn spaced<I, O, E: nom::error::ParseError<I>, F>(it: F) -> impl Fn(I) -> IResult<I, O, E>
where
    I: nom::InputTakeAtPosition,
    <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    F: Fn(I) -> IResult<I, O, E>,
{
    delimited(space0, it, space0)
}

pub fn multispaced<I, O, E: nom::error::ParseError<I>, F>(it: F) -> impl Fn(I) -> IResult<I, O, E>
where
    I: nom::InputTakeAtPosition,
    <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    F: Fn(I) -> IResult<I, O, E>,
{
    delimited(multispace0, it, multispace0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_lines() {
        let slice = b"foo = bar\nbar=baz\n\n";
        config_lines(slice).unwrap();
    }

    #[test]
    fn test_nnet3_0() {
        let slice = b"<Nnet3>\n\nfoo=bar\n\n<NumComponents> 0\n</Nnet3>";
        nnet3(slice).unwrap();
    }

    #[test]
    fn test_nnet3_1() {
        let slice = r#"<Nnet3>

foo=bar

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
    fn fixed_affine_40x10_T40_S3() {
        let slice = std::fs::read("test_cases/fixed_affine_40x10_T40_S3/model.raw.txt").unwrap();
        nnet3(&slice).unwrap();
    }
}
