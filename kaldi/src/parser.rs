use tract_core::internal::*;

use nom::{bytes::complete::*, character::complete::*, combinator::*, sequence::*, IResult};
use std::collections::HashMap;

use error_chain::bail;

pub struct NNet;

// top level rule
pub fn nnet3(i: &[u8]) -> IResult<&[u8], NNet> {
    let (i, _) = open(i, "NNet3")?;
    let (i, _config_lines) = config_lines(i)?;
    let (i, num_components) = num_components(i)?;
    let mut components: HashMap<String, ()> = HashMap::new();
    for _ in 0..num_components {}
    let (i, _) = close(i, "NNet3")?;
    Ok((i, NNet))
}

fn config_lines(i: &[u8]) -> IResult<&[u8], NNet> {
    let (i, _) = take_until("\n\n")(i)?;
    Ok((i, NNet))
}

fn num_components(i: &[u8]) -> IResult<&[u8], usize> {
    let (i, _) = open(i, "NumComponents")?;
    let (i, n) =
        map_res(map_res(take_while(nom::character::is_digit), std::str::from_utf8), |s| {
            s.parse::<usize>()
        })(i)?;
    let (i, _) = multispace0(i)?;
    Ok((i, n))
}

fn component(i: &[u8]) -> IResult<&[u8], Box<InferenceOp>> {
    let (i, klass) = open_any(i)?;
    let (i, op) = match klass {
        b"FixedAffineComponent" => crate::ops::affine::fixed_affine_component(i)?,
        e => panic!()
    };
    let (i, _) = multispace0(i)?;
    let (i, _) = tag("</")(i)?;
    let (i, _) = tag(klass)(i)?;
    let (i, _) = tag(">")(i)?;
    let (i, _) = multispace0(i)?;
    Ok((i, op))
}

pub fn open<'a>(i: &'a[u8], t: &str) -> IResult<&'a[u8], ()>{
    let (i, _) = multispace0(i)?;
    let (i, _) = tag("<")(i)?;
    let (i, _) = tag(t.as_bytes())(i)?;
    let (i, _) = tag(">")(i)?;
    let (i, _) = multispace0(i)?;
    Ok((i, ()))
}

pub fn close<'a>(i: &'a[u8], t: &str) -> IResult<&'a[u8], ()>{
    let (i, _) = multispace0(i)?;
    let (i, _) = tag("</")(i)?;
    let (i, _) = tag(t.as_bytes())(i)?;
    let (i, _) = tag(">")(i)?;
    let (i, _) = multispace0(i)?;
    Ok((i, ()))
}

pub fn open_any(i: &[u8]) -> IResult<&[u8], &[u8]> {
    let (i, _) = multispace0(i)?;
    let (i, tag) = delimited(tag("<"), take_while(nom::character::is_alphabetic), tag(">"))(i)?;
    let (i, _) = multispace0(i)?;
    Ok((i, tag))
}

pub fn tensor(i: &[u8]) -> IResult<&[u8], Tensor> {
    let (i, _) = multispace0(i)?;
    let (i, _) = tag("[")(i)?;
    let (i, _) = multispace0(i)?;
    let (i, _) = tag("]")(i)?;
    let (i, _) = multispace0(i)?;
    let t = Tensor::from(0.0f32);
    Ok((i, t))
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
    fn test_nnet3_1() {
        let slice = b"<NNet3>\n\nfoo=bar\n\n<NumComponents> 0\n</NNet3>";
        nnet3(slice).unwrap();
    }

    #[test]
    #[ignore]
    fn fixed_affine_40x10_T40_S3() {
        let slice = std::fs::read("test_cases/fixed_affine_40x10_T40_S3/model.raw.txt").unwrap();
        nnet3(&slice).unwrap();
    }
}
