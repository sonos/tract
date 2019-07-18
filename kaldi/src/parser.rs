use tract_core::internal::*;

use nom::IResult;
use nom::{
    branch::alt,
    bytes::complete::*,
    character::complete::*,
    combinator::*,
    number::complete::{le_i32, le_i64},
    sequence::*,
};

use std::collections::HashMap;

use crate::model::{Component, KaldiProtoModel};

use itertools::Itertools;

mod bin;
mod components;
mod config_lines;
mod descriptor;
mod text;

pub fn nnet3(slice: &[u8]) -> TractResult<KaldiProtoModel> {
    let (_, (config, components)) = parse_top_level(slice).map_err(|e| match e {
        nom::Err::Error(err) => format!(
            "Parsing kaldi enveloppe at: {:?}",
            err.0.iter().map(|b| format!("{:02x}", b)).join(" ")
        ),
        e => format!("{:?}", e),
    })?;
    let config_lines = config_lines::parse_config(config)?;
    Ok(KaldiProtoModel { config_lines, components, adjust_final_offset: 0 })
}

pub fn if_then_else<'a, T>(
    condition: bool,
    then: impl Fn(&'a [u8]) -> IResult<&'a [u8], T>,
    otherwise: impl Fn(&'a [u8]) -> IResult<&'a [u8], T>,
) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], T> {
    map(pair(cond(condition, then), cond(!condition, otherwise)), |(a, b)| a.or(b).unwrap())
}

fn parse_top_level(i: &[u8]) -> IResult<&[u8], (&str, HashMap<String, Component>)> {
    let (i, bin) = map(opt(tag([0, 0x42])), |o| Option::is_some(&o))(i)?;
    let (i, _) = open(i, "Nnet3")?;
    let (i, config_lines) = map_res(take_until("<NumComponents>"), std::str::from_utf8)(i)?;
    let (i, num_components) = num_components(bin, i)?;
    let mut components = HashMap::new();
    let mut i = i;
    for _ in 0..num_components {
        let (new_i, (name, op)) = pair(component_name, component(bin))(i)?;
        i = new_i;
        components.insert(name.to_owned(), op);
    }
    let (i, _) = close(i, "Nnet3")?;
    Ok((i, (config_lines, components)))
}

fn num_components(bin: bool, i: &[u8]) -> IResult<&[u8], usize> {
    let (i, _) = open(i, "NumComponents")?;
    let (i, n) = multispaced(integer(bin))(i)?;
    Ok((i, n as usize))
}

fn component(bin: bool) -> impl Fn(&[u8]) -> IResult<&[u8], Component> {
    move |i: &[u8]| {
        let (i, klass) = open_any(i)?;
        let (i, attributes) = if bin { bin::attributes(i, klass)? } else { text::attributes(i)? };
        let (i, _) = close(i, klass)?;
        Ok((i, Component { klass: klass.to_string(), attributes }))
    }
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
    map_res(
        recognize(pair(
            alpha1,
            nom::multi::many0(nom::branch::alt((alphanumeric1, tag("."), tag("_"), tag("-")))),
        )),
        std::str::from_utf8,
    )(i)
}

pub fn integer<'a>(bin: bool) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], i32> {
    if_then_else(
        bin,
        alt((preceded(tag([4]), le_i32), preceded(tag([8]), map(le_i64, |i| i as i32)))),
        map_res(
            map_res(
                recognize(pair(opt(tag("-")), take_while(nom::character::is_digit))),
                std::str::from_utf8,
            ),
            |s| s.parse::<i32>(),
        ),
    )
}

pub fn spaced<I, O, E: nom::error::ParseError<I>, F>(it: F) -> impl Fn(I) -> nom::IResult<I, O, E>
where
    I: nom::InputTakeAtPosition,
    <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    F: Fn(I) -> nom::IResult<I, O, E>,
{
    delimited(space0, it, space0)
}

pub fn multispaced<I, O, E: nom::error::ParseError<I>, F>(
    it: F,
) -> impl Fn(I) -> nom::IResult<I, O, E>
where
    I: nom::InputTakeAtPosition,
    <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    F: Fn(I) -> nom::IResult<I, O, E>,
{
    delimited(multispace0, it, multispace0)
}
