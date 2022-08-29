use std::str::FromStr;

use nom::branch::permutation;
use nom::character::complete::digit1;
use nom::combinator::map_res;
use tract_core::internal::*;

use nom::{bytes::complete::*, multi::*};
use nom::{combinator::all_consuming, IResult};
use nom::{combinator::opt, number::complete::float};

use crate::ast::*;

use super::parse::{identifier, logical_literal, stag, translate_error};

#[inline(never)]
pub fn parse_quantization(doc: &str) -> TractResult<Vec<(String, QuantFormat)>> {
    all_consuming(many0(quantization))(doc).map(|pair| pair.1).map_err(translate_error)
}

// <quantization> ::= "<identifier>": <qparam>
fn quantization(i: &str) -> IResult<&str, (String, QuantFormat)> {
    let (i, _) = stag("")(i)?;
    let (i, _) = tag("\"")(i)?;
    let (i, id) = identifier(i)?;
    let (i, _) = tag("\"")(i)?;
    let (i, _) = stag(":")(i)?;
    let (i, qp) = qparam(i)?;
    let (i, _) = stag(";")(i)?;
    Ok((i, (id, qp)))
}

fn integer_numeric<T: FromStr>(i: &str) -> IResult<&str, T> {
    map_res(digit1, |s: &str| s.parse::<T>())(i)
}

// <qparam> ::= "<identifier>": <qparam>
fn qparam(i: &str) -> IResult<&str, QuantFormat> {
    let (i, id) =
        nom::branch::alt((stag("linear_quantize"), stag("zero_point_linear_quantize")))(i)?;
    let (i, _) = stag("(")(i)?;
    let (i, params, bits, signed) = match id {
        "linear_quantize" => {
            let (i, (bits, max, min)) =
                permutation((arg("bits", integer_numeric), arg("max", float), arg("min", float)))(
                    i,
                )?;

            (i, QParams::MinMax { min, max }, bits, true)
        }
        "zero_point_linear_quantize" => {
            let (i, (zero_point, scale, bits, signed, _)) = permutation((
                arg("zero_point", integer_numeric),
                arg("scale", float),
                arg("bits", integer_numeric),
                arg("signed", logical_literal),
                opt(arg("symmetric", logical_literal)),
            ))(i)?;
            (i, QParams::ZpScale { zero_point, scale }, bits, signed)
        }
        _ => unreachable!(),
    };

    let (i, _) = stag(")")(i)?;
    Ok((i, QuantFormat::Linear { params, bits, signed }))
}
// <arg>(<id>, <f>) ::= <id> "=" <f> ","
fn arg<'s, T, F>(name: &'static str, f: F) -> impl Fn(&'s str) -> IResult<&'s str, T>
where
    F: Fn(&'s str) -> IResult<&'s str, T>,
{
    move |i: &str| {
        let (i, _) = stag(name)(i)?;
        let (i, _) = stag("=")(i)?;
        let (i, num) = f(i)?;
        let (i, _) = opt(stag(","))(i)?;
        Ok((i, num))
    }
}

pub(crate) fn write_quant_format(
    w: &mut impl std::io::Write,
    name: String,
    format: QuantFormat,
) -> TractResult<()> {
    match format {
        QuantFormat::Linear {
            params: QParams::ZpScale {zero_point, scale}, bits, signed
        } => writeln!(w, "\"{}\": zero_point_linear_quantize(zero_point = {}, scale = {:.9}, bits = {}, signed = {}, symmetric = {});", name, zero_point, scale, bits, signed, zero_point == 0)?,
        QuantFormat::Linear {
            params: QParams::MinMax {min, max}, bits, signed: _
        } => writeln!(w, "\"{}\": linear_quantize(max = {:.9}, min = {:.9}, bits = {});", name, max, min, bits)?,
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use nom::combinator::all_consuming;

    fn p<'s, P, O, E>(parser: P, i: &'s str) -> O
    where
        O: std::fmt::Debug,
        P: FnMut(&'s str) -> IResult<&'s str, O, E>,
        E: nom::error::ParseError<&'s str> + std::fmt::Debug,
    {
        let res = all_consuming(parser)(i).unwrap();
        res.1
    }

    #[test]
    fn test_arg() {
        assert_eq!(p(arg("arg", float), "arg = 2.35,"), 2.35);

        assert_eq!(p(arg("test", tag("a")), "test = a"), "a");
    }

    #[test]
    fn test_qparam() {
        assert_eq!(
            p(qparam, "linear_quantize(min = 0.5, max = 123.8, bits = 8)"),
            QuantFormat::Linear {
                params: QParams::MinMax { min: 0.5, max: 123.8 },
                bits: 8,
                signed: true
            }
        );
    }

    #[test]
    fn test_quantization() {
        assert_eq!(
            p(quantization, r#""tensor_name": linear_quantize(min = 0.5, max = 123.8, bits = 8);"#),
            (
                "tensor_name".to_string(),
                QuantFormat::Linear {
                    params: QParams::MinMax { min: 0.5, max: 123.8 },
                    bits: 8,
                    signed: true
                }
            )
        );
    }

    #[test]
    fn test_quant_file() {
        assert_eq!(
            p(
                many0(quantization),
                r#"
                   "tensor_name1": linear_quantize(min = 0.5, max = 123.8, bits = 8);
                   "tensor_name2": linear_quantize(max = 0.52, min = 123.82, bits = 82);
                   "tensor_name3": zero_point_linear_quantize(zero_point = 52, scale = 123.83, bits = 83, signed = true, symmetric = false);"#
            ),
            vec![
                (
                    "tensor_name1".to_string(),
                    QuantFormat::Linear {
                        params: QParams::MinMax { min: 0.5, max: 123.8 },
                        bits: 8,
                        signed: true
                    }
                ),
                (
                    "tensor_name2".to_string(),
                    QuantFormat::Linear {
                        params: QParams::MinMax { max: 0.52, min: 123.82 },
                        bits: 82,
                        signed: true
                    }
                ),
                (
                    "tensor_name3".to_string(),
                    QuantFormat::Linear {
                        params: QParams::ZpScale { zero_point: 52, scale: 123.83 },
                        bits: 83,
                        signed: true
                    }
                )
            ]
        );
    }
}
