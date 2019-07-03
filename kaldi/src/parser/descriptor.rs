use nom::IResult;
use nom::{
    bytes::complete::*, character::complete::*, combinator::*, multi::separated_list,
    number::complete::float, sequence::*,
};

use crate::model::GeneralDescriptor;
use crate::parser::spaced;

pub fn parse_general(i: &str) -> IResult<&str, GeneralDescriptor> {
    spaced(nom::branch::alt((
        map(
            delimited(
                tag("Append("),
                separated_list(spaced(tag(",")), parse_general),
                spaced(tag(")")),
            ),
            GeneralDescriptor::Append,
        ),
        map(
            delimited(
                tag("Offset("),
                separated_pair(parse_general, spaced(tag(",")), integer),
                spaced(tag(")")),
            ),
            |(inner, offset)| GeneralDescriptor::Offset(Box::new(inner), offset as isize),
        ),
        map(super::config_lines::identifier, |i| GeneralDescriptor::Name(i.to_string())),
    )))(i)
}

pub fn integer(i: &str) -> IResult<&str, i32> {
    map_res(recognize(pair(opt(tag("-")), digit1)), |s: &str| s.parse::<i32>())(i)
}

#[cfg(test)]
mod test {
    use super::*;
    use GeneralDescriptor::*;

    fn name(s: &str) -> GeneralDescriptor {
        GeneralDescriptor::Name(s.to_string())
    }

    #[test]
    fn test_offset() {
        assert_eq!(parse_general("Offset(input, -1)").unwrap().1, Offset(name("input").into(), -1))
    }

    #[test]
    fn test_conv() {
        assert_eq!(
            parse_general("Append(Offset(input, -1), input, Offset(input, 1))").unwrap().1,
            Append(vec!(
                Offset(name("input").into(), -1),
                name("input"),
                Offset(name("input").into(), 1)
            ))
        )
    }

    #[test]
    fn test_lstm() {
        assert_eq!(
            parse_general("Append(input, IfDefined(Offset(lstm1, -1)))").unwrap().1,
            Append(vec!(name("input"), IfDefined(Offset(name("lstm1.c").into(), -1).into())))
        )
    }
}
