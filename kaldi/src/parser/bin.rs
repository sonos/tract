use tract_hir::internal::*;

use nom::combinator::*;
use nom::IResult;

use super::components::COMPONENTS;

pub fn attributes<'a>(i: &'a [u8], klass: &str) -> IResult<&'a [u8], HashMap<String, Arc<Tensor>>> {
    map(nom::multi::many0(|j| attribute(j, klass)), |v| v.into_iter().collect())(i)
}

fn attribute<'a>(i: &'a [u8], klass: &str) -> IResult<&'a [u8], (String, Arc<Tensor>)> {
    let (i, name) = super::open_any(i)?;
    let (i, value) = COMPONENTS[klass][name].parse_bin(i)?;
    Ok((i, (name.to_string(), value.into_arc_tensor())))
}
