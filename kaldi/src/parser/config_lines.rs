use tract_core::internal::*;

use nom::IResult;
use nom::{
    branch::alt, bytes::complete::*, character::complete::*, combinator::*, multi::many1,
    sequence::*,
};

use crate::model::{ComponentNode, ConfigLines, DimRangeNode};
use crate::parser::spaced;

pub fn parse_config(s: &str) -> TractResult<ConfigLines> {
    let mut input_node: Option<(String, usize)> = None;
    let mut component_nodes = HashMap::new();
    let mut dim_range_nodes = HashMap::new();
    let mut output_node: Option<(String, String)> = None;
    for line in s.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let line = line.split('#').next().unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let line_kind = line.split(" ").next().unwrap();
        match line_kind {
            "input-node" => {
                input_node = Some(
                    parse_input_node_line(line)
                        .map_err(|e| format!("Error {:?} while parsing {}", e, line))?
                        .1,
                )
            }
            "dim-range-node" => {
                let (name, it) = parse_dim_range_node_line(line)
                    .map_err(|e| format!("Error {:?} while parsing {}", e, line))?
                    .1;
                dim_range_nodes.insert(name, it);
            }
            "component-node" => {
                let (name, it) = parse_component_node_line(line)
                    .map_err(|e| format!("Error {:?} while parsing {}", e, line))?
                    .1;
                component_nodes.insert(name, it);
            }
            "output-node" => {
                output_node = Some(
                    parse_output_node_line(line)
                        .map_err(|e| format!("Error {:?} while parsing {}", e, line))?
                        .1,
                )
            }
            _ => bail!("Unknown config line {}", line_kind),
        }
    }
    let (input_name, input_dim) = input_node.unwrap();
    let (output_name, output_input) = output_node.unwrap();
    Ok(ConfigLines { input_dim, input_name, component_nodes, output_name, output_input, dim_range_nodes })
}

fn parse_input_node_line(i: &str) -> IResult<&str, (String, usize)> {
    let (i, _) = tag("input-node")(i)?;
    nom::branch::permutation((
        spaced(map(preceded(tag("name="), identifier), |n: &str| n.to_string())),
        spaced(preceded(tag("dim="), uinteger)),
    ))(i)
}

fn parse_component_node_line(i: &str) -> IResult<&str, (String, ComponentNode)> {
    let (i, _) = tag("component-node")(i)?;
    let (i, (name, component, input)) = nom::branch::permutation((
        spaced(map(preceded(tag("name="), identifier), |n: &str| n.to_string())),
        spaced(map(preceded(tag("component="), identifier), |n: &str| n.to_string())),
        spaced(preceded(tag("input="), super::descriptor::parse_general)),
    ))(i)?;
    Ok((i, (name, ComponentNode { component, input })))
}

fn parse_dim_range_node_line(i: &str) -> IResult<&str, (String, DimRangeNode)> {
    let (i, _) = tag("dim-range-node")(i)?;
    let (i, (name, input, dim, offset)) = nom::branch::permutation((
        spaced(map(preceded(tag("name="), identifier), |n: &str| n.to_string())),
        spaced(preceded(tag("input-node="), super::descriptor::parse_general)),
        spaced(preceded(tag("dim="), uinteger)),
        spaced(preceded(tag("dim-offset="), uinteger)),
    ))(i)?;
    Ok((i, (name,DimRangeNode { input, dim, offset })))
}

fn parse_output_node_line(i: &str) -> IResult<&str, (String, String)> {
    let (i, _) = tag("output-node")(i)?;
    nom::branch::permutation((
        spaced(map(preceded(tag("name="), identifier), |n: &str| n.to_string())),
        spaced(map(preceded(tag("input="), identifier), |n: &str| n.to_string())),
    ))(i)
}

pub fn identifier(i: &str) -> IResult<&str, &str> {
    recognize(pair(
        alpha1,
        nom::multi::many0(nom::branch::alt((alphanumeric1, tag("."), tag("_"), tag("-")))),
    ))(i)
}

pub fn uinteger(i: &str) -> IResult<&str, usize> {
    map(digit1, |s:&str| s.parse().unwrap())(i)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn identifiet_with_dot() {
        assert_eq!(identifier("lstm.c").unwrap().1, "lstm.c")
    }
}
