use tract_core::internal::*;

use nom::{bytes::complete::*, character::complete::*, combinator::*, sequence::*, IResult};

use crate::model::{ComponentNode, ConfigLines};

pub fn parse_config(s: &str) -> TractResult<ConfigLines> {
    let mut input_node: Option<(String, usize)> = None;
    let mut component_nodes = HashMap::new();
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
    Ok(ConfigLines { input_dim, input_name, component_nodes, output_name, output_input })
}

fn parse_input_node_line(i: &str) -> IResult<&str, (String, usize)> {
    let mut name: Option<String> = None;
    let mut dim: Option<usize> = None;
    let (i, _) = tag("input-node")(i)?;
    for (k, v) in
        iterator(i, preceded(space0, separated_pair(identifier, tag("="), identifier))).into_iter()
    {
        match k {
            "name" => name = Some(v.to_owned()),
            "dim" => dim = Some(v.parse().unwrap()),
            e => panic!("un-handled key in input_node line"),
        }
    }
    match (name, dim) {
        (Some(name), Some(dim)) => Ok((i, (name.to_string(), dim))),
        (None, _) => panic!("expect name"),
        (_, None) => panic!("expect dim"),
    }
}

fn parse_component_node_line(i: &str) -> IResult<&str, (String, ComponentNode)> {
    let mut name: Option<String> = None;
    let mut component: Option<String> = None;
    let mut input: Option<String> = None;
    let (i, _) = tag("component-node")(i)?;
    for (k, v) in
        iterator(i, preceded(space0, separated_pair(identifier, tag("="), identifier))).into_iter()
    {
        match k {
            "name" => name = Some(v.to_owned()),
            "input" => input = Some(v.to_owned()),
            "component" => component = Some(v.to_owned()),
            e => panic!("un-handled key {} in component-node line", e),
        }
    }
    match (name, component, input) {
        (Some(name), Some(component), Some(input)) => {
            Ok((i, (name.to_string(), ComponentNode { component, input })))
        }
        (None, _, _) => panic!("expect name"),
        (_, None, _) => panic!("expect component"),
        (_, _, None) => panic!("expect input"),
    }
}

fn parse_output_node_line(i: &str) -> IResult<&str, (String, String)> {
    let mut name: Option<String> = None;
    let mut input: Option<String> = None;
    let (i, _) = tag("output-node")(i)?;
    for (k, v) in
        iterator(i, preceded(space0, separated_pair(identifier, tag("="), identifier))).into_iter()
    {
        match k {
            "name" => name = Some(v.to_owned()),
            "input" => input = Some(v.to_owned()),
            e => panic!("un-handled key {} output-node line", k),
        }
    }
    match (name, input) {
        (Some(name), Some(input)) => Ok((i, (name.to_string(), input))),
        (None, _) => panic!("expect name"),
        (_, None) => panic!("expect input"),
    }
}

fn identifier(i: &str) -> IResult<&str, &str> {
    alphanumeric1(i)
}
