use errors::*;
use ops::Op;
use Model;
use Plan;

mod types;
pub use self::types::*;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod helpers;
pub mod graphviz;

/// Attempts to unify two tensor facts into a more specialized one.
pub fn unify(x: &TensorFact, y: &TensorFact) -> Result<TensorFact> {
    let tensor = TensorFact {
        datatype: unify_datatype(&x.datatype, &y.datatype)?,
        shape: unify_shape(&x.shape, &y.shape)?,
        value: unify_value(&x.value, &y.value)?,
    };

    Ok(tensor)
}

/// Attempts to unify two datatype facts.
pub fn unify_datatype(x: &TypeFact, y: &TypeFact) -> Result<TypeFact> {
    use self::TypeFact::*;

    let datatype = match (x, y) {
        (_, Any) => x,
        (Any, _) => y,
        (Only(a), Only(b)) if a == b => x,
        _ => bail!("Impossible to unify datatypes {:?} and {:?}.", x, y)
    };

    Ok(*datatype)
}

/// Attempts to unify two shape facts.
pub fn unify_shape(x: &ShapeFact, y: &ShapeFact) -> Result<ShapeFact> {
    use self::DimFact::*;
    use self::ShapeFact::*;
    use itertools::EitherOrBoth::{Both, Left, Right};
    use itertools::Itertools;

    let xi = x.inner().iter();
    let yi = y.inner().iter();

    let dimensions: Vec<_> = xi.zip_longest(yi)
        .map(|r| match r {
            Both(Any, Any) => Ok(Any),

            Both(Only(i), Any) | Both(Any, Only(i)) => Ok(Only(*i)),

            Both(Only(i), Only(j)) if i == j => Ok(Only(*i)),
            Both(Only(i), Only(j)) => bail!("Impossible to unify dimensions {:?} and {:?}.", i, j),

            Left(d) if y.is_open() => Ok(*d),
            Right(d) if x.is_open() => Ok(*d),

            Left(_) | Right(_) => bail!(
                "Impossible to unify closed shapes of different rank (found {:?} and {:?}).",
                x,
                y
            ),
        })
        .collect::<Result<_>>()?;

    if x.is_open() && y.is_open() {
        Ok(Open(dimensions))
    } else {
        Ok(Closed(dimensions))
    }
}

/// Attempts to unify two value facts.
pub fn unify_value(x: &ValueFact, y: &ValueFact) -> Result<ValueFact> {
    use self::ValueFact::*;

    let value = match (x, y) {
        (_, Any) => x.clone(),
        (Any, _) => y.clone(),
        (Only(a), Only(b)) => if a == b {
            x.clone()
        } else {
            bail!("Impossible to unify values {:?} and {:?}.", x, y);
        },
    };

    Ok(value)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub id: usize,
    pub from_node: usize,
    pub from_out: usize,
    pub to_node: usize,
    pub tensor: TensorFact,
}

/// Runs the analyser on the given graph.
///
/// The output argument is used to infer an execution plan for the graph.
/// Changing it won't alter the correctness of the analysis, but it might
/// take much longer to complete.
pub fn analyse<'a>(model: &'a Model, output: usize, debug: bool) -> Result<(Vec<(usize, String, String)>, Vec<Edge>)> {
    // We first give an identity to each edge of the graph.
    let mut nodes = vec![];
    let mut edges = vec![];
    let mut prev_edges = vec![Vec::new(); model.nodes().len() + 1];
    let mut next_edges = vec![Vec::new(); model.nodes().len() + 1];

    for node in model.nodes() {
        for input in &node.inputs {
            let id = edges.len();

            edges.push(Edge {
                id,
                from_node: input.0,
                from_out: input.1.unwrap_or(0),
                to_node: node.id,
                tensor: TensorFact::new(),
            });

            prev_edges[node.id].push(id);
            next_edges[input.0].push(id);
        }

        nodes.push((
            node.id,
            node.name.clone(),
            node.op_name.clone(),
        ));
    }

    // Add a special output node.
    let special_node_id = nodes.len();
    let special_edge_id = edges.len();
    nodes.push((special_node_id, "output".to_string(), "output".to_string()));
    edges.push(Edge {
        id: special_edge_id,
        from_node: output,
        from_out: 0,
        to_node: nodes.len() - 1,
        tensor: TensorFact::new(),
    });

    next_edges[output].push(special_edge_id);
    prev_edges[special_node_id].push(special_edge_id);

    // Compute and run an execution plan for the graph.
    let plan = Plan::for_node(model, output)?;
    let mut changed;
    let mut forward = true;

    macro_rules! one_pass {
        ($source:ident, $target:ident, $fn:ident) => {{
            // TODO(liautaud): Remove this.
            if debug {
                println!("");
                println!("Starting a round of {}.", stringify!($fn));
            }

            for &n in &plan.order {
                let inferred = {
                    let sources: Vec<_> = $source[n].iter().map(|&i| &edges[i].tensor).collect();

                    // Don't do anything on the output node.
                    if n == special_node_id {
                        continue;
                    }

                    let node = model.get_node_by_id(n)?;
                    let inferred = node.op.$fn(sources);

                    if inferred.is_err() {
                        // TODO(liautaud): Remove this.
                        if debug {
                            println!("[{}] ({}): {}", n, node.op_name, inferred.unwrap_err());
                        }
                        continue;
                    }

                    inferred.unwrap()
                };

                for (i, &j) in $target[n].iter().enumerate() {
                    let unified = unify(&inferred[i], &edges[j].tensor)?;
                    if unified != edges[j].tensor {
                        edges[j].tensor = unified;
                        changed = true;
                    }

                    // TODO(liautaud): Remove this.
                    if debug {
                        let node_name = format!("[{}] ({})", n, model.get_node_by_id(n)?.op_name);
                        let mut inferred_display = format!("{:?}", inferred);
                        inferred_display.truncate(150);
                        println!("{} Inferred: {}", node_name, inferred_display);
                    }
                }

                // TODO(liautaud): Remove this.
                if debug && model.get_node_by_id(n)?.op_name != "Const" {
                    graphviz::display_graph("debug".to_string(), &nodes, &edges, &vec![n], forward)?;
                }
            }
        }};
    };

    // TODO(liautaud): Remove this.
    if debug {
        graphviz::display_graph("debug".to_string(), &nodes, &edges, &vec![], true)?;
    }

    loop {
        changed = false;

        if forward {
            one_pass!(prev_edges, next_edges, infer_forward);
        } else {
            one_pass!(next_edges, prev_edges, infer_backward);
        }

        forward = !forward;

        if !changed {
            break;
        }
    }

    Ok((nodes, edges))
}

#[cfg(tests)]
mod tests {
    #[test]
    fn unify_same_datatype() {
        let dt = TypeFact::Only(DataType::DT_FLOAT);
        assert_eq!(unify_datatype(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datatypes_only() {
        let dt1 = TypeFact::Only(DataType::DT_FLOAT);
        let dt2 = TypeFact::Only(DataType::DT_DOUBLE);
        assert!(unify_datatype(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_datatypes_any_left() {
        let dt = TypeFact::Only(DataType::DT_FLOAT);
        assert_eq!(unify_datatype(&TypeFact::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datatypes_any_right() {
        let dt = TypeFact::Only(DataType::DT_FLOAT);
        assert_eq!(unify_datatype(&dt, &TypeFact::Any).unwrap(), dt);
    }

    #[test]
    fn unify_same_shape_1() {
        let s = ShapeFact::Closed(vec![]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_2() {
        use super::DimFact::*;
        let s = ShapeFact::Closed(vec![Any]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_3() {
        use super::DimFact::*;
        let s = ShapeFact::Closed(vec![Only(1), Only(2)]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_different_shapes_1() {
        use super::DimFact::*;
        let s1 = ShapeFact::Closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::Closed(vec![Only(1)]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_2() {
        use super::DimFact::*;
        let s1 = ShapeFact::Closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::Closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_3() {
        use super::DimFact::*;
        let s1 = ShapeFact::Open(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::Closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_4() {
        use super::DimFact::*;
        let s1 = ShapeFact::Closed(vec![Any]);
        let s2 = ShapeFact::Closed(vec![Any]);
        let sr = ShapeFact::Closed(vec![Any]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_5() {
        use super::DimFact::*;
        let s1 = ShapeFact::Closed(vec![Any]);
        let s2 = ShapeFact::Closed(vec![Only(1)]);
        let sr = ShapeFact::Closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_6() {
        use super::DimFact::*;
        let s1 = ShapeFact::Open(vec![]);
        let s2 = ShapeFact::Closed(vec![Only(1)]);
        let sr = ShapeFact::Closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_7() {
        use super::DimFact::*;
        let s1 = ShapeFact::Open(vec![Any, Only(2)]);
        let s2 = ShapeFact::Closed(vec![Only(1), Any, Any]);
        let sr = ShapeFact::Closed(vec![Only(1), Only(2), Any]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_same_value() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_values_only() {
        use ndarray::prelude::*;
        let dt1 = ValueFact::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
        let dt2 = ValueFact::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[2]))));
        assert!(unify_value(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_values_any_left() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&ValueFact::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_values_any_right() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&dt, &ValueFact::Any).unwrap(), dt);
    }
}
