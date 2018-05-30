use super::*;

/// Infers every property when all the values are concrete.
pub fn infer_forward_concrete(op: &Op, inputs: &Vec<&TensorFact>) -> Result<Vec<TensorFact>> {
    let input_values: Result<Vec<_>> = inputs
        .iter()
        .map(|t| t.value.concretize())
        .collect();

    match input_values {
        Ok(v) => {
            // If we know the value of all the inputs, we can deduce everything.
            let input_inputs: Vec<_> = v
                .into_iter()
                .map(|v| v.clone().into())
                .collect();

            let output_value = op.eval(input_inputs)?.pop().unwrap();
            let output = TensorFact {
                datatype: inputs[0].datatype,
                shape: output_value.shape().into(),
                value: valuefact!(output_value.into_matrix())
            };

            Ok(vec![output])
        },

        _ => bail!("Can't infer value: some inputs are not concrete.")
    }
}

/// Infers basic properties in the case of unary or binary operators.
pub fn infer_forward_basic(op: &Op, inputs: Vec<&TensorFact>) -> Result<Vec<TensorFact>> {
    if let Ok(output) = infer_forward_concrete(op, &inputs) {
        return Ok(output);
    }

    // Otherwise we can only deduce the type and shape of the output.
    let input_shapes: Vec<_> = inputs
        .iter()
        .map(|t| &t.shape)
        .collect();

    let output = TensorFact {
        datatype: inputs[0].datatype,
        shape: infer_shape_broadcasting(input_shapes)?,
        value: valuefact!(_)
    };

    Ok(vec![output])
}

/// Infers basic shape properties in the case of broadcasting operators.
pub fn infer_shape_broadcasting(shapes: Vec<&ShapeFact>) -> Result<ShapeFact> {
    if shapes.iter().any(|s| s.is_open()) {
        bail!("Can't infer shape for broadcasting operators when some inputs have an open shape.");
    }

    let shapes: Vec<_> = shapes.iter()
        .map(|s| s.inner())
        .collect();
    let bound = shapes.iter()
        .map(|s| s.len())
        .max()
        .unwrap();

    let mut output_shape = vec![];

    for i in 1..bound {
        let mut previous = None;
        let mut unknown = 0;

        for shape in &shapes {
            if shape.len() < i {
                continue;
            }

            match &shape[shape.len() - i] {
                DimFact::Any => unknown += 1,
                DimFact::Only(j) => match previous {
                    Some(k) if k != j => bail!("Invalid shape (broadcasting): {} is not compatible with {}.", j, k),
                    _ => previous = Some(j)
                }
            };
        }

        if unknown > 1 {
            bail!("Can't infer shape (broadcasting): there are multiple unknown values at same index.");
        } else if unknown == 1 && previous != None {
            bail!("Can't infer shape (broadcasting): there are both unknown and known values at same index.");
        } else if unknown == 1 && previous == None {
            output_shape.push(DimFact::Any);
        } else {
            output_shape.push(DimFact::Only(*previous.unwrap()));
        }
    }

    Ok(ShapeFact::Closed(output_shape))
}

/// Returns the most specific closed shape out of an iterator.
pub fn most_specific_shape<'a, I: IntoIterator<Item=&'a ShapeFact>>(iter: I) -> Result<&'a ShapeFact> {
    let mut prev_rank = None;
    let mut prev_concrete = None;
    let mut best = None;

    for shape in iter {
        match shape {
            ShapeFact::Open(_) => continue,
            ShapeFact::Closed(s) => {
                let rank = s.len();

                if prev_rank.is_some() && rank != prev_rank.unwrap() {
                    bail!("Rank mismatch between different shapes.");
                } else {
                    prev_rank = Some(rank);
                }

                let concrete = s.iter().filter(|d| d.is_concrete()).count();

                if prev_concrete.is_none() || concrete > prev_concrete.unwrap() {
                    prev_concrete = Some(concrete);
                    best = Some(shape)
                }
            }
        };
    }

    Ok(best.unwrap())
}
